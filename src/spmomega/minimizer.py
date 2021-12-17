import numpy as np
import sys
from jax import grad
from scipy.optimize import minimize
from . import _xp

_conj_dispatch = {
    dict: lambda x: {k: v.conj() for k, v in x.items()},
    list: lambda x: [v.conj() for v in x]
}

def _conj(data):
    """Complex conjugate of a dict/list

    For a real-valued function f(x + yi),
    JAX computes (\frac{d f}{d x}, -\frac{d f}{d y}).
    This means that the value of the function decreases
    when we move along the conjugate of the grad computed by JAX.
    For convenience, this function takes the conjugate
    of keys in a given dict.

    Arguments:
    ----------
    data: dict/list
        Dict/list containing complex-valued tensors
    """
    return _conj_dispatch[type(data)](data)

def _grad_conj(func):
    """Thin wapper for JAX grad (takes the conjugate of the results)"""
    #print("Recompiled _grad_conj!")
    func_jax = grad(func)
    return lambda z: _conj(func_jax(z))

def _copy_to_dict(gfs, tensor_values):
    if isinstance(gfs, dict):
        gfs = gfs.values()
    for g in gfs:
        g.copy_to_dict(tensor_values)

def _copy_from_dict(gfs, tensor_values):
    if isinstance(gfs, dict):
        gfs = gfs.values()
    for g in gfs:
        g.copy_from_dict(tensor_values)

class Minimizer:
    def __init__(self, func, gfs):
        self._func = func
        self._gfs = gfs
        self._tensor_shapes = {}
        self._tot_size = 0

        x0 = {}
        _copy_to_dict(gfs, x0)
        self._tensor_names = []
        for k, v in x0.items():
            self._tensor_shapes[k] = v.shape
            self._tensor_names.append(k)
            self._tot_size += np.prod(v.shape)

    def _from_flatten_array(self, x):
        offset = 0
        tensor_values = {}
        for name in self._tensor_names:
            shape = self._tensor_shapes[name]
            size = np.prod(shape)
            real_part = x[offset:offset+size].reshape(shape)
            imag_part = x[offset+size:offset+2*size].reshape(shape)
            tensor_values[name] = real_part + 1J*imag_part
            offset += 2*size
        return tensor_values

    def _to_flatten_array(self, tensor_values):
        arrays = []
        for name in self._tensor_names:
            x = tensor_values[name]
            arrays.append(_xp.real(x).ravel())
            arrays.append(_xp.imag(x).ravel())
        return np.ascontiguousarray(_xp.concatenate(arrays))

    def run(self, verbosity=0, method='BFGS', **kwargs):
        cost0 = self._func(self._gfs)
        if verbosity > 0:
            print('Initial cost= ', cost0)

        def f_(param_tensors):
            _copy_from_dict(self._gfs, param_tensors)
            return self._func(self._gfs)
        
        f_jit = f_
        g_jit = _grad_conj(f_)

        x0 = {}
        _copy_to_dict(self._gfs, x0)
        x0 = self._to_flatten_array(x0)

        def f_scipy(x):
            r = f_jit(self._from_flatten_array(x))
            return r

        def g_scipy(x):
            g_dict = g_jit(self._from_flatten_array(x))
            # Somehow need to copy the array because
            # L-BFGS-B somehow requires the grad vector to be writable.
            r = np.array(self._to_flatten_array(g_dict), copy=True)
            # The current MPI code assumes that we differentiate
            # a function w.r.t the same data on all processes,
            # and the value of the function is returned after being allreduced.
            # This means that the value of the derivative must be allreduced as well.
            return r

        cost_hist = []
        def callback(x):
            tmp = f_scipy(x)
            if verbosity > 0:
                print(len(cost_hist), " cost ", tmp)
                sys.stdout.flush()
            cost_hist.append(tmp)
        
        if method == 'BFGS':
            if verbosity > 0:
                print("Using BFGS with iter_lim= ", kwargs['iter_lim'])
            res = minimize(f_scipy,
                x0,
                tol=1e-20,
                method='BFGS',
                jac=g_scipy,
                callback=callback,
                options={'maxiter': kwargs['iter_lim'], 'gtol': 1e-20}
            )
        elif method == 'L-BFGS-B':
            res = minimize(f_scipy,
                x0,
                method='L-BFGS-B',
                jac=g_scipy,
                callback=callback,
                options={'maxiter': kwargs['iter_lim'], 'xtol': 1e-8, 'maxcor': 100}
            )
        else:
            raise ValueError("Unknown method: " + method)

        if verbosity > 1:
            print(res)
        xres = self._from_flatten_array(res.x)
        _copy_from_dict(self._gfs, xres)
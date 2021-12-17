import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize
from scipy.interpolate import interp1d

import irbasis3
from . import _xp

from jax.config import config
config.update("jax_enable_x64", True)

import jax
from jax import numpy as jnp
from jax import grad, jit, jvp

from .quad import composite_leggauss, scale_quad

def __oversample(x):
    xmid = 0.5*(x[1:] + x[:-1])
    return np.unique(np.hstack((x, xmid)))

def _oversample(x, n=1):
    for i in range(n):
        x = __oversample(x)
    return x

def _second_deriv(x, y):
    """
    Compute second derivate on x[1:-1]
    """
    assert all(x[1:] > x[0:-1]), "x must be in increasing order!"
    rest_dim = y.shape[1:]
    y = y.reshape((x.size, -1))
    dx_forward  = (x[2:] - x[1:-1])[:,None]
    dx_backward = (x[1:-1] - x[0:-2])[:,None]
    y_ = y[1:-1,:]
    y_forward = y[2:,:]
    y_backward = y[:-2,:]
    ypp = 2*(dx_backward * y_forward + dx_forward * y_backward - (dx_forward+dx_backward) * y_)/ \
        (dx_backward**2 * dx_forward + dx_forward**2 * dx_backward)
    return ypp.reshape((x.size-2,) + rest_dim)

def _hvp(grad_f, x, v):
  """ Compute Hessian-vector product """
  return jvp(grad_f, [x], [v])[1]


class Interpolator(object):
    """
    Interpolate a matrixed-valued complex-valued function 
    along the first axix
    """
    def __init__(self, x, y):
        self._x = x
        self._y = y
    
    def __call__(self, x_new):
        raise NotImplementedError()

class LinearInterpolator(Interpolator):
    """
    Linear interpolation
    """
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._interpl_re = interp1d(x, y.real, axis=0)
        self._interpl_im = interp1d(x, y.imag, axis=0)
    
    def __call__(self, x_new):
        return self._interpl_re(x_new) + 1j*self._interpl_im(x_new)

class CubicSplineInterpolator(Interpolator):
    """
    Cubic spline interpolation
    """
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._interpl = _xp._cspline(x, y)
    
    def __call__(self, x_new):
        return self._interpl(x_new)

class MultiOrbitalSolver:
    def __init__(self, basis, interpl_cls=CubicSplineInterpolator, deg=10):
        """
        basis: irbasis3.FiniteTempBasis instance
        """
        self._basis = basis
        self._beta = basis.beta
        self._wmax = basis.wmax
        self._interpl_cls = interpl_cls

        roots = self._basis.v[-1].roots()
        #self._smpl_points = np.hstack((-self._wmax, roots, self._wmax))
        self._smpl_points = np.linspace(-self._wmax, self._wmax, 500)
        self._quad_points, self._quad_w = composite_leggauss(self._smpl_points, deg)

        self._prj_rho_l = basis.v(self._quad_points).T

        self._smpl_points_ovsmpl = _oversample(self._smpl_points, 3)
    
    @property
    def smpl_points(self):
        return self._smpl_points
    
    def interpolate(self, b_smpl, omegas):
        b = self._interpl_cls(self._smpl_points, b_smpl)(omegas)
        return _xp.einsum("wji,wjk->wik", b.conjugate(), b)
    
    def rho_l(self, b_smpl):
        """
        rho_{ij}(\omega_k) = (b_smpl^\dagger(\omega_k) b_smpl(\omega_k))_{ij}
        """
        b_quad = self._interpl_cls(self._smpl_points, b_smpl)(self._quad_points)
        rho_quad = _xp.einsum("wji,wjk->wik", b_quad.conjugate(), b_quad)
        return _xp.einsum('w,wij,wl->lij', self._quad_w, rho_quad, self._prj_rho_l)
    
    def sum_w(self, b_smpl):
        """
        Compute \int d w rho_{ij}(w)
        """
        b_quad = self._interpl_cls(self._smpl_points, b_smpl)(self._quad_points)
        return _xp.einsum('w,wij->ij', self._quad_w, b_quad)
    
    def g_l(self, b_smpl):
        """
        Compute g_{ij}(l) = - s_l * \rho_{ij}(l)
        """
        return -1.* _xp.einsum("l,l...->l...", self._basis.s, self.rho_l(b_smpl))
    
    def fit_g_l(self, g_l, reg_L2=1e-10, b_smpl0 = None, tol=1e-15, gtol=1e-15, maxiter=15000):
        assert g_l.ndim == 3
        assert g_l.shape[1] == g_l.shape[2]
        ncomp = g_l.shape[1]

        def _cost(x):
            x = x.reshape((self.smpl_points.size, ncomp, ncomp, 2))
            b_smpl = x[:,:,:,0] + 1j*x[:,:,:,1]
            #rho_smpl = _xp.einsum("wji,wjk->wik", b_smpl.conjugate(), b_smpl)
            rho_l_fit = self.rho_l(b_smpl)
            g_l_fit = - self._basis.s[:,None,None] * rho_l_fit

            rho_intpl = self.interpolate(b_smpl, self._smpl_points_ovsmpl)
            rho_pp = _second_deriv(self._smpl_points_ovsmpl, rho_intpl)
            r = 0.5 * _xp.norm(g_l - g_l_fit)**2 + reg_L2 * _xp.norm(rho_pp)**2
            return _xp._get_xp(x).log(r)
            
            #return 0.5 * _xp.norm(g_l - g_l_fit)**2 + reg_L2 * _xp.sum(_xp.abs(rho_pp))
            #return 0.5 * _xp.norm(g_l - g_l_fit)**2 + reg_L2 * _xp.norm(rho_l_fit)**2
            #return 0.5 * _xp.norm(g_l - g_l_fit)**2 + reg_L2 * _xp.sum(_xp.abs(rho_l_fit))

        def callback(x):
            r = _cost(x)
            print("cost", r, np.exp(r))

        cost_jit = jit(_cost)
        jac = jit(grad(_cost))
        if b_smpl0 is None:
            x0 = np.zeros((self.smpl_points.size,) + g_l.shape[1:] + (2,))
        else:
            if not np.iscomplexobj(b_smpl0):
                b_smpl0 = np.array(b_smpl0, dtype=np.complex128)
            x0 = b_smpl0.view(dtype=np.float64)

        grad_f = jit(grad(_cost))
        hessp = lambda x, v: _hvp(grad_f, x.ravel(), v.ravel())
        #r = minimize(cost_jit, x0, method='BFGS', jac=jac, tol=tol, options={'gtol': gtol, 'maxiter': maxiter}, callback=callback)
        #r = minimize(cost_jit, x0, method='BFGS', jac=jac, tol=tol, options={'gtol': gtol, 'maxiter': maxiter}, callback=callback)
        #print(hessp(x0, np.ones_like(x0)), x0.shape)
        r = minimize(cost_jit, x0, method='Newton-CG', jac=jac, hessp=hessp, callback=callback,
                options={'maxiter': maxiter, 'xtol': tol}
            )
        #print(r)
        x = r.x.reshape((self.smpl_points.size, ncomp, ncomp, 2))
        return x[:,:,:,0] + 1j*x[:,:,:,1]

class SpMSolver:
    def __init__(self, basis):
        """
        basis: irbasis3.FiniteTempBasis instance
        """
        self._basis = basis
        self._beta = basis.beta
        self._wmax = basis.wmax

        roots = self._basis.v[-1].roots()
        self._smpl_points = _oversample(np.hstack((-self._wmax, roots, self._wmax)))

    
    @property
    def smpl_points(self):
        return self._smpl_points
    
    def fit_g_l(self, g_l, reg_L1=1e-10, rho_l0 = None, tol=1e-15, gtol=1e-15, penalty=0.0):
        assert g_l.ndim == 1
        nl = self._basis.size
        prj_w = self._basis.v(self._smpl_points).T

        def _cost(x):
            xp = _xp._get_xp(x)
            rho_l_fit = x
            g_l_fit = - self._basis.s * rho_l_fit
            rho_w = prj_w @ rho_l_fit
            term0 =  0.5 * _xp.norm(g_l - g_l_fit)**2 + reg_L1 * _xp.sum(_xp.abs(rho_l_fit))
            term1 = penalty * _xp.sum(_xp.ReLU(-rho_w))
            return term0 + term1

        jac = jit(grad(_cost))
        if rho_l0 is None:
            x0 = np.zeros(self.nl)
        else:
            x0 = rho_l0
        r = minimize(_cost, x0, method="BFGS", jac=jac, tol=tol, options={'gtol': gtol})
        return r.x

    def fit_gtau(self, tau, gtau, reg_L1=1e-10, rho_l0 = None, tol=1e-15, gtol=1e-15, penalty=0.0):
        assert gtau.ndim == 1
        assert gtau.size == tau.size
        prj_w = self._basis.v(self._smpl_points).T
        prj_tau = -self._basis.u(tau).T * self._basis.s[None,:]

        def _cost(x):
            gtau_fit = prj_tau @ x
            rho_w = prj_w @ x
            term0 =  0.5 * _xp.norm(gtau - gtau_fit)**2 + reg_L1 * _xp.sum(_xp.abs(x))
            term1 = penalty * _xp.norm(_xp.ReLU(-rho_w))**2
            return _xp._get_xp(x).log(term0 + term1)

        jac = jit(grad(_cost))
        if rho_l0 is None:
            x0 = np.zeros(self.nl)
        else:
            x0 = rho_l0
        r = minimize(_cost, x0, method="BFGS", jac=jac, tol=tol, options={'gtol': gtol})
        return r.x

def _projector(basis, smpl_points):
    nl = basis.size
    mid_bins = 0.5*(smpl_points[0:-1] + smpl_points[1:])

    # Projector from sampled values to rho_l
    prj = np.zeros((nl, mid_bins.size))

    # Projector from sampled values to \int dw rho(w)
    prj_sum_rule = np.zeros(mid_bins.size)

    x_, w_ = leggauss(10)

    for s in range(mid_bins.size):
        x, w = scale_quad(x_, w_, smpl_points[s], smpl_points[s+1])
        prj_sum_rule[s] += smpl_points[s+1] - smpl_points[s]
        for l in range(nl):
            prj[l, s] = np.sum(basis.v[l](x) * w)
    
    return prj_sum_rule, prj

class SmoothSolver:
    def __init__(self, basis, deg=10, n_oversample=1):
        """
        basis: irbasis3.FiniteTempBasis instance
        """
        self._basis = basis
        self._beta = basis.beta
        self._wmax = basis.wmax

        roots = self._basis.v[-1].roots()
        self._smpl_points = _oversample(np.hstack((-self._wmax, roots, self._wmax)), n_oversample)
        self._quad_points, self._quad_w = composite_leggauss(self._smpl_points, deg)
        self._prj_sum, self._prj_rho_l = _projector(basis, self._smpl_points)
    
    @property
    def smpl_points(self):
        return self._smpl_points

    def rho_w(self, b_smpl):
        return _xp.einsum("wji,wjk->wik", b_smpl.conjugate(), b_smpl)
    
    def rho_l(self, b_smpl):
        """
        rho_{ij}(\omega_k) = (b_smpl^\dagger(\omega_k) b_smpl(\omega_k))_{ij}
        """
        print(self.rho_w(b_smpl).shape, self._prj_rho_l.shape)
        return _xp.einsum('wij,lw->lij', self.rho_w(b_smpl), self._prj_rho_l)
    
    def sum_w(self, b_smpl):
        """
        Compute \int d w rho_{ij}(w)
        """
        return _xp.einsum('wij,w->ij', self.rho_w(b_smpl), self._prj_sum)
    
    def g_l(self, b_smpl):
        """
        Compute g_{ij}(l) = - s_l * \rho_{ij}(l)
        """
        return -1.* _xp.einsum("l,l...->l...", self._basis.s, self.rho_l(b_smpl))
    
    def fit_gtau(self, tau, gtau, reg_L2=1e-10, b_smpl0 = None, tol=1e-15, gtol=1e-15, maxiter=15000, log=False):
        assert gtau.ndim == 3
        assert gtau.shape[1] == gtau.shape[2]
        ncomp = gtau.shape[1]

        uni_mesh = np.linspace(-1, 1, self._smpl_points.size)
        prj_tau = self.basis.u(tau).T

        def _cost(x):
            x = x.reshape((self.smpl_points.size, ncomp, ncomp, 2))
            b_smpl = x[:,:,:,0] + 1j*x[:,:,:,1]
            rho_w = self.rho_w(b_smpl)
            rho_l_fit = self.rho_l(b_smpl)
            gtau_fit = _xp.einsum('tl, lij->tij', prj_tau, -self._basis.s[:,None,None] * rho_l_fit)
            rho_pp = _second_deriv(uni_mesh, rho_w)
            r = 0.5 * _xp.norm(gtau - gtau_fit)**2 + reg_L2 * _xp.norm(rho_pp)**2
            return _xp._get_xp(x).log(r) if log else r

        def callback(x):
            r = _cost(x)
            print("cost", r, np.exp(r) if log else r)

        cost_jit = jit(_cost)
        jac = jit(grad(_cost))
        if b_smpl0 is None:
            x0 = np.zeros((self.smpl_points.size,) + gtau.shape[1:] + (2,))
        else:
            if not np.iscomplexobj(b_smpl0):
                b_smpl0 = np.array(b_smpl0, dtype=np.complex128)
            x0 = b_smpl0.view(dtype=np.float64)

        grad_f = jit(grad(_cost))
        hessp = lambda x, v: _hvp(grad_f, x.ravel(), v.ravel())
        r = minimize(cost_jit, x0, method='BFGS', jac=jac, tol=tol, options={'gtol': gtol, 'maxiter': maxiter}, callback=callback)
        #r = minimize(cost_jit, x0, method='BFGS', jac=jac, tol=tol, options={'gtol': gtol, 'maxiter': maxiter}, callback=callback)
        #print(hessp(x0, np.ones_like(x0)), x0.shape)
        #r = minimize(cost_jit, x0, method='Newton-CG', jac=jac, hessp=hessp, callback=callback,
                #options={'maxiter': maxiter, 'xtol': tol}
            #)
        #print(r)
        x = r.x.reshape((self.smpl_points.size, ncomp, ncomp, 2))
        return x[:,:,:,0] + 1j*x[:,:,:,1]

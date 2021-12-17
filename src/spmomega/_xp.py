# A common interface for numpy/jax arrays

import numpy as np
import jax.numpy as jnp
from scipy.interpolate import CubicSpline
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline


class _CubicSplineJAX:
    def __init__(self, x, y):
        self._intpl = InterpolatedUnivariateSpline(x, y.ravel())

    def __call__(self, x):
        return self._intpl(x)[:,None,None]


def _cspline(x, y):
    if _is_jax(x) or _is_jax(y):
        return _CubicSplineJAX(x, y)
    else:
        return CubicSpline(x, y)

def _get_xp(x):
    if isinstance(x, np.ndarray):
        return np
    elif _is_jax(x):
        return jnp
    else:
        raise ValueError("Unknown object: ", x)

def _is_jax(x):
    if hasattr(x, '__module__') and x.__module__[0:3] == 'jax':
        return True
    else:
        return False

def transpose(a, axes):
    return _get_xp(a).transpose(a, axes)


def einsum(subscripts, *operands):
    use_jnp = any(map(_is_jax, operands))
    if use_jnp:
        return jnp.einsum(subscripts, *operands)
    xp = _get_xp(operands[0])
    return xp.einsum(subscripts, *operands)


def copy(x):
    if _is_jax(x):
        return jnp.array(x, copy=True)
    return _get_xp(x).copy(x)

def real(x):
    if np.isscalar(x):
        return np.real(x)
    return _get_xp(x).real(x)

def imag(x):
    if np.isscalar(x):
        return np.imag(x)
    return _get_xp(x).imag(x)

def sum(x, axis=None):
    return _get_xp(x).sum(x, axis=axis)

def norm(x):
    return _get_xp(x).linalg.norm(x)

def concatenate(arrays, axis=0):
    use_jnp = any(map(_is_jax, arrays))
    if use_jnp:
        return jnp.concatenate(arrays, axis)
    else:
        return np.concatenate(arrays, axis)

def ascontiguousarray(x):
    return _get_xp(x).ascontiguousarray(x)

def asarray(x):
    return _get_xp(x).asarray(x)

def iscomplexobj(x):
    return _get_xp(x).iscomplexobj(x)

def bincount(x, weights=None):
    use_jnp = any(map(_is_jax, (x, weights)))
    if use_jnp:
        return jnp.bincount(x, weights)
    return _get_xp(x).bincount(x, weights)

def abs(x):
    return _get_xp(x).abs(x)

def log(x):
    return _get_xp(x).log(x)

def zeros_like(x):
    return _get_xp(x).zeros_like(x)

def ReLU(x):
    return _get_xp(x).maximum(x, 0)
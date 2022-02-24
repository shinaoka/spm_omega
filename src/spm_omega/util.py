import numpy as np
from numpy.polynomial.legendre import leggauss

from sparse_ir import FiniteTempBasis
from .quad import scale_quad


def __oversample(x: np.ndarray):
    xmid = 0.5*(x[1:] + x[:-1])
    return np.unique(np.hstack((x, xmid)))


def oversample(x: np.ndarray, n: int = 1):
    for i in range(n):
        x = __oversample(x)
    return x


def prj_w_to_l(
        basis: FiniteTempBasis,
        smpl_w: np.ndarray,
        deg: int) -> np.ndarray:
    """ Projector from rho(omega_i) to rho_l """
    prj_w_to_l = np.zeros((basis.size, smpl_w.size))
    x_, w_ = leggauss(deg)
    for s in range(smpl_w.size-1):
        x, w = scale_quad(x_, w_, smpl_w[s], smpl_w[s+1])
        dx = smpl_w[s+1] - smpl_w[s]
        f = (x - smpl_w[s])/dx
        g = (smpl_w[s+1] - x)/dx
        for idx_l in range(basis.size):
            prj_w_to_l[idx_l, s+1] += np.sum(w * basis.v[idx_l](x) * f)
            prj_w_to_l[idx_l, s] += np.sum(w * basis.v[idx_l](x) * g)
    return prj_w_to_l

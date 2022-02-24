import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.interpolate import interp1d

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



def regtarded_g(
        omegas_out: np.ndarray,
        rho_mesh,
        rho, epsabs=1e-5):
    """
    Compute retarded Green's function from spectral function

    omegas_out:
        real frequencies where G^R(w) is evaluated

    rho_mesh:
        real frequecies where rho(w) is given (in ascending order)

    rho:
        3D array representing rho(w).
        The first axis corresponds to the frequency.
    """
    assert rho.ndim == 3
    nso = rho.shape[1]
    res = np.zeros((omegas_out.size, nso, nso), dtype=np.complex128)

    interp = interp1d(rho_mesh, rho[:, 0, 0].real)

    for i in range(nso):
        for j in range(nso):
            interp = interp1d(
                        rho_mesh, rho[:, i, j].real,
                        fill_value=(0.0, 0.0), bounds_error=False)
            for k in range(omegas_out.size):
                res[k, i, j] += quad(
                                interp, rho_mesh[0], rho_mesh[-1],
                                weight="cauchy", wvar=omegas_out[k],
                                limit=1000, epsabs=epsabs)[0]
                res[k, i, j] += -1j*np.pi * interp(omegas_out[k])

            interp = interp1d(
                        rho_mesh, rho[:, i, j].imag,
                        fill_value=(0.0, 0.0), bounds_error=False)
            for k in range(omegas_out.size):
                res[k, i, j] += 1j*quad(
                                    interp,
                                    rho_mesh[0], rho_mesh[-1],
                                    weight="cauchy", wvar=omegas_out[k],
                                    limit=1000, epsabs=epsabs)[0]
                res[k, i, j] += np.pi * interp(omegas_out[k])
    return res

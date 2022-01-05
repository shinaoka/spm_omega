import numpy as np
from typing import Optional, Union
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize
from scipy.interpolate import interp1d

import irbasis3
from irbasis3 import FiniteTempBasis

from .quad import composite_leggauss, scale_quad

from admmsolver.objectivefunc import ConstrainedLeastSquares, L2Regularizer, SemiPositiveDefinitePenalty
from admmsolver.optimizer import SimpleOptimizer, Problem
from admmsolver.matrix import identity, DiagonalMatrix
from admmsolver.util import smooth_regularizer_coeff


def __oversample(x: np.ndarray):
    xmid = 0.5*(x[1:] + x[:-1])
    return np.unique(np.hstack((x, xmid)))

def _oversample(x: np.ndarray, n: int =1):
    for i in range(n):
        x = __oversample(x)
    return x

def _prj_w_to_l(
        basis: FiniteTempBasis,
        smpl_w: np.ndarray,
        deg: int
    ) -> np.ndarray:
    """ Projector from rho(omega_i) to rho_l """
    prj_w_to_l = np.zeros((basis.size, smpl_w.size))
    x_, w_ = leggauss(deg)
    for s in range(smpl_w.size-1):
        x, w = scale_quad(x_, w_, smpl_w[s], smpl_w[s+1])
        dx = smpl_w[s+1] - smpl_w[s]
        f = (x - smpl_w[s])/dx
        g = (smpl_w[s+1] - x)/dx
        for l in range(basis.size):
            prj_w_to_l[l, s+1] += np.sum(w * basis.v[l](x) * f)
            prj_w_to_l[l, s] += np.sum(w * basis.v[l](x) * g)
    return prj_w_to_l


class SpMOmega:
    def __init__(self, basis: FiniteTempBasis, deg: int=10) -> None:
        """
        """
        self._basis = basis
        self._beta = basis.beta
        self._wmax = basis.wmax

        self._smpl_w = _oversample(basis.v[-1].roots(), 1)
        self._prj_w = basis.v(self._smpl_w).T

        # From rho(omega_i) to \int domega rho(\omega)
        prj_sum = basis.s * (basis.u(0) + basis.u(self._beta))
        self._prj_sum = prj_sum.reshape((1,-1))

        # From sampled values to rho_l
        self._prj_w_to_l = _prj_w_to_l(basis, self._smpl_w, deg)

        # From sampled values to sum
        self._prj_sum_from_w = self._prj_sum @ self._prj_w_to_l


    @property
    def smpl_w(self) -> np.ndarray:
        return self._smpl_w
    
    def solve(
            self,
            gl: np.ndarray,
            alpha: float,
            mom: Optional[np.ndarray] = None,
            niter: int = 10000
        ) -> tuple[np.ndarray, dict]:
        """
        gl: 3d array
           Shape must be (nl, nf, nf),
           where nl is the number of IR basis functions
           and nf is the number of flavors.
        
        alpha: float
           L2 regularization parameter

        mom: 2d array
           First moment m_{ij} = int domega rho_{ij}(omega)
        
        niter: int
           Max number of iterations
        """
        assert gl.shape[0] == self._basis.size, \
            "shape of gl is not consistent with the basis!"
        nf = gl.shape[1] # Number of flavors
        smpl_w = self.smpl_w
        
        # Fitting matrix
        Aw = - self._basis.s[:,None] * self._prj_w_to_l
        A = np.einsum("lw,ij->liwj", Aw, np.identity(nf**2)).\
            reshape(Aw.shape[0]*nf**2, Aw.shape[1]*nf**2)
        
        # Sum-rule constraint
        V = np.einsum("xw,ij->xiwj",
            self._prj_sum_from_w, np.identity(nf**2))
        V = V.reshape(nf**2, smpl_w.size*nf**2)

        # Smoothness condition
        smooth_prj = np.einsum(
            "Ww,ij->Wiwj",
            smooth_regularizer_coeff(smpl_w),
            np.identity(nf**2))
        smooth_prj = smooth_prj.reshape(-1, smpl_w.size * nf**2)

        # Optimizer
        lstsq = ConstrainedLeastSquares(
          1.0, A, gl.ravel(),
          V, mom.ravel()
        )
        l2 = L2Regularizer(alpha, smooth_prj)
        nn = SemiPositiveDefinitePenalty((smpl_w.size, nf, nf), 0)
        equality_conditions = [
            (0, 1, identity(smpl_w.size*nf**2), identity(smpl_w.size*nf**2)),
            (0, 2, identity(smpl_w.size*nf**2), identity(smpl_w.size*nf**2)),
        ]
        problem = Problem([lstsq, l2, nn], equality_conditions)
        opt = SimpleOptimizer(problem, x0=None)

        # Run
        opt.solve(niter)

        info = {
            "optimizer": opt,
        }

        return opt.x[2], info
from stat import FILE_ATTRIBUTE_NOT_CONTENT_INDEXED
import numpy as np
from typing import Optional, Union, Tuple
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize
from scipy.interpolate import interp1d

import irbasis3
from irbasis3 import FiniteTempBasis

from .quad import composite_leggauss, scale_quad

from admmsolver.objectivefunc import LeastSquares, ConstrainedLeastSquares, L2Regularizer, SemiPositiveDefinitePenalty
from admmsolver.optimizer import SimpleOptimizer, Model
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

        # From rho_l to \int domega rho(\omega)
        prj_sum = basis.s * (basis.u(0) + basis.u(self._beta))
        self._prj_sum = prj_sum.reshape((1,-1))

        # From rho(omega_i) to rho_l
        self._prj_w_to_l = _prj_w_to_l(basis, self._smpl_w, deg)

        # From rho(omega_i) to \int domega rho(\omega)
        self._prj_sum_from_w = self._prj_sum @ self._prj_w_to_l


    @property
    def smpl_w(self) -> np.ndarray:
        return self._smpl_w
    
    def solve(
            self,
            gl: np.ndarray,
            alpha: float,
            moment: Optional[Union[np.ndarray,str]] = None,
            niter: int = 10000,
            spd: bool = True,
            interval_update_mu: int =100,
            fixed_boundary_condition: bool = True,
            initial_guess: Optional[np.ndarray] = None,
            rtol: float = 1e-10
        ) -> Tuple[np.ndarray, dict]:
        """
        gl: 3d array
           Shape must be (nl, nf, nf),
           where nl is the number of IR basis functions
           and nf is the number of flavors.
        
        alpha: float
           L2 regularization parameter

        moment: None, 2d array, or "auto"
           First moment m_{ij} = int domega rho_{ij}(omega)
           If moment=="auto", the moment will be estimated from the input data.
        
        niter: int
           Max number of iterations

        spd:
           Use semi-positive-definite condition

        fixed_boundary_condition:
           Set rho(omega) = 0 at the boundaries

        interval_update_mu:
          Interval for updating mu

        initial_guess:
          Initial guess

        rtol:
          Stopping condition. Check if all relative norms of all primal and dual residuals are smaller than rtol.
        """
        assert gl.ndim == 3
        assert gl.shape[0] == self._basis.size, \
            "shape of gl is not consistent with the basis!"
        assert gl.shape[1] == gl.shape[2], \
            "Invalid shape of gl"
        if isinstance(moment, np.ndarray):
            assert gl.shape[1:] == moment.shape
        
        nf = gl.shape[1] # Number of flavors
        smpl_w = self.smpl_w

        ###
        # Fitting matrix
        ###
        Aw = - self._basis.s[:,None] * self._prj_w_to_l
        A = np.einsum("lw,ij->liwj", Aw, np.identity(nf**2)).\
            reshape(Aw.shape[0]*nf**2, Aw.shape[1]*nf**2)
        
        ###
        # Various contraints V*x = W
        ###
        V = []
        W = []

        # Sum-rule constraint
        if moment == "auto":
            moment = np.einsum('l,lij->ij',
                 - (self._basis.u(0) + self._basis.u(self._beta)),
                 gl
            )
            moment = 0.5*(moment + moment.T.conj())

        if moment is not None:
            V_sumrule = np.einsum("xw,ij->xiwj",
                self._prj_sum_from_w, np.identity(nf**2))
            V.append(V_sumrule.reshape(nf**2, smpl_w.size*nf**2))
            W.append(moment.ravel())

        # Fixed-boundary condition
        if fixed_boundary_condition:
            V_bd = np.zeros((2, nf, nf) + (smpl_w.size, nf, nf))
            for i in range(nf):
               for j in range(nf):
                   V_bd[0,i,j,  0,i,j] = 1 # At omega = -wmax
                   V_bd[1,i,j, -1,i,j] = 1 # At omega = wmax
            V.append(V_bd.reshape(2*nf**2, smpl_w.size*nf**2))
            W.append(np.zeros(2*nf**2))

        ###
        #  Smoothness condition
        ###
        smooth_prj = np.einsum(
            "Ww,ij->Wiwj",
            smooth_regularizer_coeff(smpl_w),
            np.identity(nf**2))
        smooth_prj = smooth_prj.reshape(-1, smpl_w.size * nf**2)

        # Optimizer
        equality_conditions = [
            (0, 1, identity(smpl_w.size*nf**2), identity(smpl_w.size*nf**2)),
        ]
        if len(V) != 0:
            lstsq = ConstrainedLeastSquares(
              1.0, A, gl.ravel(),
              np.vstack(V),
              np.hstack(W)
            )
        else:
            lstsq = LeastSquares(1.0, A, gl.ravel())
        l2 = L2Regularizer(alpha, smooth_prj)
        terms = [lstsq, l2]
        if spd:
            nn = SemiPositiveDefinitePenalty((smpl_w.size, nf, nf), 0)
            equality_conditions.append((0, 2, identity(smpl_w.size*nf**2), identity(smpl_w.size*nf**2)))
            terms.append(nn)
        model = Model(terms, equality_conditions)
        if initial_guess is not None:
            initial_guess = len(terms) * [initial_guess]
        opt = SimpleOptimizer(model, x0=initial_guess)

        # Run
        opt.solve(niter, interval_update_mu=interval_update_mu, rtol=rtol)

        info = {
            "optimizer": opt,
            "lstsq": lstsq(opt.x[0])
        }

        return opt.x[0].reshape((self.smpl_w.size,) + gl.shape[1:]), info

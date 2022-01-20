from stat import FILE_ATTRIBUTE_NOT_CONTENT_INDEXED
import numpy as np
from typing import Optional, Union, Tuple, Dict
from numpy.polynomial.legendre import leggauss

from sparse_ir import FiniteTempBasis, KernelFFlat, MatsubaraSampling, TauSampling

from .quad import scale_quad

from admmsolver.objectivefunc import LeastSquares, ConstrainedLeastSquares
from admmsolver.objectivefunc import L2Regularizer, SemiPositiveDefinitePenalty, L1Regularizer
from admmsolver.optimizer import SimpleOptimizer, Model, EqualityCondition
from admmsolver.matrix import identity, PartialDiagonalMatrix, ScaledIdentityMatrix
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


class SpM:
    """
    Analytic continuation with L1 regularization
    """
    def __init__(
        self,
        beta: float,
        statistics: str,
        wmax: float,
        vsample: np.ndarray = None,
        tausample: np.ndarray = None,
        eps: float=1e-12,
        n_oversample: int =1
        ) -> None:
        """
        beta:
           Inverse temperature

        statistics:
           "F" or "B"

        wmax:
           Frequency cutoff

        vsample:
           Frequencies (odd for fermion, even for boson)

        tausample:
           times in [0, beta]

        eps:
           Cut-off value for singular values
        
        n_oversample:
           Over-sampling factor for mesh points in real frequencie
           used for imposing semi-positive definite conditions.
           The number of mesh points is increased by the factor of 2^n_oversample.
        """
        basis = FiniteTempBasis(KernelFFlat(beta*wmax), statistics, beta, eps)
        self._basis = basis
        self._beta = basis.beta
        self._vsample = None
        self._tausample = None
        if vsample is not None:
            stat_shift = {"F": 1, "B": 0}[statistics]
            assert all(vsample%2 == stat_shift)
            self._smpl_points = vsample
            self._smpl = MatsubaraSampling(basis, self._smpl_points)
        else:
            assert vsample is None
            self._smpl_points = tausample
            self._smpl = TauSampling(basis, self._smpl_points)
        self._wmax = basis.wmax

        # For spd condition
        self._smpl_w = _oversample(basis.v[-1].roots(), n_oversample)

        # From rho_l to \int domega rho(\omega)
        self._prj_l_to_sumw = basis.s * (basis.u(0) + basis.u(self._beta))


    @property
    def smpl_w(self) -> np.ndarray:
        return self._smpl_w

    def predict(self, rho_l: np.ndarray) -> np.ndarray:
        return self._smpl.evaluate(rho_l, axis=0)

    
    def solve(
            self,
            ginput: np.ndarray,
            alpha: float,
            moment: Optional[np.ndarray] = None,
            niter: int = 10000,
            spd: bool = True,
            interval_update_mu: int =100,
            initial_guess: Optional[np.ndarray] = None,
            rtol: float = 1e-10
        ) -> Dict:
        """
        ginput: 3d array of shape (nsmpl, nf, nf),
           where nv is the number of Matsubara frequeicies/times,
           and nf is the number of flavors.
        
        alpha: float
           L2 regularization parameter

        moment: None or 2d array
           First moment m_{ij} = int domega rho_{ij}(omega)
        
        niter: int
           Max number of iterations

        spd:
           Use semi-positive-definite condition

        interval_update_mu:
          Interval for updating mu

        rtol:
          Stopping condition.
          Check if all relative norms of all primal and dual residuals are smaller than rtol.
        """
        assert ginput.ndim == 3
        assert ginput.shape[0] == self._smpl_points.size, \
            "shape of ginput is not consistent with sampling frequencies!"
        assert ginput.shape[1] == ginput.shape[2], \
            "Invalid shape of ginput"
        if isinstance(moment, np.ndarray):
            assert ginput.shape[1:] == moment.shape
        
        nf = ginput.shape[1] # Number of flavors
        nl = self._basis.size
        x_size = nl*nf**2

        terms = []
        equality_conditions = []

        ###
        # Fitting matrix
        ###
        # FIXME: Use PartialDiagonalMatrix
        Aw = - self._basis.s[None,:] * self._smpl.matrix.a
        A = np.einsum("Ww,ij->Wiwj", Aw, np.identity(nf**2)).\
            reshape(Aw.shape[0]*nf**2, Aw.shape[1]*nf**2)
        
        ###
        # Various contraints V*x = W
        ###
        V = []
        W = []

        # Sum-rule constraint
        if moment is not None:
            V_sumrule = np.einsum("xw,ij->xiwj",
                self._prj_l_to_sumw, np.identity(nf**2))
            V.append(V_sumrule.reshape(nf**2, nl.size*nf**2))
            W.append(moment.ravel())

        if len(V) != 0:
            lstsq = ConstrainedLeastSquares(
                    1.0, A, ginput.ravel(),
                    np.vstack(V),
                    np.hstack(W)
                )
        else:
            lstsq = LeastSquares(1.0, A, ginput.ravel())
        terms.append(lstsq)

        ###
        # L1
        ###
        terms.append(L1Regularizer(alpha, x_size))
        equality_conditions.append(EqualityCondition(0, 1, identity(x_size), identity(x_size)))

        ###
        # Semi-positive definite condition
        ###
        if spd:
            # x0: least squares, x2: sampling real frequencies
            terms.append(SemiPositiveDefinitePenalty((self._smpl_w.size, nf, nf), 0))
            e20 = PartialDiagonalMatrix(
                        self._basis.v(self._smpl_w).T,
                        (nf, nf)
                    )
            e02 = ScaledIdentityMatrix(self._smpl_w.size*nf**2, 1.0)
            eq = EqualityCondition(0, len(terms)-1, e20, e02)
            equality_conditions.append(eq)
    
        model = Model(terms, equality_conditions)
        opt = SimpleOptimizer(model)

        # Run
        opt.solve(niter, interval_update_mu=interval_update_mu, rtol=rtol)

        info = {
            "optimizer": opt,
            "lstsq": lstsq(opt.x[0])
        }

        return opt.x[0].reshape((-1,) + ginput.shape[1:]), info



class SpMSmooth:
    """
    Analytic continuation with smooth conditions
    """
    def __init__(
        self,
        beta: float,
        statistics: str,
        wmax: float,
        vsample: np.ndarray = None,
        tausample: np.ndarray = None,
        deg: int=10,
        eps: float=1e-7) -> None:
        """
        """
        basis = FiniteTempBasis(KernelFFlat(beta*wmax), statistics, beta, eps)
        self._basis = basis
        self._beta = basis.beta
        self._vsample = None
        self._tausample = None
        if vsample is not None:
            stat_shift = {"F": 1, "B": 0}[statistics]
            assert all(vsample%2 == stat_shift)
            self._smpl_points = vsample
            self._smpl = MatsubaraSampling(basis, self._smpl_points)
        else:
            assert vsample is None
            self._smpl_points = tausample
            self._smpl = TauSampling(basis, self._smpl_points)
        self._wmax = basis.wmax

        self._smpl_w = _oversample(basis.v[-1].roots(), 1)

        # From rho(omega_i) to rho_l
        prj_w_to_l = _prj_w_to_l(basis, self._smpl_w, deg)

        # From rho(omega_i) to sampling points
        self._prj_w_to_smpl = self._smpl.matrix.a @ (-basis.s[:,None] * prj_w_to_l)

        # From rho(omega_i) to \int domega rho(\omega)
        prj_sum = basis.s * (basis.u(0) + basis.u(self._beta))
        self._prj_w_to_sumw = prj_sum @ prj_w_to_l


    @property
    def smpl_w(self) -> np.ndarray:
        return self._smpl_w

    def predict(self, rhow: np.ndarray) -> np.ndarray:
        return np.einsum('Ww,wij->Wij', self._prj_w_to_smpl, rhow, optimize=True)

    
    def solve(
            self,
            ginput: np.ndarray,
            alpha: float,
            moment: Optional[np.ndarray] = None,
            niter: int = 10000,
            spd: bool = True,
            interval_update_mu: int =100,
            fixed_boundary_condition: bool = True,
            initial_guess: Optional[np.ndarray] = None,
            rtol: float = 1e-10
        ) -> Dict:
        """
        ginput: 3d array of shape (nsmpl, nf, nf),
           where nv is the number of Matsubara frequeicies/times,
           and nf is the number of flavors.
        
        alpha: float
           L2 regularization parameter

        moment: None or 2d array
           First moment m_{ij} = int domega rho_{ij}(omega)
        
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
          Stopping condition.
          Check if all relative norms of all primal and dual residuals are smaller than rtol.
        """
        assert ginput.ndim == 3
        assert ginput.shape[0] == self._smpl_points.size, \
            "shape of ginput is not consistent with sampling frequencies!"
        assert ginput.shape[1] == ginput.shape[2], \
            "Invalid shape of ginput"
        if isinstance(moment, np.ndarray):
            assert ginput.shape[1:] == moment.shape
        
        nf = ginput.shape[1] # Number of flavors
        smpl_w = self.smpl_w
        x_size = smpl_w.size*nf**2

        ###
        # Fitting matrix
        ###
        Aw = self._prj_w_to_smpl
        A = np.einsum("Ww,ij->Wiwj", Aw, np.identity(nf**2)).\
            reshape(Aw.shape[0]*nf**2, Aw.shape[1]*nf**2)
        
        ###
        # Various contraints V*x = W
        ###
        V = []
        W = []

        # Sum-rule constraint
        if moment is not None:
            V_sumrule = np.einsum("xw,ij->xiwj",
                self._prj_w_to_sumw, np.identity(nf**2))
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
        smooth_prj_ = smooth_regularizer_coeff(smpl_w)
        smooth_prj = np.einsum(
            "Ww,ij->Wiwj", smooth_prj_, np.identity(nf**2))
        smooth_prj = smooth_prj.reshape(-1, x_size)

        # Optimizer
        equality_conditions = [
            (0, 1, identity(x_size), identity(x_size)),
        ]
        if len(V) != 0:
            lstsq = ConstrainedLeastSquares(
              1.0, A, ginput.ravel(),
              np.vstack(V),
              np.hstack(W)
            )
        else:
            lstsq = LeastSquares(1.0, A, ginput.ravel())
        l2 = L2Regularizer(alpha, smooth_prj)
        terms = [lstsq, l2]
        if spd:
            nn = SemiPositiveDefinitePenalty((smpl_w.size, nf, nf), 0)
            equality_conditions.append((0, 2, identity(x_size), identity(x_size)))
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

        return opt.x[0].reshape((-1,) + ginput.shape[1:]), info

SpMOmegaSmpl = SpMSmooth
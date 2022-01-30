# Copyright (C) 2021-2022 Hiroshi Shinaoka and others
# SPDX-License-Identifier: MIT
import numpy as np
from typing import Optional, Union, Tuple, Dict, List
from numpy.polynomial.legendre import leggauss

from sparse_ir import FiniteTempBasis, KernelFFlat, MatsubaraSampling, TauSampling

from .quad import scale_quad

from admmsolver.objectivefunc import LeastSquares, ConstrainedLeastSquares, ObjectiveFunctionBase
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
            times in [0, beta] (in ascending order)

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
        elif tausample is not None:
            assert vsample is None
            self._smpl_points = tausample
            self._smpl = TauSampling(basis, self._smpl_points)
        else:
            raise ValueError("One of vsample and tausample must be given!")
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
        ):
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
            V.append(V_sumrule.reshape(nf**2, nl*nf**2))
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


class SpMSmoothBase:
    r"""
    Base solver for analytic continuation with smooth conditions

    The spectral representation is given by
        G_ij(iv) = \int d\omga  K(iv, \omega) \rho_ij(\omega) + P(iv) c_{ij},
    where rho_{ij}(\omega) is a positive semi-definite matrix at a given \omega.
    The second term of the RHS represents augmentation, such as a constant term in tau/frequency.

    In the time, the spectral representation reads
        G_ij(\tau) = \int d\omega  K(\tau, \omega) \rho_ij(\omega) + P(\tau) c_{ij}.
    """
    def __init__(
        self,
        beta: float,
        smpl_w: np.ndarray,
        K: np.ndarray,
        P_aug: Optional[np.ndarray] = None,
        P_sum: Optional[np.ndarray] = None
        ) -> None:
        """
        beta: inverse temperature
        """
        assert isinstance(smpl_w, np.ndarray) and smpl_w.ndim == 1
        assert all(smpl_w[0:-1] < smpl_w[1:])
        if P_aug is not None:
            assert isinstance(P_aug, np.ndarray) and P_aug.ndim == 1

        self._beta = beta
        self._smpl_w = smpl_w

        self._K = K
        self._P_aug = P_aug
        self._P_sum = P_sum

    @property
    def smpl_w(self):
        return self._smpl_w

    def predict(
        self,
        x: np.ndarray
        ) -> np.ndarray:
        """
        Evaluate Green's function
        """
        if self._P_aug is not None:
            rho = x[0:self._smpl_w.size, :, :], x[self._smpl_w.size, :, :]
        else:
            rho = x, None
        r = np.einsum('Ww,wij->Wij', self._K, rho[0], optimize=True)
        if self._P_aug is not None:
            assert rho[1] is not None
            r += np.einsum('Ww,wij->Wij', self._P_aug, rho[1], optimize=True)
        return r


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
        ):
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

        initial_guess:
            Initial guess

        rtol:
            Stopping condition.
            Check if all relative norms of all primal and dual residuals are smaller than rtol.
        """
        assert ginput.ndim == 3
        assert ginput.shape[0] == self._K.shape[0], \
            "shape of ginput is not consistent with sampling frequencies!"
        assert ginput.shape[1] == ginput.shape[2], \
            "Invalid shape of ginput"
        if isinstance(moment, np.ndarray):
            assert ginput.shape[1:] == moment.shape

        nf = ginput.shape[1] # type: int
        nsmpl = self._K.shape[0] # type: int
        nsmpl_w = self._smpl_w.shape[0] # type: int

        x_size = nsmpl_w * nf**2 # type: int
        if self._P_aug is not None:
            x_size += nf**2

        # x: (N, nsmpl_w)
        # y: (N, 1)
        # np.hstack((x,y)): (N, nsmpl_w+1)
        def add_zero_column(x):
            assert x.shape[1] == nsmpl_w
            N = x.shape[0]
            return np.hstack([x, np.zeros((N, 1)) ])

        ###
        # Fitting matrix
        ###
        K_ = self._K
        if self._P_aug is not None:
            # K_: (nsmpl, nsmpl_w)
            # new K: (nsmpl, nsmpl_w+1)
            K_ = np.hstack((K_, self._P_aug[:,None]))
        K = PartialDiagonalMatrix(K_, (nf, nf))

        ###
        # Various contraints V*x = W on the normal part
        ###
        V = []
        W = []

        # Sum-rule constraint
        if moment is not None:
            assert self._P_sum is not None
            V_sumrule = np.einsum("xw,ij->xiwj",
                self._P_sum, np.identity(nf**2))
            V.append(V_sumrule.reshape(nf**2, nsmpl * nf**2))
            W.append(moment.ravel())

        if self._P_aug is not None:
            V = list(map(add_zero_column, V))

        ###
        #  Smoothness condition
        ###
        smooth_prj_mat = smooth_regularizer_coeff(self._smpl_w) # type: np.ndarray
        if self._P_aug is not None:
            smooth_prj_mat = add_zero_column(smooth_prj_mat)
        smooth_prj = PartialDiagonalMatrix(smooth_prj_mat, (nf, nf))

        # Optimizer
        equality_conditions = [
            EqualityCondition(0, 1, identity(x_size), identity(x_size)),
        ] # type: list[EqualityCondition]
        if len(V) != 0:
            lstsq = ConstrainedLeastSquares(
                1.0, K, ginput.ravel(),
                np.vstack(V),
                np.hstack(W)
            )
        else:
            lstsq = LeastSquares(1.0, K, ginput.ravel())
        l2 = L2Regularizer(alpha, smooth_prj)
        terms = [lstsq, l2] # type: List[ObjectiveFunctionBase]
        if spd:
            nn = SemiPositiveDefinitePenalty((nsmpl_w, nf, nf), 0)
            equality_conditions.append(
                    EqualityCondition(
                        0, 2,
                        ScaledIdentityMatrix((nsmpl_w*nf**2, x_size), 1.0),
                        ScaledIdentityMatrix(nsmpl_w*nf**2, 1.0)
                    )
                )
            terms.append(nn)
        model = Model(terms, equality_conditions)
        x0 = None
        if initial_guess is not None:
            x0 = len(terms) * [initial_guess]
        opt = SimpleOptimizer(model, x0=x0)

        # Run
        opt.solve(niter, interval_update_mu=interval_update_mu, rtol=rtol)

        info = {
            "optimizer": opt,
            "lstsq": lstsq(opt.x[0])
        }

        x = opt.x[0].reshape((-1,) + ginput.shape[1:])
        return x, info



class SpMSmooth(SpMSmoothBase):
    r"""
    Analytic continuation with smooth conditions

    The spectral representation is given by
        G_ij(iv) = \int d\omga  K(iv, \omega) \rho_ij(\omega) + P(iv) c_{ij},
    where rho_{ij}(\omega) is a positive semi-definite matrix at a given \omega.
    The second term of the RHS represents augmentation, such as a constant term in tau/frequency.

    In the time, the spectral representation reads
        G_ij(\tau) = \int d\omega  K(\tau, \omega) \rho_ij(\omega) + P(\tau) c_{ij}.
    """
    def __init__(
            self,
            beta: float,
            statistics: str,
            wmax: float,
            vsample: np.ndarray = None,
            tausample: np.ndarray = None,
            deg: int=10,
            eps: float=1e-7,
            hartree_fock_term: bool = False,
            omega0_term: bool = False,
        ) -> None:

        basis = FiniteTempBasis(KernelFFlat(beta*wmax), statistics, beta, eps)
        self._basis = basis

        assert not (vsample is not None and tausample is not None), "vsample and tausample are mutually exclusive." # Exclusive parameters
        assert not (hartree_fock_term and omega0_term) # Exclusive parameters

        if vsample is not None:
            stat_shift = {"F": 1, "B": 0}[statistics]
            assert all(vsample%2 == stat_shift)
            self._smpl_points = vsample
            self._smpl = MatsubaraSampling(basis, self._smpl_points)
        elif tausample is not None:
            self._smpl_points = tausample
            self._smpl = TauSampling(basis, self._smpl_points)
        self._wmax = basis.wmax

        smpl_w = _oversample(np.hstack((-wmax, basis.v[-1].roots(), wmax)), 1)

        # From rho(omega_i) to rho_l
        prj_w_to_l = _prj_w_to_l(basis, smpl_w, deg)

        # From rho(omega_i) to sampling points
        K = self._smpl.matrix.a @ (-basis.s[:,None] * prj_w_to_l)

        # From rho(omega_i) to \int domega rho(\omega)
        prj_sum = basis.s * (basis.u(0) + basis.u(beta))
        P_sum = prj_sum @ prj_w_to_l

        P_aug = None
        if hartree_fock_term:
            assert vsample is not None
            P_aug = np.ones(self._smpl_points.size)
        elif omega0_term:
            assert vsample is not None
            assert statistics == "B"
            pos = np.where(self._smpl_points==0)[0]
            assert pos.size == 1, pos
            P_aug = np.zeros(self._smpl_points.size)
            P_aug[pos] = 1

        super().__init__(beta, smpl_w, K, P_sum=P_sum, P_aug=P_aug)
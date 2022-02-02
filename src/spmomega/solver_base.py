# Copyright (C) 2021-2022 Hiroshi Shinaoka and others
# SPDX-License-Identifier: MIT
import numpy as np
from scipy.linalg import block_diag
from typing import Optional, Union, Tuple, Dict, List, cast

from sparse_ir import FiniteTempBasis, MatsubaraSampling, TauSampling
from sparse_ir.composite import CompositeBasis

from admmsolver.objectivefunc import LeastSquares, ConstrainedLeastSquares, ObjectiveFunctionBase
from admmsolver.objectivefunc import L2Regularizer, SemiPositiveDefinitePenalty, L1Regularizer
from admmsolver.optimizer import SimpleOptimizer, Model, EqualityCondition
from admmsolver.matrix import identity, PartialDiagonalMatrix, ScaledIdentityMatrix, MatrixBase, DenseMatrix, DiagonalMatrix

from .util import oversample


class SingularTermModel(object):
    r"""
    Green's function may contain contributions that are not compactly
    representated by IR basis such as a Hartree-Fock term or
    a zero-energy-model excitation for bosons.
    SingularTermModel is a base class for models that models such terms.
    """
    def __init__(self, sampling_points: np.ndarray, matrix: np.ndarray) -> None:
        assert all(sampling_points[:-1] < sampling_points[1:])
        assert matrix.ndim == sampling_points.ndim == 1
        assert matrix.size == sampling_points.size

        self._sampling_points = sampling_points
        self._matrix = matrix

    def evaluate(self, coeff: np.ndarray)->np.ndarray:
        assert isinstance(coeff, np.ndarray) and coeff.ndim == 2
        return np.einsum('w,ij->wij', self._matrix, coeff)

    @property
    def matrix(self)->np.ndarray:
        return self._matrix


class MatsubaraHartreeFockModel(SingularTermModel):
    """
    Modeling a Hartree-Fock term in frequency
    """
    def __init__(self, sampling_points: np.ndarray) -> None:
        super().__init__(
                sampling_points,
                np.ones((self._sampling_points.size,))
            )

class MatsubaraZeroFrequencyPoleModel(SingularTermModel):
    r"""
    Modeling a pole at omega = 0, rho(omega) = coeff * \delta(\omega)
    """
    def __init__(self, sampling_points: np.ndarray) -> None:
        matrix = np.zeros(self._sampling_points.size)
        zero_pos = np.where(sampling_points==0)[0][0]
        matrix[zero_pos] = 1
        super().__init__(sampling_points, matrix)



class AnaContBase:
    r"""
    Base solver for analytic continuation with smooth conditions

    The spectral representation is given by
        G_ij(iv) = \int d\omga  K(iv, \omega) \rho_ij(\omega) + P(iv) c_{ij},
    where rho_{ij}(\omega) is a positive semi-definite matrix at a given \omega.
    The second term of the RHS represents a constant term in tau/frequency (optionally).

    We model this analytic-continuation problem as
        G_{s,ij} = \sum_l U_{sl} * rho_{l,ij} + |\sum_l B_{tl} * rho_{l,ij}|_p^p +
                        alpha * \sum_n C_{sn} x'_{n,ij},
    where s denotes an imaginary time or an imaginary frequency, i and j denote spin orbitals,
    alpha a regularization.
    The expansion coefficients in IR are given by
        rho_{l,ij} = \sum_m A_{l,m} x_{m,ij}.
    The first and second terms of the RHS denote a normal component and a singular (augment) component, respectively.
    The sum rules reads
        \sum_l S_{k,l} rho_{l,ij} = T_{l,ij}.

    The singular component represents an additonal contribution that is not complactly (or conveniently)
    represented in IR (e.g., Hartree-Fock term, a delta peak at omega=0 for bosons).
    """
    def __init__(
            self,
            #basis: Union[FiniteTempBasis, CompositeBasis],
            sampling: Union[TauSampling, MatsubaraSampling],
            a: MatrixBase,
            b: MatrixBase,
            c: MatrixBase,
            reg_type: str = "L2",
            sum_rule: Optional[Tuple[np.ndarray,np.ndarray]] = None,
        ) -> None:
        r"""
        basis:
            If basis is a CompositeBasis instance,

        sampling:
            Sampling object associated with a FiniteTempBasis instance or CompositeBasis instance,
            If a CompositeBasis instance is given,
            the first/second component is regarded as a normal/singular component.

        sampling:
            Sampling object

        a:
            Transformation matrix `A` from the normal component `x` to IR

        b:
            Transformation matrix `B` from the normal component `x` to the space
            where L1/L2 regularation is imposed.

        c:
            Transformation matrix `C` to sampling real-frequencies
            where L1/L2 regularation is imposed on the fitting.

        reg_type: str
            `L1` or `L2`

        sum_rule: (S, T)
            Sum rule for the normal part.
            S, T are 2D and 1D array, respectively
        """
        assert a.shape[1] == b.shape[1], f"{a.shape} {b.shape}"
        assert reg_type in ["L1", "L2"]
        assert type(sampling) in [MatsubaraSampling, TauSampling]

        self._sampling = sampling
        self._basis = sampling.basis
        assert isinstance(self._basis, FiniteTempBasis) or \
            (isinstance(self._basis, CompositeBasis) and len(self._basis.bases) > 1)

        self._beta = self._basis.beta
        self._bases = self._basis.bases if isinstance(self._basis, CompositeBasis) else [self._basis]
        self._is_augmented = isinstance(self._basis, CompositeBasis) and len(self._basis.bases) > 1

        self._a = a
        self._b = b
        self._c = c
        self._reg_type = reg_type
        self._sum_rule = sum_rule

        wmax = self._bases[0].wmax
        self._smpl_real_w = oversample(np.hstack((-wmax, self._bases[0].v[-1].roots(), wmax)), 1)

    def predict(
            self,
            x: np.ndarray
        ) -> np.ndarray:
        """
        Evaluate Green's function

        x:
            x or np.vstack((x, x'))
        """
        if self._is_augmented:
            assert len(x) == 2
            res = self._sampling.evaluate(np.vstack((self._a @ x[0], x[1])), axis=0)
        else:
            res = self._sampling.evaluate(self._a @ x, axis=0)
        assert res.ndim == 3
        return cast(np.ndarray, res)


    def solve(
            self,
            ginput: np.ndarray,
            alpha: float,
            niter: int = 10000,
            spd: bool = True,
            interval_update_mu: int =100,
            rtol: float = 1e-10
        ):
        r"""
        ginput: 3d array of shape (nsmpl, nf, nf),
            where nv is the number of Matsubara frequeicies/times,
            and nf is the number of flavors.

        input_type:
            'tau' or 'matsubara'

        alpha: float
            L2 regularization parameter

        niter: int
            Max number of iterations

        spd:
            Use semi-positive-definite condition for normal part

        interval_update_mu:
            Interval for updating mu

        rtol:
            Stopping condition.
            Check if all relative norms of all primal and dual residuals are smaller than rtol.
        """
        assert ginput.ndim == 3
        assert ginput.shape[0] == self._sampling.sampling_points.size, \
            "shape of ginput is not consistent with sampling frequencies!"
        assert ginput.shape[1] == ginput.shape[2], \
            "Invalid shape of ginput"
        if isinstance(self._sum_rule, np.ndarray):
            assert ginput.shape[1:] == self._sum_rule.shape

        assert ginput.shape[0] == self._sampling.sampling_points.size
        nf = ginput.shape[1] # type: int
        nparam_normal = self._a.shape[1]
        nparam_w = nparam_normal//(nf**2)
        nparam_singular = 0 if not self._is_augmented else nf**2
        x_size = nparam_normal + nparam_singular

        # a: (N, M)
        # y: (N, 1)
        # np.hstack((a,y)): (N, M+1)
        def add_zero_column(a):
            N = a.shape[0]
            y = np.zeros((N, 1))
            print("debug", a.shape, y.shape)
            return np.hstack((a, y))

        ###
        # Fitting matrix U * A
        ###
        if self._is_augmented:
            ua_ = self._sampling.matrix.a @ \
                block_diag(self._a.asmatrix(), np.identity(self._basis.bases[1].size)) # type: np.ndarray
        else:
            ua_ = self._sampling.matrix.a @ self._a.asmatrix() # type: np.ndarray
        ua_full = PartialDiagonalMatrix(ua_, (nf, nf))

        ###
        # Various contraints V*x = W on the normal part
        ###
        V = [] # List[np.ndarray]
        W = [] # List[np.ndarray]

        # Sum-rule constraint
        if self._sum_rule is not None:
            a, b = self._sum_rule
            assert a.ndim == 2 and b.ndim == 1
            V.append(a)
            W.append(b)

        # Extend the shape of V to include singular component
        V = list(map(add_zero_column, V))

        ###
        #  Regularization
        ###
        assert isinstance(self._b, DenseMatrix)
        b_ = self._b.asmatrix()
        if self._is_augmented:
            b_ =  add_zero_column(b_)
        b_full = PartialDiagonalMatrix(b_, (nf, nf))

        # Optimizer
        equality_conditions = [
            EqualityCondition(0, 1, identity(x_size), identity(x_size)),
        ] # type: list[EqualityCondition]
        if len(V) != 0:
            lstsq = ConstrainedLeastSquares(
                1.0, ua_full, ginput.ravel(),
                np.vstack(V),
                np.hstack(W)
            )
        else:
            lstsq = LeastSquares(1.0, ua_full, ginput.ravel())

        assert b_full is not None
        reg = {"L2": L2Regularizer, "L1": L1Regularizer}[self._reg_type](alpha, b_full)
        terms = [lstsq, reg] # type: List[ObjectiveFunctionBase]
        if spd:
            # Semi-positive definite condition on normal component
            nsmpl_w = self._smpl_real_w.size #type: int
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
        opt = SimpleOptimizer(model)

        # Run
        opt.solve(niter, interval_update_mu=interval_update_mu, rtol=rtol)

        info = {
            "optimizer": opt,
            "lstsq": lstsq(opt.x[0])
        }

        x = opt.x[0].reshape((-1,) + ginput.shape[1:])
        return x, info
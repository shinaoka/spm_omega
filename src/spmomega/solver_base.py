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
from enum import Enum


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


class InputType(Enum):
    TIME = 0
    FREQ = 1


class AnaContBase:
    r"""
    Base solver for analytic continuation with smooth conditions

    We model this analytic-continuation problem as
        G_{s,ij} = - \sum_l S_l U_{sl} * rho_{l,ij} + alpha * |\sum_l B_{tl} * rho_{l,ij}|_p^p +
                        \sum_n C_{sn} x'_{n,ij},
    where
        s: an imaginary time or an imaginary frequency,
        i and j: spin orbitals,
        alpha: a regularization,
        S_l: singular values.

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
            sampling: Union[TauSampling, MatsubaraSampling],
            a: MatrixBase,
            b: MatrixBase,
            c: MatrixBase,
            reg_type: str = "L2",
            sum_rule: Optional[Tuple[np.ndarray,np.ndarray]] = None,
        ) -> None:
        r"""
        basis:
            If basis is a `CompositeBasis` instance,

        sampling:
            Sampling object associated with a `FiniteTempBasis` instance or `CompositeBasis` instance,
            If a CompositeBasis instance is given,
            the first/second component is regarded as a normal/singular component.

        a:
            Transformation matrix `A` from the normal component `x` to rho_l

        b:
            Transformation matrix `B` from the normal component `x` to the space
            where L1/L2 regularation is imposed.

        c:
            Transformation matrix `C` from the normal component `x` to the space
            where the SPD condition is imposed. This matrix must be consistent with `smpl_reqal_w`.

        reg_type: str
            `L1` or `L2`

        sum_rule: (S, T)
            Sum rule for the normal component.
            S, T are 2D and 1D array, respectively
        """
        assert a.shape[1] == b.shape[1], f"{a.shape} {b.shape}"
        assert reg_type in ["L1", "L2"]
        assert type(sampling) in [MatsubaraSampling, TauSampling]

        self._sampling = sampling
        self._basis = sampling.basis
        assert isinstance(self._basis, FiniteTempBasis) or \
            (isinstance(self._basis, CompositeBasis) and len(self._basis.bases) > 1)

        self._bases = self._basis.bases if isinstance(self._basis, CompositeBasis) else [self._basis]
        self._beta = cast(float, self._bases[0]) # type: float
        self._is_augmented = isinstance(self._basis, CompositeBasis) and len(self._basis.bases) > 1 # type: bool

        self._a = a
        self._b = b
        self._c = c
        self._reg_type = reg_type
        self._sum_rule = sum_rule

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
            Use semi-positive-definite condition for normal component

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
        nparam_normal = self._a.shape[1] * nf**2
        nparam_singular = 0 if not self._is_augmented else nf**2
        x_size = nparam_normal + nparam_singular


        ###
        # Fitting matrix
        # Let (T0, T1) = self._sampling.matrix.a.
        #
        # Normal component:
        #  g_smpl = - (T0 @ S @ A0) x
        #
        # Singular component:
        #  g_smpl = T1 x'
        ###
        T0 = self._sampling.matrix.a[:, 0:self._bases[0].size]
        T0S = T0.copy()
        T0S[:, 0:self._bases[0].size] *= -self._bases[0].s[None,:]
        T0SA0 = T0S @ self._a.asmatrix() # type: np.ndarray
        if self._is_augmented:
            T1 = self._sampling.matrix.a[:, self._bases[0].size:]
            tmp = np.hstack((T0SA0, T1))
            #sua_ = self._sampling.matrix.a @ \
                #block_diag(self._a.asmatrix(), np.identity(self._basis.bases[1].size)) # type: np.ndarray
            #sua_ = su_ @ self._a.asmatrix() # type: np.ndarray
            #sua__ = su_ @ self._a.asmatrix() # type: np.ndarray
            sua_full = PartialDiagonalMatrix(tmp, (nf, nf))
        else:
            sua_full = PartialDiagonalMatrix(T0SA0, (nf, nf))

        ###
        # Various contraints V*x = W on the normal component
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
        V = list(map(_add_zero_column, V))

        ###
        #  Regularization
        ###
        assert isinstance(self._b, DenseMatrix)
        b_ = self._b.asmatrix()
        if self._is_augmented:
            b_ =  _add_zero_column(b_)
        b_full = PartialDiagonalMatrix(b_, (nf, nf))

        # Optimizer
        equality_conditions = [
            EqualityCondition(0, 1, identity(x_size), identity(x_size)),
        ] # type: list[EqualityCondition]
        if len(V) != 0:
            lstsq = ConstrainedLeastSquares(
                1.0, sua_full, ginput.ravel(),
                np.vstack(V),
                np.hstack(W)
            )
        else:
            lstsq = LeastSquares(1.0, sua_full, ginput.ravel())

        assert b_full is not None
        reg = {"L2": L2Regularizer, "L1": L1Regularizer}[self._reg_type](alpha, b_full)
        terms = [lstsq, reg] # type: List[ObjectiveFunctionBase]
        if spd:
            # Semi-positive definite condition on normal component
            nsmpl_w = self._c.shape[0]
            nn = SemiPositiveDefinitePenalty((nsmpl_w, nf, nf), 0)
            c_ext = _add_zero_column(self._c) if self._is_augmented else self._c
            equality_conditions.append(
                    EqualityCondition(
                        0, 2,
                        PartialDiagonalMatrix(c_ext, (nf, nf)),
                        PartialDiagonalMatrix(ScaledIdentityMatrix(nsmpl_w, 1.0), (nf, nf)),
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


# a: (N, M)
# y: (N, 1)
# np.hstack((a,y)): (N, M+1)
def _add_zero_column(a: Union[np.ndarray, MatrixBase]):
    if isinstance(a, np.ndarray):
        assert a.ndim == 2
        N = a.shape[0]
        y = np.zeros((N, 1))
        return np.hstack((a, y))
    elif isinstance(a, ScaledIdentityMatrix):
        diagonals = np.full(min(*a.shape), a.coeff)
        return DiagonalMatrix(diagonals, shape=(a.shape[0], a.shape[1]+1))
    else:
        raise RuntimeError(f"Invalid type{type(a)}!")
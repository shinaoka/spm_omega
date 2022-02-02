# Copyright (C) 2021-2022 Hiroshi Shinaoka and others
# SPDX-License-Identifier: MIT
import numpy as np
from typing import Optional, Union, Tuple, Dict, List, cast

from sparse_ir import FiniteTempBasis, MatsubaraSampling, TauSampling

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
    The second term of the RHS represents augmentation, such as a constant term in tau/frequency.

    In the time, the spectral representation reads
        G_ij(\tau) = \int d\omega  K(\tau, \omega) \rho_ij(\omega) + P(\tau) c_{ij}.

    For a given G^{input}, we minimize
        |G^{input} - (G^{norm} + G^{singlar})|_^2 + (regulariation term),
    where we dropped the indices for time/frequency and spin oribtals for simplicity.
    G^{singular} represents an additonal contribution that is not complactly (or conveniently)
    represented in IR (e.g., Hartree-Fock term, a delta peak at omega=0 for bosons).

    The regularization term is defined...

    """
    def __init__(
            self,
            basis: FiniteTempBasis,
            sampling: Union[TauSampling, MatsubaraSampling],
            prj_to_IR: MatrixBase,
            prj_to_reg: MatrixBase,
            reg_type: str = "L2",
            singular_term_model: Optional[SingularTermModel] = None,
            sum_rule: Optional[Tuple[np.ndarray,np.ndarray]] = None,
        ) -> None:
        r"""
        basis:
            IR basis

        sampling:
            Sampling object

        prj_to_IR:
            Transformation matrix from parameters of the normal part to rho_l

        prj_to_reg:
            Transformation matrix from parameters of the normal part to the space
            where L1/L2 regularation is imposed on the fitting.

        reg_type: str
            `L1` or `L2`

        prj_to_sum_rule:

        singular_term_model:
            Modeling the singular part

        sum_rule: (a, b)
            Sum rule for the normal part.
            a, b are 2D and 1D array, respectively
            \sum_{l} a_{n,l} rho_{l,ij} = b_{n,ij}
        """
        assert prj_to_IR.shape[1] == prj_to_reg.shape[1], f"{prj_to_IR.shape} {prj_to_reg.shape}"
        assert isinstance(sampling, TauSampling) or isinstance(sampling, MatsubaraSampling)
        assert reg_type in ["L1", "L2"]

        self._beta = basis.beta
        self._basis = basis

        self._prj_to_IR = prj_to_IR
        self._prj_to_reg = prj_to_reg
        self._reg_type = reg_type
        self._sampling = sampling
        self._singular_term_model = singular_term_model
        self._smpl_w = oversample(np.hstack((-basis.wmax, basis.v[-1].roots(), basis.wmax)), 1)
        self._sum_rule = sum_rule


    def predict(
            self,
            x: Tuple[np.ndarray, Optional[np.ndarray]]
        ) -> np.ndarray:
        """
        Evaluate Green's function

        x:
            (Parameters for normal part, parameters for singular part)
        """
        rho_w = self._sampling.evaluate(self._prj_to_IR @ x[0], axis=0)
        if self._singular_term_model is not None:
            assert len(x) > 2
            assert x[1] is not None
            rho_w += self._singular_term_model.evaluate(x[1])
        return rho_w


    def solve(
            self,
            ginput: np.ndarray,
            alpha: float,
            niter: int = 10000,
            spd: bool = True,
            interval_update_mu: int =100,
            initial_guess: Optional[np.ndarray] = None,
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

        initial_guess:
            Initial guess

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
        nparam_normal = self._prj_to_IR.shape[1]
        nparam_w = nparam_normal//(nf**2)
        nparam_singular = 0 if self._singular_term_model is None else nf**2
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
        # Fitting matrix
        #    Fitting matrix is a dense matrix
        ###
        # Fitting parameters of normal component -> sampling points
        K_ = self._sampling.matrix.a @ self._prj_to_IR.asmatrix() # type: np.ndarray
        if self._singular_term_model is not None:
            # K_: (nsmpl, nsmpl_w)
            # new K: (nsmpl, nsmpl_w+1)
            K_ = np.hstack((K_, self._singular_term_model.matrix[:,None])) # type: np.ndarray
        K = PartialDiagonalMatrix(K_, (nf, nf))

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
        assert isinstance(self._prj_to_reg, DenseMatrix)
        reg_prj = None
        if isinstance(self._prj_to_reg, DenseMatrix):
            prj_reg_ext = self._prj_to_reg.asmatrix()
            if self._singular_term_model is not None:
                prj_reg_ext =  add_zero_column(prj_reg_ext)
            reg_prj = PartialDiagonalMatrix(prj_reg_ext, (nf, nf))

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
        assert reg_prj is not None
        reg = {"L2": L2Regularizer, "L1": L1Regularizer}[self._reg_type](alpha, reg_prj)
        terms = [lstsq, reg] # type: List[ObjectiveFunctionBase]
        if spd:
            # Semi-positive definite condition on normal component
            nsmpl_w = self._smpl_w.size #type: int
            nn = SemiPositiveDefinitePenalty((nsmpl_w, nf, nf), 0)
            equality_conditions.append(
                    EqualityCondition(
                        0, 2,
                        ScaledIdentityMatrix((nsmpl_w*nf**2, x_size), 1.0),
                        ScaledIdentityMatrix(nsmpl_w*nf**2, 1.0)
                    )
                )
            terms.append(nn)
        #print("debug")
        #for i, f in enumerate(terms):
            #print(i, f.size_x)
        #for i, e in enumerate(equality_conditions):
            #print(i, e.E1.shape, e.E2.shape)
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
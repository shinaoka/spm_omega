# Copyright (C) 2021-2022 Hiroshi Shinaoka and others
# SPDX-License-Identifier: MIT
import numpy as np
from typing import Dict, Optional, Union, Tuple, List, cast, Sequence, Any

from sparse_ir import MatsubaraSampling, TauSampling

from admmsolver.objectivefunc import LeastSquares, ConstrainedLeastSquares, \
    ObjectiveFunctionBase, L2Regularizer, \
    SemiPositiveDefinitePenalty, L1Regularizer
from admmsolver.optimizer import SimpleOptimizer, Model, EqualityCondition
from admmsolver.matrix import identity, PartialDiagonalMatrix, \
    ScaledIdentityMatrix, MatrixBase, DenseMatrix, DiagonalMatrix

from enum import Enum


class InputType(Enum):
    TIME = 0
    FREQ = 1


class AnaContBase:
    r"""
    Base solver for analytic continuation

    We model this analytic-continuation problem as
        G_{s,ij} = - \sum_l S_l U_{sl} * rho_{l,ij}
                        + alpha * |\sum_l B_{tl} * rho_{l,ij}|_p^p
                        + \sum_n C_{sn} x'_{n,ij},
    where
        s: an imaginary time or an imaginary frequency,
        i and j: spin orbitals,
        alpha: a regularization,
        S_l: singular values.
    The first and second terms of the RHS denote a normal component
    and a singular (augment) component, respectively.

    The expansion coefficients in IR are given by
        rho_{l,ij} = \sum_m A_{l,m} x_{m,ij}.
    The sum rules reads
        \sum_l S_{k,l} rho_{l,ij} = T_{l,ij}.

    The singular component represents an additonal contribution
    that is not complactly (or conveniently) represented in IR
    (e.g., Hartree-Fock term, a delta peak at omega=0 for bosons).
    """
    def __init__(
            self,
            sampling: Sequence[Union[TauSampling, MatsubaraSampling]],
            a: MatrixBase,
            b: MatrixBase,
            c: MatrixBase,
            reg_type: str = "L2",
            sum_rule: Optional[Tuple[np.ndarray, np.ndarray]] = None
            ) -> None:
        r"""
        sampling:
            A sequence of sampling objects.
            The first/second component is regarded as
            a normal/singular component.

        a:
            Transformation matrix `A` from the normal component `x` to rho_l

        b:
            Transformation matrix `B` from the normal component `x`
            to the space where L1/L2 regularation is imposed.

        c:
            Transformation matrix `C` from the normal component `x`
            to the space where the SPD condition is imposed.
            This matrix must be consistent with `smpl_reqal_w`.

        reg_type: str
            `L1` or `L2`

        sum_rule: (S, T)
            Sum rule for the normal component.
            S, T are 2D and 1D array, respectively
        """
        assert a.shape[1] == b.shape[1], f"{a.shape} {b.shape}"
        assert a.shape[1] == c.shape[1], f"{a.shape} {c.shape}"
        assert reg_type in ["L1", "L2"]
        assert len(sampling) == 1 or len(sampling) == 2
        assert np.unique([s_.sampling_points.size for s_ in sampling]).size \
            == 1

        self._sampling = sampling

        self._bases = [s.basis for s in sampling]
        self._beta = cast(float, self._bases[0])  # type: float
        self._is_augmented = len(sampling) > 1

        self._a = a
        self._b = b
        self._c = c
        self._reg_type = reg_type
        self._sum_rule = sum_rule

    def rho_omega(
            self,
            x: np.ndarray,
            omega: np.ndarray) -> np.ndarray:
        """ Compute rho(omega) """
        assert x.ndim == 3
        assert omega.ndim == 1
        rho_l = np.einsum(
            'lx,xij->lij',
            self._a.asmatrix(),
            x[0:self._a.shape[1], ...],
            optimize=True)
        v_omega = self._bases[0].v(omega)
        return np.einsum('lw,lij->wij', v_omega, rho_l, optimize=True)

    def solve_elbow(
            self,
            ginput: np.ndarray,
            alpha_min: float,
            alpha_max: float,
            n_alpha: int,
            **kwargs) -> Tuple[np.ndarray, Dict]:
        assert alpha_min < alpha_max
        assert n_alpha > 0
        alphas = np.exp(
            np.linspace(np.log(alpha_max), np.log(alpha_min), n_alpha))

        lstsq_alpha = []
        info = {}  # type: Dict[str, Any]
        info["info"] = {}
        info["x"] = {}
        initial_guess = None  # type: Optional[List[np.ndarray]]
        for alpha in alphas:
            x_, info_ = self.solve(
                ginput, alpha, initial_guess=initial_guess, **kwargs)
            lstsq_alpha.append(info_["lstsq"])
            info["x"][alpha] = x_
            info["info"][alpha] = info_
            initial_guess = info_["optimizer"].x
        info["alphas"] = alphas
        info["lstsq_alphas"] = np.asarray(lstsq_alpha)

        # Find optimal alpha
        x = np.log(alphas)
        y = np.log(lstsq_alpha)
        a = (y[-1]-y[0])/(x[-1]-x[0])
        y = -(y - (a*(x-x[0]) + y[0]))
        ialpha_opt = np.argmax(y)
        info["ialpha_opt"] = ialpha_opt

        return info["x"][alphas[ialpha_opt]], info

    def solve(
            self,
            ginput: np.ndarray,
            alpha: float,
            niter: int = 10000,
            spd: bool = True,
            interval_update_mu: int = 100,
            rtol: float = 1e-10,
            initial_guess: Sequence[np.ndarray] = None):
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
            Check if all relative norms of all primal and dual residuals
            are smaller than rtol.

        initial_guess:
            Initial guess
        """
        assert ginput.ndim == 3
        assert ginput.shape[0] == self._sampling[0].sampling_points.size, \
            "shape of ginput is not consistent with sampling frequencies!"
        assert ginput.shape[1] == ginput.shape[2], \
            "Invalid shape of ginput"
        if isinstance(self._sum_rule, np.ndarray):
            assert ginput.shape[1:] == self._sum_rule.shape

        nf = ginput.shape[1]  # type: int
        nparam_normal = self._a.shape[1] * nf**2
        assert isinstance(nparam_normal, int)
        nparam_singular = 0 if not self._is_augmented else nf**2
        assert isinstance(nparam_singular, int)
        x_size = nparam_normal + nparam_singular
        assert isinstance(x_size, int)

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
        T0 = self._sampling[0].matrix.a
        T0S = T0.copy()
        T0S[:, 0:self._bases[0].size] *= -self._bases[0].s[None, :]
        T0SA0 = T0S @ self._a.asmatrix()  # type: np.ndarray
        if self._is_augmented:
            T1 = self._sampling[1].matrix.a
            tmp = np.hstack((T0SA0, T1))
            sua_full = PartialDiagonalMatrix(tmp, (nf, nf))
        else:
            sua_full = PartialDiagonalMatrix(T0SA0, (nf, nf))

        ###
        # Various contraints V*x = W on the normal component
        ###
        V = []  # List[np.ndarray]
        W = []  # List[np.ndarray]

        # Sum-rule constraint
        if self._sum_rule is not None:
            a, b = self._sum_rule
            assert a.ndim == 2 and b.ndim == 1
            V.append(a)
            W.append(b)

        # Extend the shape of V to include singular component
        V = [cast(np.ndarray, _add_zero_column(V_)) for V_ in V]

        equality_conditions = []  # type: List[EqualityCondition]

        ###
        #  Regularization
        ###
        reg = None  # type: Optional[ObjectiveFunctionBase]
        if self._reg_type == "L1":
            # x1 = b * x0 and |x1|_1
            b_ = self._b
            if self._is_augmented:
                b_ = _add_zero_column(b_)
            b_full = PartialDiagonalMatrix(b_, (nf, nf))
            equality_conditions.append(
                EqualityCondition(
                    0, 1, b_full, identity(int(b_full.shape[0]))
                ),
            )
            reg = L1Regularizer(alpha, int(b_full.shape[0]))
        elif self._reg_type == "L2":
            # x1 = x0 and |b * x1|_2^2
            if isinstance(self._b, DenseMatrix):
                b_ = self._b.asmatrix()
                if self._is_augmented:
                    b_ = _add_zero_column(b_)
                b_full = PartialDiagonalMatrix(b_, (nf, nf))
            elif isinstance(self._b, ScaledIdentityMatrix):
                b_ = self._b
                if self._is_augmented:
                    b_ = _add_zero_column(b_)
                b_full = PartialDiagonalMatrix(b_, (nf, nf))
            else:
                raise RuntimeError("Unsupported!")
            equality_conditions.append(
                EqualityCondition(
                    0, 1, identity(x_size), identity(x_size)
                ),
            )
            reg = L2Regularizer(alpha, b_full)
        assert reg is not None

        # Optimizer
        if len(V) != 0:
            lstsq = ConstrainedLeastSquares(
                1.0, sua_full, ginput.ravel(),
                np.vstack(V),
                np.hstack(W)
            )
        else:
            lstsq = LeastSquares(1.0, sua_full, ginput.ravel())

        terms = [lstsq, reg]  # type: List[ObjectiveFunctionBase]
        if spd:
            # Semi-positive definite condition on normal component
            nsmpl_w = self._c.shape[0]
            nn = SemiPositiveDefinitePenalty((nsmpl_w, nf, nf), 0)
            c_ext = _add_zero_column(self._c) \
                if self._is_augmented else self._c
            equality_conditions.append(
                    EqualityCondition(
                        0, 2,
                        PartialDiagonalMatrix(c_ext, (nf, nf)),
                        PartialDiagonalMatrix(
                            ScaledIdentityMatrix(nsmpl_w, 1.0), (nf, nf)),
                    )
                )
            terms.append(nn)

        model = Model(terms, equality_conditions)
        opt = SimpleOptimizer(model, x0=initial_guess)

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
    elif isinstance(a, DenseMatrix):
        return DenseMatrix(_add_zero_column(a.asmatrix()))
    else:
        raise RuntimeError(f"Invalid type{type(a)}!")

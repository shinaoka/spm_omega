# Copyright (C) 2021-2022 Hiroshi Shinaoka and others
# SPDX-License-Identifier: MIT
import numpy as np
from typing import Optional, Union, Tuple, Dict, List
from typing import Sequence

from sparse_ir import FiniteTempBasis, MatsubaraSampling, TauSampling
from sparse_ir.augment import LegendreBasis, MatsubaraConstBasis

from admmsolver.matrix import DenseMatrix, ScaledIdentityMatrix

from .solver_base import AnaContBase, InputType
from .util import oversample


class AnaContSpM(object):
    r"""
    Analytic continuation solver based on SpM
    """
    def __init__(
            self,
            beta: float,
            wmax: float,
            statistics: str,
            input_type: Union[str, InputType],
            sampling_points: np.ndarray,
            moment: Optional[np.ndarray] = None,
            singular_term: Union[None, str] = None,
            eps=1e-15,
            oversampling=1,
            reg_type: str = "L1") -> None:
        r"""
        To be written...

        moment: None or 2d array
            First moment m_{ij} = int domega rho_{ij}(omega)

        """
        assert isinstance(beta, float)
        assert isinstance(wmax, float)
        assert isinstance(eps, float)
        assert type(input_type) in [InputType, str]
        if isinstance(input_type, str):
            input_type = {"time": InputType.TIME, "freq": InputType.FREQ}[input_type]
        assert isinstance(sampling_points, np.ndarray) and \
            sampling_points.ndim == 1
        assert moment is None or isinstance(moment, np.ndarray)

        basis = FiniteTempBasis(statistics, beta, wmax, eps=eps)

        # Fitting parameters (i.e, rho_l) to rho_l
        a = ScaledIdentityMatrix(basis.size, coeff=1.0)

        b = ScaledIdentityMatrix(basis.size, coeff=1.0)

        # From rho_l to the integral of rho(omega)
        sum_rule = None
        if moment is not None:
            assert moment.ndim == 2 and moment.shape[0] == moment.shape[1]
            sum_rule_coeff = basis.s * (basis.u(0) + basis.u(basis.beta))
            sum_rule_coeff = sum_rule_coeff.reshape((1,-1))
            sum_rule = (sum_rule_coeff, moment)

        self.basis = basis
        bases = [basis] #type: List[Union[FiniteTempBasis, LegendreBasis, MatsubaraConstBasis]]
        if singular_term is not None:
            if singular_term == "omega0":
                assert statistics == "B"
                bases.append(LegendreBasis(statistics, beta, 1))
            elif singular_term == "HF":
                bases.append(MatsubaraConstBasis(statistics, beta, 1.0))
            else:
                raise RuntimeError(f"Invalid singular_term {singular_term}")

        sampling = []
        smpl_cls = {InputType.FREQ: MatsubaraSampling, InputType.TIME: TauSampling}[input_type]
        for b_ in bases:
            sampling.append(smpl_cls(b_, sampling_points))

        # For semi positive definite condition
        self._smpl_real_w = oversample(np.hstack((-wmax, basis.v[-1].roots(), wmax)), oversampling)
        assert bases[0].v is not None
        c = DenseMatrix(bases[0].v(self._smpl_real_w).T)

        self._solver = AnaContBase(
                sampling, a, b, c,
                sum_rule=sum_rule,
                reg_type=reg_type
            )

    @property
    def smpl_real_w(self):
        return self._smpl_real_w


    def rho_omega(
            self,
            x: np.ndarray,
            omega: np.ndarray
        ) -> np.ndarray:
        """ Compute rho(omega) """
        assert x.ndim == 3
        assert omega.ndim == 1
        rho_l = np.einsum(
            'lx,xij->lij',
            self._solver._a.asmatrix(),
            x[0:self._solver._a.shape[1],...],
            optimize=True)
        v_omega = self._solver._bases[0].v(omega)
        return np.einsum('lw,lij->wij', v_omega, rho_l, optimize=True)

    def solve(
            self,
            ginput: np.ndarray,
            alpha: float,
            niter: int = 10000,
            spd: bool = True,
            interval_update_mu: int =100,
            rtol: float = 1e-10,
            initial_guess: Sequence[np.ndarray] = None
        ) -> Tuple[np.ndarray, Dict]:
        return self._solver.solve(ginput, alpha, niter, spd, interval_update_mu, rtol, initial_guess=initial_guess)

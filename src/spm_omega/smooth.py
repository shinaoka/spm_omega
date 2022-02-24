# Copyright (C) 2021-2022 Hiroshi Shinaoka and others
# SPDX-License-Identifier: MIT
import numpy as np
from typing import Optional, Union, Tuple, Dict, List
from scipy.interpolate import interp1d

from sparse_ir import FiniteTempBasis, MatsubaraSampling, TauSampling
from sparse_ir.augment import LegendreBasis, MatsubaraConstBasis

from admmsolver.matrix import DenseMatrix, ScaledIdentityMatrix
from admmsolver.util import smooth_regularizer_coeff

from .solver_base import AnaContBase, InputType, SimpleAnaContBaseL2
from .util import oversample, prj_w_to_l


class AnaContSmooth(object):
    r"""
    To be written...
    """
    def __init__(
            self,
            beta: float,
            wmax: float,
            statistics: str,
            input_type: Union[str, InputType],
            sampling_points: np.ndarray,
            oversampling: int = 1,
            moment: Optional[np.ndarray] = None,
            singular_term: Union[None, str] = None,
            reg_type: str = "L2",
            eps: float = 1e-15) -> None:
        r"""
        To be written...

        moment: None or 2d array
            First moment m_{ij} = int domega rho_{ij}(omega)

        See docstring of AnaContBase for other options
        """
        assert isinstance(beta, float)
        assert isinstance(wmax, float)
        assert isinstance(eps, float)
        assert type(input_type) in [InputType, str]
        if isinstance(input_type, str):
            input_type = {"time": InputType.TIME, "freq": InputType.FREQ}[input_type]
        assert isinstance(sampling_points, np.ndarray) and \
            sampling_points.ndim == 1
        assert isinstance(oversampling, int)
        assert moment is None or isinstance(moment, np.ndarray)

        basis = FiniteTempBasis(statistics, beta, wmax, eps=eps)

        # Fitting parameters for the normal component (sampling points in the real-frequency space)
        self._smpl_real_w = oversample(np.hstack((-wmax, basis.v[-1].roots(), wmax)), oversampling)

        # From rho(omega_i) to rho_l (using linear interpolation)
        a = prj_w_to_l(basis, self._smpl_real_w, 10)

        # From rho(omea_i) to rho''(omega) [Check uniform mesh]
        b = smooth_regularizer_coeff(self._smpl_real_w) # type: np.ndarray

        # From rho(omega_i) to the integral of rho(omega)
        sum_rule = None
        if moment is not None:
            assert moment.ndim == 2 and moment.shape[0] == moment.shape[1]
            sum_rule_coeff = basis.s * (basis.u(0) + basis.u(basis.beta))
            sum_rule_coeff = sum_rule_coeff.reshape((1,-1))
            sum_rule = (sum_rule_coeff, moment)

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

        c = ScaledIdentityMatrix(self._smpl_real_w.size, 1.0)

        self._solver = AnaContBase(
                sampling,
                DenseMatrix(a),
                DenseMatrix(b),
                c,
                sum_rule=sum_rule,
                reg_type=reg_type
            )

    @property
    def smpl_real_w(self):
        return self._smpl_real_w

    def rho_omega(
            self,
            x: np.ndarray,
            omega: np.ndarray) -> np.ndarray:
        """ Compute rho(omega) """
        intp = interp1d(
            self._smpl_real_w, x[0:self._smpl_real_w.size, :, :], axis=0)
        return intp(omega)

    def solve(
            self,
            ginput: np.ndarray,
            alpha: float,
            niter: int = 10000,
            spd: bool = True,
            interval_update_mu: int = 100,
            rtol: float = 1e-10) -> Tuple[np.ndarray, Dict]:
        return self._solver.solve(
            ginput, alpha, niter, spd, interval_update_mu, rtol)

    def solve_elbow(
            self,
            ginput: np.ndarray,
            alpha_min: float,
            alpha_max: float,
            n_alpha: int,
            **kwargs) -> Tuple[np.ndarray, Dict]:
        return self._solver.solve_elbow(
            ginput, alpha_min, alpha_max, n_alpha, **kwargs)


class AnaContSmoothOpt(object):
    r"""
    To be written...
    """
    def __init__(
            self,
            beta: float,
            wmax: float,
            statistics: str,
            input_type: Union[str, InputType],
            sampling_points: np.ndarray,
            oversampling: int = 1,
            moment: Optional[np.ndarray] = None,
            eps: float = 1e-15,
            scale_alpha: bool = False) -> None:
        r"""
        To be written...

        moment: None or 2d array
            First moment m_{ij} = int domega rho_{ij}(omega)

        See docstring of AnaContBase for other options
        """
        assert isinstance(beta, float)
        assert isinstance(wmax, float)
        assert isinstance(eps, float)
        assert type(input_type) in [InputType, str]
        if isinstance(input_type, str):
            input_type = {
                "time": InputType.TIME, "freq": InputType.FREQ}[input_type]
        assert isinstance(sampling_points, np.ndarray) and \
            sampling_points.ndim == 1
        assert isinstance(oversampling, int)
        assert moment is None or isinstance(moment, np.ndarray)

        basis = FiniteTempBasis(statistics, beta, wmax, eps=eps)

        # Fitting parameters for the normal component
        # (sampling points in the real-frequency space)
        self._smpl_real_w = oversample(
            np.hstack((-wmax, basis.v[-1].roots(), wmax)), oversampling)

        # From rho(omega_i) to rho_l (using linear interpolation)
        a = prj_w_to_l(basis, self._smpl_real_w, 10)

        # From rho(omea_i) to rho''(omega) [Check uniform mesh]
        b = smooth_regularizer_coeff(self._smpl_real_w)  # type: np.ndarray

        # From rho(omega_i) to the integral of rho(omega)
        sum_rule = None
        if moment is not None:
            assert moment.ndim == 2 and moment.shape[0] == moment.shape[1]
            sum_rule_coeff = basis.s * (basis.u(0) + basis.u(basis.beta))
            sum_rule_coeff = sum_rule_coeff.reshape((1, -1))
            sum_rule = (sum_rule_coeff, moment)

        sampling = []
        smpl_cls = {
            InputType.FREQ: MatsubaraSampling,
            InputType.TIME: TauSampling}[input_type]
        sampling = smpl_cls(basis, sampling_points)

        c = ScaledIdentityMatrix(self._smpl_real_w.size, 1.0)

        # alpha
#        self._coeff_alpha = 1.0
#        if scale_alpha:
#            svd_a = np.linalg.svd(a)
#            svd_b = np.linalg.svd(b)
#            self.coeff_alpha = (svd_a[1][0]/svd_b[1][0])**2

        self._solver = SimpleAnaContBaseL2(
                sampling,
                DenseMatrix(a),
                DenseMatrix(b),
                c,
                sum_rule=sum_rule,
            )

    @property
    def smpl_real_w(self):
        return self._smpl_real_w

    def rho_omega(
            self,
            x: np.ndarray,
            omega: np.ndarray) -> np.ndarray:
        """ Compute rho(omega) """
        intp = interp1d(
            self._smpl_real_w, x[0:self._smpl_real_w.size, :, :], axis=0)
        return intp(omega)

    def solve(
            self,
            ginput: np.ndarray,
            alpha: float,
            niter: int = 10000,
            spd: bool = True,
            interval_update_mu: int = 100,
            rtol: float = 1e-10) -> Tuple[np.ndarray, Dict]:
        return self._solver.solve(
            ginput, alpha, niter, spd, interval_update_mu, rtol)

    def solve_elbow(
            self,
            ginput: np.ndarray,
            alpha_min: float,
            alpha_max: float,
            n_alpha: int,
            **kwargs) -> Tuple[np.ndarray, Dict]:
        return self._solver.solve_elbow(
            ginput, alpha_min, alpha_max, n_alpha, **kwargs)

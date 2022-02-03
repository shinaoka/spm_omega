# Copyright (C) 2021-2022 Hiroshi Shinaoka and others
# SPDX-License-Identifier: MIT
import numpy as np
from typing import Optional, Union, Tuple, Dict, List

from sparse_ir import FiniteTempBasis, MatsubaraSampling, TauSampling, KernelFFlat

from admmsolver.matrix import DenseMatrix
from admmsolver.util import smooth_regularizer_coeff

from .solver_base import AnaContBase, SingularTermModel, InputType
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
            input_type: InputType,
            sampling_points: np.ndarray,
            oversampling: int = 1,
            moment: Optional[np.ndarray] = None,
            singular_basis: Union[None, str] = None
        ) -> None:
        r"""
        To be written...

        moment: None or 2d array
            First moment m_{ij} = int domega rho_{ij}(omega)

        See docstring of AnaContBase for other options
        """
        assert isinstance(beta, float)
        assert isinstance(wmax, float)
        assert isinstance(input_type, InputType)
        assert isinstance(sampling_points, np.ndarray) and \
            sampling_points.ndim == 1
        assert isinstance(oversampling, int)
        assert moment is None or isinstance(moment, int)

        basis = FiniteTempBasis(
            statistics, beta, wmax, eps=1e-15, kernel=KernelFFlat(beta * wmax))

        # Fitting parameters for the normal component (sampling points in the real-frequency space)
        self._smpl_real_w = oversample(np.hstack((-wmax, basis.v[-1].roots(), wmax)), oversampling)

        sampling = {
            InputType.FREQ: MatsubaraSampling,
            InputType.TIME: TauSampling,
        }[input_type](basis, sampling_points)

        # From sampling points in the real frequency space to IR (using linear interpolation)
        a = prj_w_to_l(basis, self._smpl_real_w, 10)

        # From sampling points in the real frequency space to the integral of second derivative of rho(omega)
        b = smooth_regularizer_coeff(self._smpl_real_w) # type: np.ndarray

        # From sampling points in the real frequency space to the integral of rho(omega)
        sum_rule = None
        if moment is not None:
            assert moment.ndim == 2 and moment.shape[0] == moment.shape[1]
            prj_sum_ = basis.s * (basis.u(0) + basis.u(basis.beta))
            sum_rule = (prj_sum_ @ a, moment)

        if singular_basis is not None:
            pass

        self._solver = AnaContBase(
            sampling,
            DenseMatrix(a),
            DenseMatrix(b),
            smpl_real_w = self._smpl_real_w,
            sum_rule=sum_rule)

    @property
    def smpl_real_w(self):
        return self._smpl_real_w

    def predict(
            self,
            x: np.ndarray
        ) -> np.ndarray:
        return self._solver.predict(x)

    def solve(
            self,
            ginput: np.ndarray,
            alpha: float,
            niter: int = 10000,
            spd: bool = True,
            interval_update_mu: int =100,
            rtol: float = 1e-10
        ) -> Tuple[np.ndarray, Dict]:
        return self._solver.solve(ginput, alpha, niter, spd, interval_update_mu, rtol)
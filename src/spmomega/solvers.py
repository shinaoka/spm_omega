# Copyright (C) 2021-2022 Hiroshi Shinaoka and others
# SPDX-License-Identifier: MIT
import numpy as np
from typing import Optional, Union, Tuple, Dict, List

from sparse_ir import FiniteTempBasis, MatsubaraSampling, TauSampling

from admmsolver.matrix import DenseMatrix
from admmsolver.util import smooth_regularizer_coeff

from .solver_base import AnaContBase, SingularTermModel
from .util import oversample, prj_w_to_l


class AnaContSmooth(AnaContBase):
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
            basis: FiniteTempBasis,
            sampling: Union[TauSampling, MatsubaraSampling],
            reg_type: str = "L2",
            singular_term_model: Optional[SingularTermModel] = None,
            oversampling: int = 1,
            moment: Optional[np.ndarray] = None,
        ) -> None:
        r"""
        To be written...

        moment: None or 2d array
            First moment m_{ij} = int domega rho_{ij}(omega)

        See docstring of AnaContBase for other options
        """
        assert type(sampling) in [TauSampling, MatsubaraSampling]

        # Fitting parameters for the normal component (sampling points in the real-frequency space)
        self._smpl_w = oversample(np.hstack((-basis.wmax, basis.v[-1].roots(), basis.wmax)), oversampling)

        # From sampling points in the real frequency space to IR (using linear interpolation)
        prj_to_IR = prj_w_to_l(basis, self._smpl_w, 10)

        # From sampling points in the real frequency space to the integral of second derivative of rho(omega)
        prj_to_reg = smooth_regularizer_coeff(self._smpl_w) # type: np.ndarray
        print("debug", prj_to_reg.shape)

        # From sampling points in the real frequency space to the integral of rho(omega)
        sum_rule = None
        if moment is not None:
            assert moment.ndim == 2 and moment.shape[0] == moment.shape[1]
            prj_sum = basis.s * (basis.u(0) + basis.u(basis.beta))
            sum_rule = (prj_sum @ prj_to_IR, moment)

        super().__init__(
            basis, sampling,
            DenseMatrix(prj_to_IR), DenseMatrix(prj_to_reg),
            reg_type,
            singular_term_model,
            sum_rule=sum_rule)

    @property
    def smpl_w(self):
        return self._smpl_w

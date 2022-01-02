from spmomega.solver import _second_deriv

import numpy as np

def test_second_deriv():
    """
    f(x) = x^2
    f''(x) = 2
    """
    N  = 10
    x = np.cos(np.linspace(np.pi, 0, N))
    y = x**2
    ypp = _second_deriv(x, y[:,None])
    np.testing.assert_allclose(ypp.ravel(), np.full(N-2, 2))
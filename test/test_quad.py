import numpy as np
from numpy.polynomial.legendre import leggauss
from spm_omega.quad import scale_quad, composite_leggauss

def test_scale_quad():
    """ Integrate x^3 over [0, 3] """
    xmax = 3
    x_, w_ = leggauss(10)
    x, w = scale_quad(x_, w_, 0, xmax)
    f = lambda x: x**3
    res = np.sum(w * f(x))
    ref = 0.25*(xmax**4)
    assert np.abs(res - ref) < 1e-10

def test_composite_leggauss():
    xmax = 3
    segments = np.array([0, 0.1, 3.0])
    deg = 10
    x, w = composite_leggauss(segments, deg)
    f = lambda x: x**3
    res = np.sum(w * f(x))
    ref = 0.25*(xmax**4)
    assert np.abs(res - ref) < 1e-10
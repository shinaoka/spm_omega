import numpy as np
from numpy.polynomial.legendre import leggauss

def scale_quad(x, w, xmin, xmax):
    """ Scale weights and notes of quadrature to the interval [xmin, xmax] """
    assert xmin < xmax
    dx = xmax - xmin
    w_ = 0.5 * dx * w
    x_ = (0.5 * dx) * (x + 1) + xmin
    return x_, w_

def composite_leggauss(segments, deg):
    nsp = segments.size
    x, w = leggauss(deg)
    x_comp, w_comp = [], []
    for s in range(nsp-1):
        x_, w_ = scale_quad(x, w, segments[s], segments[s+1])
        x_comp.append(x_)
        w_comp.append(w_)
    return np.hstack(x_comp), np.hstack(w_comp)
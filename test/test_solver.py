from spmomega.solver import _prj_w_to_l, SpMOmega

import numpy as np
from irbasis3 import FiniteTempBasis, KernelFFlat


def test_prj_w_to_l():
    lambda_ = 1e+1
    beta = 1
    eps = 1e-5
    deg = 10
    wmax = lambda_/beta

    basis = FiniteTempBasis(KernelFFlat(lambda_), "F", beta, eps)

    smpl_w = np.linspace(-wmax, wmax, 1000)
    prj = _prj_w_to_l(basis, smpl_w, deg)

    np.testing.assert_allclose(
        prj @ basis.v[0](smpl_w),
        np.hstack((1, np.zeros(basis.size-1))),
        rtol=0,
        atol=1e-5
    )

    np.testing.assert_allclose(
        prj @ basis.v[-1](smpl_w),
        np.hstack((np.zeros(basis.size-1), 1)),
        rtol=0,
        atol=1e-3
    )


def test_single_orbital_three_gaussian_peaks():
    """ Three-Gaussian model from SpM paper """
    gaussian = lambda x, mu, sigma: np.exp(-((x-mu)/sigma)**2)/(np.sqrt(np.pi)*sigma)
    rho = lambda omega: 0.2*gaussian(omega, 0.0, 0.15) + \
        0.4*gaussian(omega, 1.0, 0.8) + 0.4*gaussian(omega, -1.0, 0.8)
    
    wmax = 10.0
    beta = 100.0
    alpha = 1e-10

    lambda_ = wmax * beta
    basis = FiniteTempBasis(
        KernelFFlat(lambda_),
        "F", beta, eps=1e-7)
    
    # Compute exact rho_l and g_l
    rho_l = basis.v.overlap(rho)
    g_l = -basis.s * rho_l

    solver = SpMOmega(basis)
    rho_w, _ = solver.solve(g_l[:,None,None], alpha, np.ones((1,1)), niter=1000)

    np.testing.assert_allclose(rho_w.ravel(), rho(solver.smpl_w), rtol=0, atol=0.05)
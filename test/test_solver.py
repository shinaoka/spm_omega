from spmomega.solver import _prj_w_to_l, SpMOmega, SpMOmegaMatsubara

import numpy as np
from irbasis3 import FiniteTempBasis, KernelFFlat, MatsubaraSampling
import pytest

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

gaussian = lambda x, mu, sigma: np.exp(-((x-mu)/sigma)**2)/(np.sqrt(np.pi)*sigma)

# Three-Gaussian model from SpM paper
def rho_single_orb(omega):
    res = 0.2*gaussian(omega, 0.0, 0.15) + \
        0.4*gaussian(omega, 1.0, 0.8) + 0.4*gaussian(omega, -1.0, 0.8)
    return res[:,None,None]


def rho_two_orb(omega):
    theta = 0.2*np.pi
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rho_omega = np.zeros((omega.size, 2, 2))
    rho_omega[:,0,0] = 0.2*gaussian(omega, 0.0, 0.15)
    rho_omega[:,1,1] = 0.4*gaussian(omega, 1.0, 0.8) + 0.4*gaussian(omega, -1.0, 0.8)
    return np.einsum('ji,wjk,kl->wil', rot_mat, rho_omega, rot_mat)

@pytest.mark.parametrize("rho", [rho_single_orb, rho_two_orb])
def test_reconst_rho_w(rho):
    wmax = 10.0
    beta = 100.0
    alpha = 1e-10

    lambda_ = wmax * beta
    basis = FiniteTempBasis(
        KernelFFlat(lambda_),
        "F", beta, eps=1e-7)
    
    # Compute exact rho_l and g_l
    rho_test = rho(np.linspace(-1,1,100))
    rho_l = basis.v.overlap(rho, axis=0)
    rho_l = rho_l.reshape((basis.size,) + rho_test.shape[1:])
    g_l = -basis.s[:,None,None] * rho_l

    solver = SpMOmega(basis)
    rho_w, _ = solver.solve(g_l, alpha, niter=1000)

    np.testing.assert_allclose(rho_w, rho(solver.smpl_w), rtol=0, atol=0.05)

@pytest.mark.parametrize("rho", [(rho_single_orb), (rho_two_orb)])
def test_reconst_rho_w_matsubara(rho):
    wmax = 10.0
    beta = 100.0
    alpha = 1e-10
    niv = 1000
    vsample = 2*np.arange(-niv, niv)+1

    lambda_ = wmax * beta
    basis = FiniteTempBasis(
        KernelFFlat(lambda_),
        "F", beta, eps=1e-12)
    smpl = MatsubaraSampling(basis, vsample)
    
    # Compute exact rho_l, g_l, giv
    rho_test = rho(np.linspace(-1,1,100))
    rho_l = basis.v.overlap(rho, axis=0)
    rho_l = rho_l.reshape((basis.size,) + rho_test.shape[1:])
    g_l = -basis.s[:,None,None] * rho_l
    g_iv = smpl.evaluate(g_l, axis=0)
    nf = g_iv.shape[1]
    
    solver = SpMOmegaMatsubara(beta, "F", wmax, vsample)
    rho_w, info = solver.solve(g_iv, alpha, niter=1000)

    np.testing.assert_allclose(rho_w, rho(solver.smpl_w), rtol=0, atol=0.05)
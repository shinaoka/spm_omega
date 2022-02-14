from typing import Optional

from spm_omega import AnaContSmooth, AnaContSpM

import numpy as np
from sparse_ir import FiniteTempBasis, KernelFFlat,\
    MatsubaraSampling, TauSampling
import pytest


def gaussian(x, mu, sigma): return np.exp(-((x-mu)/sigma)**2) / \
    (np.sqrt(np.pi)*sigma)

# Three-Gaussian model from SpM paper


def rho_single_orb(omega):
    res = 0.2*gaussian(omega, 0.0, 0.15) + \
        0.4*gaussian(omega, 1.0, 0.8) + 0.4*gaussian(omega, -1.0, 0.8)
    return res[:, None, None]


def rho_two_orb(omega):
    theta = 0.2*np.pi
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
    rho_omega = np.zeros((omega.size, 2, 2))
    rho_omega[:, 0, 0] = 0.2*gaussian(omega, 0.0, 0.15)
    rho_omega[:, 1, 1] = 0.4 * \
        gaussian(omega, 1.0, 0.8) + 0.4*gaussian(omega, -1.0, 0.8)
    return np.einsum('ji,wjk,kl->wil', rot_mat, rho_omega, rot_mat)


def get_singular_term_matsubara(type: Optional[str], ginput: np.ndarray):
    if type == "HF":
        return np.ones_like(ginput)
    return np.zeros_like(ginput)


def _test_solver(SolverType, stat, rho, augment, reg_type):
    wmax = 10.0
    beta = 100.0
    alpha = 1e-8
    niv = 1000
    shift = {"F": 1, "B": 0}[stat]
    vsample = 2*np.arange(-niv, niv) + shift
    tausample = np.linspace(0, beta, 2*niv)
    niter = {"L1": 1000, "L2": 1000}[reg_type]

    basis = FiniteTempBasis(stat, beta, wmax, eps=1e-12,
                            kernel=KernelFFlat(beta*wmax))
    smpl_matsu = MatsubaraSampling(basis, vsample)
    smpl_tau = TauSampling(basis, tausample)

    # Compute exact rho_l, g_l, g_iv, g_tau
    rho_test = rho(np.linspace(-1, 1, 100))
    rho_l = basis.v.overlap(rho, axis=0)
    rho_l = rho_l.reshape((basis.size,) + rho_test.shape[1:])
    g_l = -basis.s[:, None, None] * rho_l

    # From Matsubara
    g_iv = smpl_matsu.evaluate(g_l, axis=0)
    g_iv += get_singular_term_matsubara(augment, g_iv)
    solver = SolverType(
        beta, wmax, stat, "freq", vsample, singular_term=augment,
        reg_type=reg_type
    )
    x, info = solver.solve(g_iv, alpha, niter=niter, spd=True)
    rho_w = solver.rho_omega(x, solver.smpl_real_w)
    for x_, y_ in zip(
            rho_w[0:solver.smpl_real_w.size], rho(solver.smpl_real_w)):
        print(x_[0, 0].real, y_[0, 0].real)
    np.testing.assert_allclose(
        rho_w[0:solver.smpl_real_w.size],
        rho(solver.smpl_real_w), rtol=0, atol=0.15)

    # From tau
    if augment != "HF":
        gtau = smpl_tau.evaluate(g_l, axis=0)
        solver = AnaContSmooth(beta, wmax, stat, "time",
                               tausample, reg_type=reg_type)
        x, _ = solver.solve(gtau, alpha, niter=niter)
        rho_w = solver.rho_omega(x, solver.smpl_real_w)
        np.testing.assert_allclose(rho_w, rho(
            solver.smpl_real_w), rtol=0, atol=0.15)


@pytest.mark.parametrize("stat", ["F", "B"])
@pytest.mark.parametrize("rho", [(rho_single_orb), (rho_two_orb)])
@pytest.mark.parametrize("augment", [None, "HF"])
@pytest.mark.parametrize("reg_type", ["L1", "L2"])
def test_smooth(stat, rho, augment, reg_type):
    _test_solver(AnaContSmooth, stat, rho, augment, reg_type)


@pytest.mark.parametrize("stat", ["F", "B"])
@pytest.mark.parametrize("rho", [(rho_single_orb), (rho_two_orb)])
@pytest.mark.parametrize("augment", [None, "HF"])
@pytest.mark.parametrize("reg_type", ["L1"])
def test_spm(stat, rho, augment, reg_type):
    _test_solver(AnaContSpM, stat, rho, augment, reg_type)


def test_elbow():
    stat = "F"
    wmax = 10.0
    beta = 100.0
    alpha_min = 1e-10
    alpha_max = 1.0
    n_alpha = 5
    niv = 500
    shift = {"F": 1, "B": 0}[stat]
    vsample = 2*np.arange(-niv, niv) + shift
    niter = 1000

    basis = FiniteTempBasis(stat, beta, wmax, eps=1e-12,
                            kernel=KernelFFlat(beta*wmax))
    smpl_matsu = MatsubaraSampling(basis, vsample)

    # Compute exact rho_l, g_l, g_iv, g_tau
    rho_test = rho_single_orb(np.linspace(-1, 1, 100))
    rho_l = basis.v.overlap(rho_single_orb, axis=0)
    rho_l = rho_l.reshape((basis.size,) + rho_test.shape[1:])
    g_l = -basis.s[:, None, None] * rho_l

    # From Matsubara
    g_iv = smpl_matsu.evaluate(g_l, axis=0)
    solver = AnaContSmooth(beta, wmax, stat, "freq", vsample)
    x, info = solver.solve_elbow(
        g_iv, alpha_min, alpha_max, n_alpha, niter=niter)

#    for alpha, lst in zip(info["alphas"], info["lstsq_alphas"]):
#        print(alpha, lst)
    rho_w = solver.rho_omega(x, solver.smpl_real_w)
#    for x_, y_ in zip(
#            rho_w[0:solver.smpl_real_w.size],
#            rho_single_orb(solver.smpl_real_w)):
#        print(x_[0, 0].real, y_[0, 0].real)
    np.testing.assert_allclose(
        rho_w[0:solver.smpl_real_w.size],
        rho_single_orb(solver.smpl_real_w), rtol=0, atol=0.15)

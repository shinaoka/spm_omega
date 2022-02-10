from typing import Optional

from spm_omega.solver import _prj_w_to_l, SpMSmooth, SpM
from spm_omega.solver_base import InputType
from spm_omega.solvers import AnaContSmooth

import numpy as np
from sparse_ir import FiniteTempBasis, KernelFFlat, MatsubaraSampling, TauSampling
import pytest

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


#def get_augmentation_freq(type: str, ginput: np.ndarray):
    #if type == "HF":
        ## DEBUG
        ##return np.ones_like(ginput)
        #return np.zeros_like(ginput)
    #elif type == "omega0":
        #pass

def get_singular_term_matsubara(type: Optional[str], ginput: np.ndarray):
    if type == "HF":
        return np.ones_like(ginput)
    return np.zeros_like(ginput)


@pytest.mark.parametrize("stat", ["B"])
#@pytest.mark.parametrize("rho", [(rho_single_orb), (rho_two_orb)])
@pytest.mark.parametrize("rho", [(rho_single_orb)])
#@pytest.mark.parametrize("augment", [None, "HF"])
@pytest.mark.parametrize("augment", [None])
def test_smooth(stat, rho, augment):
    wmax = 10.0
    beta = 100.0
    alpha = 1e-10
    niv = 1000
    shift = {"F": 1, "B": 0}[stat]
    vsample = 2*np.arange(-niv, niv) + shift
    tausample = np.linspace(0, beta, 2*niv)

    basis = FiniteTempBasis(stat, beta, wmax, eps=1e-12, kernel=KernelFFlat(beta*wmax))
    smpl_matsu = MatsubaraSampling(basis, vsample)
    smpl_tau = TauSampling(basis, tausample)

    # Compute exact rho_l, g_l, g_iv, g_tau
    rho_test = rho(np.linspace(-1,1,100))
    rho_l = basis.v.overlap(rho, axis=0)
    rho_l = rho_l.reshape((basis.size,) + rho_test.shape[1:])
    g_l = -basis.s[:,None,None] * rho_l

    # From Matsubara
    g_iv = smpl_matsu.evaluate(g_l, axis=0)
    g_iv += get_singular_term_matsubara(augment, g_iv)
    solver = AnaContSmooth(
        beta, wmax, stat, "freq", vsample, singular_term=augment)
    rho_w, info = solver.solve(g_iv, alpha, niter=1000, spd=False)
    print("info", info["lstsq"])
    print("# ", rho_w[-1,0,0].real)
    for x_, y_ in zip(
        rho_w[0:solver.smpl_real_w.size], rho(solver.smpl_real_w)):
        print(x_[0,0].real, y_[0,0].real)
    np.testing.assert_allclose(
        rho_w[0:solver.smpl_real_w.size], rho(solver.smpl_real_w), rtol=0, atol=0.05)

    # From tau
    if augment != "HF":
        gtau = smpl_tau.evaluate(g_l, axis=0)
        solver = AnaContSmooth(beta, wmax, stat, "time", tausample)
        rho_w, _ = solver.solve(gtau, alpha, niter=1000)
        np.testing.assert_allclose(rho_w, rho(solver.smpl_real_w), rtol=0, atol=0.05)
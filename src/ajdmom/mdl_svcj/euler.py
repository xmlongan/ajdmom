"""
Module for generating samples from :abbr:`SVCJ(Stochastic Volatility with
Contemporaneous Jumps in the price and variance)`

Note that we are generating i.i.d. samples with fixed conditions instead of
a trajectory of samples!
"""
import math

import numpy as np
from ajdmom.mdl_svvj.euler import rSVVJ

rng = np.random.default_rng()


def one_step_euler(v0, mu, k, theta, sigma, rho, tau):
    eps1 = rng.normal(0, 1)
    eps2 = rng.normal(0, 1)
    diffusion1 = sigma * math.sqrt(v0 * tau) * eps1
    diffusion2 = math.sqrt(v0 * tau) * (rho * eps1 + math.sqrt(1-rho**2) * eps2)
    #
    v = v0 + k * (theta - v0) * tau + diffusion1
    v = v if v >= 0 else 0
    #
    y = (mu - v0/2) * tau + diffusion2
    return v, y

def mul_step_euler(v0, mu, k, theta, sigma, rho, tau, nsegment=10):
    delta = tau / nsegment
    cumy = 0
    if nsegment == 0:
        v, y = one_step_euler(v0, mu, k, theta, sigma, rho, tau)
        return v, y
    for i in range(nsegment):
        v, y = one_step_euler(v0, mu, k, theta, sigma, rho, delta)
        v0 = v
        cumy += y
    return v, cumy

def r_v_y(v0, mu, k, theta, sigma, rho, lmbd, mu_v, rhoJ, mu_s, sigma_s, h, nsegment=10):
    numJ = rng.poisson(lmbd * h)
    jtime = sorted(rng.uniform(0, h, numJ))
    jtime = np.array(jtime)
    jsizev = rng.exponential(mu_v, numJ)
    jsizey = rng.normal(mu_s, sigma_s, numJ) + rhoJ * jsizev
    #
    cumy = 0
    #
    if numJ == 0:
        v, cumy = mul_step_euler(v0, mu, k, theta, sigma, rho, h, nsegment)
        return v, cumy
    # fine simulation
    for i in range(numJ):
        delta = jtime[0] if i == 0 else jtime[i] - jtime[i-1]
        ratio = delta / h
        # diffusion
        nseg = math.ceil(ratio * nsegment)
        v, y = mul_step_euler(v0, mu, k, theta, sigma, rho, delta, nseg)
        # jump
        v += jsizev[i]
        y += jsizey[i]
        cumy += y
        #
        v0 = v
    # last diffusion
    delta = h - jtime[-1]
    ratio = delta / h
    nseg = math.ceil(ratio * nsegment)
    v, y = mul_step_euler(v0, mu, k, theta, sigma, rho, delta, nseg)
    cumy += y
    return v, cumy

def r_y(v0, mu, k, theta, sigma, rho, lmbd, mu_v, rhoJ, mu_s, sigma_s, h, N, nsegment=10):
    """Euler approximation"""
    # y_series = []
    ys = []
    for i in range(N):
        # v0, y = r_v_y(v0, mu, k, theta, sigma, rho, lmbd, mu_v, rhoJ, mu_s, sigma_s, h, nsegment)
        # y_series.append(y)
        v, y = r_v_y(v0, mu, k, theta, sigma, rho, lmbd, mu_v, rhoJ, mu_s, sigma_s, h, nsegment)
        ys.append(y)
    # return y_series
    return ys

def rSVCJ(v0, mu, k, theta, sigma, rho, T, jsize_v, jtime_v,
          mu_s, sigma_s, rhoJ, n, n_segment):
    r"""Generate i.i.d. Random samples from mdl_svcj by Euler approximation

    :param float v0: value of the initial variance :math:`v(0)`
    :param float mu: parameter :math:`\mu`
    :param float k: parameter :math:`k`
    :param float theta: parameter :math:`\theta`
    :param float sigma: parameter :math:`\sigma_v`
    :param float rho: parameter :math:`\rho`
    :param float T: end time
    :param tuple jsize_v: jump sizes in the variance over [0, T]
    :param tuple jtime_v: jump time points in the variance over [0, T]
    :param float mu_s: jumps in the price distributed according to
      :math:`\mathcal{N}(\mu_s + J^v\rhoJ, \sigma_s^2)`
    :param float sigma_s: standard deviation of the normal distribution of jumps
      in the price
    :param float rhoJ: :math:`\mu_s + J^v\rhoJ`
    :param int n: number of samples to generate
    :param int n_seqment: number of segments to split [0, T]
    :return: i.i.d. samples
    :rtype: numpy.ndarray
    """
    y_svvj = rSVVJ(v0, mu, k, theta, sigma, rho, T, jsize_v, jtime_v, n, n_segment)
    #
    IZ = np.zeros(n)
    for i in range(n):
        for j in range(len(jsize_v)):
            IZ[i] += rng.normal(mu_s + rhoJ * jsize_v[j], sigma_s)
    return y_svvj + IZ


if __name__ == '__main__':
    import sys

    v0, h = 0.007569, 1
    k, theta, sigma, rho = 3.46, 0.008, 0.14, -0.82
    r = 0.0319
    lmbd_v, mu_v = 0.47, 0.05
    mu_b = -0.1
    lmbd_s, sigma_s, rhoJ = lmbd_v, 0.0001, -0.38
    #
    mu_s = math.log((1 + mu_b) * (1 - rhoJ * mu_v)) - sigma_s ** 2 / 2
    #
    args = sys.argv[1:]
    numJ = rng.poisson(lmbd_v * h) if len(args) == 0 else int(args[0])
    jsize_v = rng.exponential(mu_v, numJ)
    jtime_v = rng.uniform(0, h, numJ)
    jtime_v = sorted(jtime_v)
    # change to tuples
    jsize_v, jtime_v = tuple(jsize_v), tuple(jtime_v)
    #
    n, n_segment = 20, 10
    y = rSVCJ(v0, r, k, theta, sigma, rho, h, jsize_v, jtime_v,
              mu_s, sigma_s, rhoJ, n, n_segment)
    print(f"\nGiven v0 = {v0} and {numJ} jumps in the variance, ")
    print(f"jsize_v = {jsize_v}, \njtime_v = {jtime_v}")
    print(f"\n{n} i.i.d. samples from SVCJ: ")
    print(y)

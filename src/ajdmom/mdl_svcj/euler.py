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


def rSVCJ(v0, mu, k, theta, sigma, rho, T, jsize_v, jtime_v,
          mu_s, sigma_s, rho_J, n, n_segment):
    """Generate i.i.d. Random samples from mdl_svcj by Euler approximation

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
      :math:`\mathcal{N}(\mu_s + J^v\rho_J, \sigma_s^2)`
    :param float sigma_s: standard deviation of the normal distribution of jumps
      in the price
    :param float rho_J: :math:`\mu_s + J^v\rho_J`
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
            IZ[i] += rng.normal(mu_s + rho_J * jsize_v[j], sigma_s)
    return y_svvj + IZ


if __name__ == '__main__':
    import sys

    v0, h = 0.007569, 1
    k, theta, sigma, rho = 3.46, 0.008, 0.14, -0.82
    r = 0.0319
    lmbd_v, mu_v = 0.47, 0.05
    mu_bar = -0.1
    lmbd_s, sigma_s, rho_J = lmbd_v, 0.0001, -0.38
    #
    mu_s = math.log((1 + mu_bar) * (1 - rho_J * mu_v)) - sigma_s ** 2 / 2
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
              mu_s, sigma_s, rho_J, n, n_segment)
    print(f"\nGiven v0 = {v0} and {numJ} jumps in the variance, ")
    print(f"jsize_v = {jsize_v}, \njtime_v = {jtime_v}")
    print(f"\n{n} i.i.d. samples from SVCJ: ")
    print(y)

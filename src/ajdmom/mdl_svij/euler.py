"""
Module for generating samples from :abbr:`SVIJ(Stochastic Volatility with
Independent Jumps in the price and variance)`

Note that we are generating i.i.d. samples with fixed conditions instead of
a trajectory of samples!
"""
import math
import numpy as np
from ajdmom.mdl_svvj.euler import rSVVJ

rng = np.random.default_rng()


def rSVIJ(v0, mu, k, theta, sigma, rho, T, jsize_v, jtime_v,
          lmbd_s, mu_s, sigma_s, n, n_seqment):
    """Generate i.i.d. Random samples from mdl_svij by Euler approximation

    :param float v0: value of the initial variance :math:`v(0)`
    :param float mu: parameter :math:`\mu`
    :param float k: parameter :math:`k`
    :param float theta: parameter :math:`\theta`
    :param float sigma: parameter :math:`\sigma_v`
    :param float rho: parameter :math:`\rho`
    :param float T: end time
    :param tuple jsize_v: jump sizes in the variance over [0, T]
    :param tuple jtime_v: jump time points in the variance over [0, T]
    :param float lmbd_s: jump arrival rate in the price
    :param float mu_s: mean of the normal distribution of jumps in the price
    :param float sigma_s: standard deviation of the normal distribution of jumps
      in the price
    :param int n: number of samples to generate
    :param int n_seqment: number of segments to split [0, T]
    :return: i.i.d. samples
    :rtype: numpy.ndarray
    """
    y_svvj = rSVVJ(v0, mu, k, theta, sigma, rho, T, jsize_v, jtime_v, n, n_seqment)
    #
    IZ = np.empty(n)
    for i in range(n):
        numJ = rng.poisson(lmbd_s * T)
        jsize_y = rng.normal(mu_s, sigma_s, numJ)
        IZ[i] = sum(jsize_y)
    return y_svvj + IZ


if __name__ == "__main__":
    import sys

    v0, h = 0.007569, 1
    r, k, theta, sigma, rho = 0.0319, 3.46, 0.008, 0.14, -0.82
    lmbd_v, mu_v = 0.47, 0.05  # same as that in mdl_svcj
    #
    mu_bar = -0.12
    lmbd_s, sigma_s = 0.11, 0.15
    mu_s = math.log(1 + mu_bar) - sigma_s ** 2 / 2
    # same as that in SVJ in Broadie-Kaya (2006)
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
    y = rSVIJ(v0, r, k, theta, sigma, rho, h, jsize_v, jtime_v,
              lmbd_s, mu_s, sigma_s, n, n_segment)
    print(f"\nGiven v0 = {v0} and {numJ} jumps in the variance, ")
    print(f"jsize_v = {jsize_v}, \njtime_v = {jtime_v}")
    print(f"\n{n} i.i.d. samples from SVIJ: ")
    print(y)

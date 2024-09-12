"""
Module for generating samples from :abbr:`SVVJ(Stochastic Volatility with
jumps in the Variance)`

Note that we are generating i.i.d. samples with fixed conditions instead of
a trajectory of samples!
"""
import math
import numpy as np

rng = np.random.default_rng()


def IZ(t, jumpsize, jumptime):
    f = 0
    if len(jumpsize) == 0: return f
    i = 0
    while i < len(jumpsize) and jumptime[i] <= t:
        f += jumpsize[i]
        i += 1
    return f


def IEZ(t, jumpsize, jumptime, k):
    f = 0
    if len(jumpsize) == 0: return f
    i = 0
    while i < len(jumpsize) and jumptime[i] <= t:
        f += math.exp(k * jumptime[i]) * jumpsize[i]
        i += 1
    return f


def incIEII(v0, t0, dt, k, theta, sigma):
    # IE = \int_0^t e^{ks} \sqrt{v_s} dw_s^v
    # I =  \int_0^t        \sqrt{v_s} dw_s^v
    # I2 = \int_0^t        \sqrt{v_s} dw_s
    # dv_t = k(theta - v_t)dt + sigma * sqrt(v_t)dw_t^v + dz_t
    eps_v = math.sqrt(dt) * rng.normal(0, 1)
    eps_y = math.sqrt(dt) * rng.normal(0, 1)
    #
    vt = v0 + k * (theta - v0) * dt + sigma * math.sqrt(v0) * eps_v
    #
    if vt < 0: vt = 0
    #
    I = math.sqrt(v0) * eps_v
    IE = math.exp(k * t0) * math.sqrt(v0) * eps_v
    I2 = math.sqrt(v0) * eps_y
    return IE, I, I2, vt


def rIEII(v0, k, theta, sigma, h, jumpsize, jumptime, n_segment):
    IE, I, I2 = 0, 0, 0
    dt = h / n_segment
    #
    if len(jumptime) == 0:
        for i in range(n_segment):
            t0 = i * dt
            incIE, incI, incI2, vt = incIEII(v0, t0, dt, k, theta, sigma)
            IE += incIE
            I += incI
            I2 += incI2
            v0 = vt
        return IE, I, I2
    # at least one jump
    j = 0
    for i in range(n_segment):
        t0 = i * dt
        t = (i + 1) * dt
        while j < len(jumptime) and jumptime[j] <= t:
            # print(f"within {i+1}-th segment: [{i*dt}, {(i+1)*dt})")
            # print(f"the {j+1}-th jump, jump size = {jumpsize[j]}, jump time = {jumptime[j]}")
            # pre jump
            dt1 = jumptime[j] - t0
            incIE, incI, incI2, vt = incIEII(v0, t0, dt1, k, theta, sigma)
            IE += incIE
            I += incI
            I2 += incI2
            # jump
            vt += jumpsize[j]
            #
            t0 = jumptime[j]
            v0 = vt
            j += 1
        if t0 < t:
            dt1 = t - t0
            incIE, incI, incI2, vt = incIEII(v0, t0, dt1, k, theta, sigma)
            IE += incIE
            I += incI
            I2 += incI2
            #
            v0 = vt
    return IE, I, I2


def rSVVJ(v0, mu, k, theta, sigma, rho, T, jumpsize, jumptime, n, n_segment):
    """Generate i.i.d. Random samples from mdl_svvj by Euler approximation.

    :param float v0: value of the initial variance :math:`v(0)`
    :param float mu: parameter :math:`\mu`
    :param float k: parameter :math:`k`
    :param float theta: parameter :math:`\theta`
    :param float sigma: parameter :math:`\sigma_v`
    :param float rho: parameter :math:`\rho`
    :param float T: end time
    :param tuple jumpsize: jump sizes over [0, T]
    :param tuple jumptime: jump time points over [0, T]
    :param int n: number of samples to generate
    :param int n_segment: number of segments to split [0, T]
    :return: i.i.d. samples
    :rtype: numpy.ndarray
    """
    y = np.empty(n)
    beta = (1 - math.exp(-k * T)) / (2 * k)
    c = (mu - theta / 2) * T - (v0 - theta) * beta
    c += math.exp(-k * T) * IEZ(T, jumpsize, jumptime, k) / (2 * k)
    c -= IZ(T, jumpsize, jumptime) / (2 * k)
    #
    for i in range(n):
        IE, I, I2 = rIEII(v0, k, theta, sigma, T, jumpsize, jumptime, n_segment)
        y[i] = c + math.exp(-k * T) * IE * sigma / (2 * k)
        y[i] += (rho - sigma / (2 * k)) * I
        y[i] += math.sqrt(1 - rho ** 2) * I2
    return y


if __name__ == "__main__":
    import sys

    v0, h = 0.007569, 1
    mu, k, theta, sigma, rho = 0.0319, 3.46, 0.008, 0.14, -0.82
    lmbd, mu_v = 0.47, 0.05
    #
    args = sys.argv[1:]
    numJ = rng.poisson(lmbd * h) if len(args) == 0 else int(args[0])
    jumpsize = rng.exponential(mu_v, numJ)
    jumptime = rng.uniform(0, h, numJ)
    jumptime = sorted(jumptime)
    # change to tuples
    jumpsize, jumptime = tuple(jumpsize), tuple(jumptime)
    #
    n, n_segment = 20, 10
    y = rSVVJ(v0, mu, k, theta, sigma, rho, h, jumpsize, jumptime, n, n_segment)
    print(f"\nGiven v0 = {v0} and {numJ} jumps, ")
    print(f"jumpsize = {jumpsize}, \njumptime = {jumptime}")
    print(f"\n{n} i.i.d. samples from SVVJ: ")
    print(y)

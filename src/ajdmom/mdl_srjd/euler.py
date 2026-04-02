"""
Module for generating samples from :abbr:`SRJD(Square-Root Jump Diffusion)`

Note that we are generating i.i.d. samples with fixed conditions instead of
a trajectory of samples!
"""
import math
import numpy as np

rng = np.random.default_rng()


def rSRD(v0, tau, k, theta, sigma):
    r"""Generate next Random sample from Square-Root Diffusion

    :param float v0: current state value, > 0
    :param float tau: time difference between current and next state
    :param float k: parameter :math:`k`
    :param float theta: parameter :math:`\theta`
    :param float sigma: parameter :math:`\sigma`
    :return: next sample
    :rtype: float
    """
    d = 4 * theta * k / (sigma ** 2)  # degrees of freedom
    c = (sigma ** 2) * (1 - math.exp(-k * tau)) / (4 * k)  # leading constant
    lmbd = v0 * math.exp(-k * tau) / c  # non-centrality parameter
    return c * rng.noncentral_chisquare(d, lmbd)


def rSRJD(v0, k, theta, sigma, T, jumpsize, jumptime, n):
    r"""Generate i.i.d. Random Samples from mdl_srjd given v0 and jumps

    :param float v0: value of the initial state :math: `v_0 > 0`
    :param float k: parameter :math:`k`
    :param float theta: parameter :math:`\theta`
    :param float sigma: parameter :math:`\sigma`
    :param float T: end time
    :param tuple jumpsize: jump sizes over the interval [0,T]
    :param tuple jumptime: jump time points over the interval [0,T]
    :param int n: number of samples to generate
    :return: i.i.d. samples
    :rtype: numpy.ndarray
    """
    x = np.empty(n)
    if not isinstance(jumpsize, tuple):
        jumpsize = tuple(jumpsize)
    if not isinstance(jumptime, tuple):
        jumptime = tuple(jumptime)
    numJ = len(jumpsize)
    if numJ == 0:  # diffusion only
        for i in range(n): x[i] = rSRD(v0, T, k, theta, sigma)
        return x
    #
    # now numJ  >= 1
    #
    dt = (jumptime[0],) + tuple(jumptime[i] - jumptime[i - 1] for i in range(1, numJ))
    v0_original = v0
    for i in range(n):
        v0 = v0_original
        for j in range(numJ):
            vt = rSRD(v0, dt[j], k, theta, sigma) + jumpsize[j]
            v0 = vt
        # residual time
        t_residual = T - jumptime[-1]
        if t_residual > 0: vt = rSRD(v0, t_residual, k, theta, sigma)
        x[i] = vt
    return x


def rSRJD_exact(v0, k, theta, sigma, lmbd, mu_v, T, n):
    """
    Generate n i.i.d. samples of V_T from a Square-Root Jump Diffusion.
    The jumps follow a Poisson process with intensity 'lmbd'
    and sizes following an Exponential distribution with mean 'mu_v'.

    :param float v0: Initial variance
    :param float k: Speed of mean reversion
    :param float theta: Long-run mean (diffusion part)
    :param float sigma: Volatility of volatility
    :param float lmbd: Jump intensity (Poisson rate)
    :param float mu_v: Mean jump size (Exponential distribution mean)
    :param float T: Time horizon
    :param int n: Number of samples to generate
    :return: numpy.ndarray of shape (n,) containing samples of V_T
    """
    samples = np.empty(n)

    for i in range(n):
        # 1. Sample the total number of jumps in [0, T]
        num_jumps = rng.poisson(lmbd * T)

        if num_jumps == 0:
            # Case: No jumps, default to pure CIR diffusion
            samples[i] = rSRD(v0, T, k, theta, sigma)
        else:
            # 2. Sample jump times (Uniformly distributed given num_jumps)
            jump_times = sorted(rng.uniform(0, T, num_jumps))

            # 3. Sample jump sizes (Exponentially distributed)
            jump_sizes = rng.exponential(mu_v, num_jumps)

            # 4. Path-wise simulation between jump events
            current_v = v0
            current_t = 0

            for j in range(num_jumps):
                # Diffuse from the last event to the current jump time
                dt = jump_times[j] - current_t
                current_v = rSRD(current_v, dt, k, theta, sigma)

                # Apply the instantaneous jump
                current_v += jump_sizes[j]
                current_t = jump_times[j]

            # 5. Diffuse from the last jump to the final time T
            t_residual = T - current_t
            if t_residual > 0:
                current_v = rSRD(current_v, t_residual, k, theta, sigma)

            samples[i] = current_v

    return samples


if __name__ == "__main__":
    import sys

    v0, h = 0.007569, 1
    k, theta, sigma = 3.46, 0.008, 0.14
    mu_v, lmbd = 0.05, 0.47
    #
    args = sys.argv[1:]
    numJ = rng.poisson(lmbd * h) if len(args) == 0 else int(args[0])
    jumpsize = tuple(rng.exponential(mu_v, numJ))
    jumptime = rng.uniform(0, h, numJ)
    jumptime = tuple(sorted(jumptime))
    #
    n = 20
    x = rSRJD(v0, k, theta, sigma, h, jumpsize, jumptime, n)
    print(f"Given v0 = {v0} and {numJ} jumps, ")
    print(f"jumpsize = {tuple(jumpsize)}, \njumptime = {tuple(jumptime)}")
    print(f"\n{n} replicate samples from SRJD: ")
    print(x)

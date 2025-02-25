import numpy as np
import pytest

from ajdmom.mdl_srjd.euler import rSRJD
from ajdmom.mdl_srjd.cond2_mom import m
from ajdmom.mdl_srjd.cond2_cmom import cm

rng = np.random.default_rng()


@pytest.fixture
def settings():
    v0, h = 0.007569, 1
    k, theta, sigma = 3.46, 0.008, 0.14
    mu_v, lmbd = 0.05, 0.47
    #
    rel_err = 0.100  # 10%
    abs_err = 0.001
    par = {'v0': v0, 'h': h, 'k': k, 'theta': theta, 'sigma': sigma, 'mu_v': mu_v, 'lmbd': lmbd}
    return par, rel_err, abs_err


def test_m(settings):
    par, rel_err, abs_err = settings
    #
    k, theta, sigma = par['k'], par['theta'], par['sigma']
    lmbd, mu_v = par['lmbd'], par['mu_v']
    v0, h = par['v0'], par['h']
    #
    for numJ in [0, 1, 2, 3]:  # numJ = rng.poisson(lmbd * h): the actual number of jumps
        jumpsize = rng.exponential(mu_v, numJ)
        jumptime = rng.uniform(0, h, numJ)  # unsorted
        jumpsize, jumptime = tuple(jumpsize), tuple(sorted(jumptime))  # set as tuples
        par['jumpsize'] = jumpsize
        par['jumptime'] = jumptime
        #
        x = rSRJD(v0, k, theta, sigma, h, jumpsize, jumptime, 100 * 1000)
        print(f"\nWith conditions v0 = {v0}, \njumpsize = {jumpsize}, \njumptime = {jumptime}")
        for n in [1, 2, 3, 4, 5]:
            expected = np.mean(x ** n)
            expected = pytest.approx(expected, rel=rel_err, abs=abs_err)
            actual = m(n, par)
            msg = "Diff in the {}th conditional moment is > {:} or {:.0%} "
            msg += "between theory and sample."
            devi = max(expected.abs, expected.rel * abs(expected.expected))
            temp = "{:d}-th moment: actual({: f}) V.S. expected({: f} ± {:f})"
            print(temp.format(n, actual, expected.expected, devi))
            assert actual == expected, msg.format(n, abs_err, rel_err)


def test_cm(settings):
    par, rel_err, abs_err = settings
    #
    k, theta, sigma = par['k'], par['theta'], par['sigma']
    lmbd, mu_v = par['lmbd'], par['mu_v']
    v0, h = par['v0'], par['h']
    #
    for numJ in [0, 1, 2, 3]:  # numJ = rng.poisson(lmbd * h): the actual number of jumps
        jumpsize = rng.exponential(mu_v, numJ)
        jumptime = rng.uniform(0, h, numJ)
        jumpsize, jumptime = tuple(jumpsize), tuple(sorted(jumptime))
        par['jumpsize'] = jumpsize
        par['jumptime'] = jumptime
        #
        x = rSRJD(v0, k, theta, sigma, h, jumpsize, jumptime, 100 * 1000)
        print(f"\nWith conditions v0 = {v0}, \njumpsize = {jumpsize}, \njumptime = {jumptime}")
        diff = x - np.mean(x)
        del x
        for n in [2, 3, 4, 5]:
            expected = np.mean(diff ** n)
            expected = pytest.approx(expected, rel=rel_err, abs=abs_err)
            actual = cm(n, par)
            msg = "Diff in the {}th conditional central moment is > {:} or {:.0%}"
            msg += "between theory and sample."
            devi = max(expected.abs, expected.rel * abs(expected.expected))
            temp = "{:d}-th central moment: actual({: f}) V.S. expected({: f} ± {:f})"
            print(temp.format(n, actual, expected.expected, devi))
            assert actual == expected, msg.format(n, abs_err, rel_err)

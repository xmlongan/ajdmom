import numpy as np
import pytest
import math

from ajdmom.mdl_svij.euler import rSVIJ
from ajdmom.mdl_svij.cond2_cmom import cm
from ajdmom.mdl_svij.cond2_mom import m

rng = np.random.default_rng()


@pytest.fixture
def settings():
    v0, h = 0.007569, 1
    mu, k, theta, sigma, rho = 0.0319, 3.46, 0.008, 0.14, -0.82
    lmbd_v, mu_v = 0.47, 0.05  # same as that in mdl_svcj
    #
    mu_bar = -0.12
    lmbd_s, sigma_s = 0.11, 0.15
    mu_s = math.log(1 + mu_bar) - sigma_s ** 2 / 2
    # same as that in SVJ in Broadie-Kaya (2006)
    rel_err = 0.100 # 10%
    abs_err = 0.001
    par = {'v0': v0, 'h': h, 'mu': mu, 'k': k, 'theta': theta,
           'sigma': sigma, 'rho': rho, 'lmbd_v': lmbd_v, 'mu_v': mu_v,
           'lmbd_s': lmbd_s, 'mu_s': mu_s, 'sigma_s': sigma_s}
    return par, rel_err, abs_err


def test_rSVIJ(settings):
    par, rel_err, abs_err = settings
    #
    k, theta, sigma, rho = par['k'], par['theta'], par['sigma'], par['rho']
    v0, h, mu = par['v0'], par['h'], par['mu']
    lmbd_v, mu_v = par['lmbd_v'], par['mu_v']
    lmbd_s, mu_s, sigma_s = par['lmbd_s'], par['mu_s'], par['sigma_s']
    #
    for numJ in [0, 1, 2, 3]:
        jsize_v = rng.exponential(mu_v, numJ)
        jtime_v = rng.uniform(0, h, numJ)  # unsorted
        jsize_v, jtime_v = tuple(jsize_v), tuple(sorted(jtime_v))  # set as tuples
        par['jumpsize'] = jsize_v
        par['jumptime'] = jtime_v
        n = 100 * 1000
        n_segment = [10, 50, 100]
        # n_segment = [1, 5, 10]
        # n_segment = range(1, 11)
        tm = [m(j, par) for j in [1, 2, 3, 4, 5]]
        print(f"\nnumJ = {numJ}, \njumpsize = {jsize_v}, \njumptime = {jtime_v}")
        temp = "tmean = [{: f}, {: f}, {: f}, {: f}, {: f}]"
        print(temp.format(tm[0], tm[1], tm[2], tm[3], tm[4]))
        for i in range(len(n_segment)):
            y = rSVIJ(v0, mu, k, theta, sigma, rho, h, jsize_v, jtime_v,
                      lmbd_s, mu_s, sigma_s, n, n_segment[i])
            sm = []
            for j in [1, 2, 3, 4, 5]:
                sm.append(np.mean(y ** j))
            temp = "smean = [{: f}, {: f}, {: f}, {: f}, {: f}], n_segment = {:d}"
            print(temp.format(sm[0], sm[1], sm[2], sm[3], sm[4], n_segment[i]))


def test_m(settings):
    par, rel_err, abs_err = settings
    #
    k, theta, sigma, rho = par['k'], par['theta'], par['sigma'], par['rho']
    v0, h, mu = par['v0'], par['h'], par['mu']
    lmbd_v, mu_v = par['lmbd_v'], par['mu_v']
    lmbd_s, mu_s, sigma_s = par['lmbd_s'], par['mu_s'], par['sigma_s']
    #
    for numJ in [0, 1, 2, 3]:  # numJ = rng.poisson(lmbd * h): the actual number of jumps
        jsize_v = rng.exponential(mu_v, numJ)
        jtime_v = rng.uniform(0, h, numJ)  # unsorted
        jsize_v, jtime_v = tuple(jsize_v), tuple(sorted(jtime_v))  # set as tuples
        par['jumpsize'] = jsize_v
        par['jumptime'] = jtime_v
        #
        n, n_segment = 100 * 1000, 100
        y = rSVIJ(v0, mu, k, theta, sigma, rho, h, jsize_v, jtime_v,
                  lmbd_s, mu_s, sigma_s, n, n_segment)
        print(f"\nWith conditions v0 = {v0}, \njumpsize = {jsize_v}, \njumptime = {jtime_v}")
        for n in [1, 2, 3, 4, 5]:
            expected = np.mean(y ** n)
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
    k, theta, sigma, rho = par['k'], par['theta'], par['sigma'], par['rho']
    v0, h, mu = par['v0'], par['h'], par['mu']
    lmbd_v, mu_v = par['lmbd_v'], par['mu_v']
    lmbd_s, mu_s, sigma_s = par['lmbd_s'], par['mu_s'], par['sigma_s']
    #
    for numJ in [0, 1, 2, 3]:  # numJ = rng.poisson(lmbd * h): the actual number of jumps
        jsize_v = rng.exponential(mu_v, numJ)
        jtime_v = rng.uniform(0, h, numJ)  # unsorted
        jsize_v, jtime_v = tuple(jsize_v), tuple(sorted(jtime_v))  # set as tuples
        par['jumpsize'] = jsize_v
        par['jumptime'] = jtime_v
        #
        n, n_segment = 100 * 1000, 100
        y = rSVIJ(v0, mu, k, theta, sigma, rho, h, jsize_v, jtime_v,
                  lmbd_s, mu_s, sigma_s, n, n_segment)
        print(f"\nWith conditions v0 = {v0}, \njumpsize = {jsize_v}, \njumptime = {jtime_v}")
        diff = y - np.mean(y)
        del y
        for n in [2, 3, 4, 5]:
            expected = np.mean(diff ** n)
            expected = pytest.approx(expected, rel=rel_err, abs=abs_err)
            actual = cm(n, par)
            msg = "Diff in the {}th conditional central moment is > {:} or {:.0%} "
            msg += "between theory and sample."
            devi = max(expected.abs, expected.rel * abs(expected.expected))
            temp = "{:d}-th central moment: actual({: f}) V.S. expected({: f} ± {:f})"
            print(temp.format(n, actual, expected.expected, devi))
            assert actual == expected, msg.format(n, abs_err, rel_err)

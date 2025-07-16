import pytest
import numpy as np
from fractions import Fraction as Frac

from ajdmom.poly import Poly
from ajdmom.ito_cond2_mom import moment_IEII, poly2num
# from ajdmom.ito_cond2_mom_old import moment_IEII, poly2num
from ajdmom.mdl_svvj.euler import rIEII

from ajdmom.ito_mom import moment_IEII as moment_IEII_2
from ajdmom.ito_mom import poly2num as poly2num_2

rng = np.random.default_rng()


@pytest.fixture
def moments():
    kf = ['e^{kt}', 't', 'k^{-}', 'v0-theta', 'theta', 'sigma', 'l_{1:n}', 'o_{1:n}']
    moms = {}
    P0 = Poly()
    P0.set_keyfor(kf)
    #
    mom300 = Poly({
        (2, 0, 2, 1, 0, 1, (), ()): Frac(3, 2),
        (1, 0, 2, 1, 0, 1, (), ()): -Frac(3, 1),
        (0, 0, 2, 1, 0, 1, (), ()): Frac(3, 2),
        (3, 0, 2, 0, 1, 1, (), ()): Frac(1, 2),
        (0, 0, 2, 0, 1, 1, (), ()): Frac(1, 1),
        (1, 0, 2, 0, 1, 1, (), ()): -Frac(3, 2),
        (2, 0, 2, 0, 0, 1, (0,), (0,)): Frac(3, 2),
        (0, 0, 2, 0, 0, 1, (2,), (0,)): Frac(3, 2),
        (1, 0, 2, 0, 0, 1, (1,), (0,)): -Frac(3, 1),
    })
    mom300.set_keyfor(kf)
    #
    mom210 = Poly({
        (1, 1, 1, 1, 0, 1, (), ()): Frac(1, 1),
        (1, 0, 2, 0, 1, 1, (), ()): -Frac(1, 1),
        (0, 1, 1, 1, 0, 1, (), ()): -Frac(2, 1),
        (0, 1, 1, 0, 1, 1, (), ()): -Frac(1, 1),
        (1, 0, 2, 1, 0, 1, (), ()): Frac(1, 1),
        (0, 0, 2, 1, 0, 1, (), ()): -Frac(1, 1),
        (2, 0, 2, 0, 1, 1, (), ()): Frac(1, 1),
        (1, 1, 1, 0, 0, 1, (0,), (0,)): Frac(1, 1),
        (1, 0, 1, 0, 0, 1, (0,), (1,)): -Frac(1, 1),
        (1, 0, 2, 0, 0, 1, (0,), (0,)): Frac(1, 1),
        (0, 0, 2, 0, 0, 1, (1,), (0,)): -Frac(1, 1),
        (0, 1, 1, 0, 0, 1, (1,), (0,)): -Frac(2, 1),
        (0, 0, 1, 0, 0, 1, (1,), (1,)): Frac(2, 1)
    })
    mom210.set_keyfor(kf)
    #
    mom120 = Poly({
        (0, 1, 1, 1, 0, 1, (), ()): Frac(1, 1),
        (-1, 0, 2, 1, 0, 1, (), ()): Frac(1, 1),
        (0, 0, 2, 1, 0, 1, (), ()): -Frac(1, 1),
        (0, 2, 0, 1, 0, 1, (), ()): Frac(1, 1),
        (1, 0, 2, 0, 1, 1, (), ()): Frac(5, 2),
        (0, 0, 2, 0, 1, 1, (), ()): -Frac(3, 1),
        (-1, 0, 2, 0, 1, 1, (), ()): Frac(1, 2),
        (0, 1, 1, 0, 1, 1, (), ()): -Frac(2, 1),
        (0, 1, 1, 0, 0, 1, (0,), (0,)): Frac(1, 1),
        (0, 0, 1, 0, 0, 1, (0,), (1,)): -Frac(1, 1),
        (-1, 0, 2, 0, 0, 1, (1,), (0,)): Frac(1, 1),
        (0, 0, 2, 0, 0, 1, (0,), (0,)): -Frac(1, 1),
        (0, 2, 0, 0, 0, 1, (0,), (0,)): Frac(1, 1),
        (0, 0, 0, 0, 0, 1, (0,), (2,)): Frac(1, 1),
        (0, 1, 0, 0, 0, 1, (0,), (1,)): -Frac(2, 1)
    })
    mom120.set_keyfor(kf)
    #
    mom102 = Poly({
        (0, 1, 1, 1, 0, 1, (), ()): Frac(1, 1),
        (-1, 0, 2, 1, 0, 1, (), ()): Frac(1, 1),
        (0, 0, 2, 1, 0, 1, (), ()): -Frac(1, 1),
        (1, 0, 2, 0, 1, 1, (), ()): Frac(1, 2),
        (0, 0, 2, 0, 1, 1, (), ()): -Frac(1, 1),
        (-1, 0, 2, 0, 1, 1, (), ()): Frac(1, 2),
        (0, 1, 1, 0, 0, 1, (0,), (0,)): Frac(1, 1),
        (0, 0, 1, 0, 0, 1, (0,), (1,)): -Frac(1, 1),
        (-1, 0, 2, 0, 0, 1, (1,), (0,)): Frac(1, 1),
        (0, 0, 2, 0, 0, 1, (0,), (0,)): -Frac(1, 1)
    })
    mom102.set_keyfor(kf)
    #
    mom030 = Poly({
        (-1, 1, 1, 1, 0, 1, (), ()): -Frac(3, 1),
        (-1, 0, 2, 1, 0, 1, (), ()): -Frac(3, 1),
        (0, 0, 2, 1, 0, 1, (), ()): Frac(3, 1),
        (0, 1, 1, 0, 1, 1, (), ()): Frac(3, 1),
        (-1, 0, 2, 0, 1, 1, (), ()): Frac(3, 1),
        (0, 0, 2, 0, 1, 1, (), ()): -Frac(3, 1),
        (-1, 1, 1, 0, 0, 1, (0,), (0,)): -Frac(3, 1),
        (-1, 0, 2, 0, 0, 1, (0,), (0,)): -Frac(3, 1),
        (0, 0, 2, 0, 0, 1, (-1,), (0,)): Frac(3, 1),
        (-1, 0, 1, 0, 0, 1, (0,), (1,)): Frac(3, 1)
    })
    mom030.set_keyfor(kf)
    #
    mom012 = Poly({
        (-1, 1, 1, 1, 0, 1, (), ()): -Frac(1, 1),
        (-1, 0, 2, 1, 0, 1, (), ()): -Frac(1, 1),
        (0, 0, 2, 1, 0, 1, (), ()): Frac(1, 1),
        (0, 1, 1, 0, 1, 1, (), ()): Frac(1, 1),
        (-1, 0, 2, 0, 1, 1, (), ()): Frac(1, 1),
        (0, 0, 2, 0, 1, 1, (), ()): -Frac(1, 1),
        (-1, 1, 1, 0, 0, 1, (0,), (0,)): -Frac(1, 1),
        (-1, 0, 2, 0, 0, 1, (0,), (0,)): -Frac(1, 1),
        (0, 0, 2, 0, 0, 1, (-1,), (0,)): Frac(1, 1),
        (-1, 0, 1, 0, 0, 1, (0,), (1,)): Frac(1, 1)
    })
    mom012.set_keyfor(kf)
    #
    moms[(3, 0, 0)] = mom300
    moms[(2, 1, 0)] = mom210
    moms[(2, 0, 1)] = P0
    moms[(1, 2, 0)] = mom120
    moms[(1, 1, 1)] = P0
    moms[(1, 0, 2)] = mom102
    moms[(0, 3, 0)] = mom030
    moms[(0, 2, 1)] = P0
    moms[(0, 1, 2)] = mom012
    moms[(0, 0, 3)] = P0
    return moms


def test_moment_ieii(moments):
    for key in moments:
        expected = moments[key]
        actual = moment_IEII(key[0], key[1], key[2])
        assert actual == expected


@pytest.fixture
def settings():
    v0, h = 0.007569, 1
    mu, k, theta, sigma, rho = 0.0319, 3.46, 0.008, 0.14, -0.82
    lmbd, mu_v = 0.47, 0.05
    k = 1.0
    #
    rel_err = 0.100  # 10%
    abs_err = 0.001
    par = {'v0': v0, 'h': h, 'mu': mu, 'k': k, 'theta': theta, 'sigma': sigma,
           'rho': rho, 'lmbd': lmbd, 'mu_v': mu_v}
    return par, rel_err, abs_err


def test_moment_ieii_numeric(settings):
    par, rel_err, abs_err = settings
    #
    k, theta, sigma, rho = par['k'], par['theta'], par['sigma'], par['rho']
    lmbd, mu_v = par['lmbd'], par['mu_v']
    v0, h, mu = par['v0'], par['h'], par['mu']
    #
    # orders = [(0,0,0), (1,0,0), (2,0,0), (3,0,0), (4,0,0)]  # close
    # orders = [(0,0,0), (0,1,0), (0,2,0), (0,3,0), (0,4,0)]  # close
    orders = [(1,1,0), (2,1,0), (1,2,0), (3,1,0), (4,1,0)]  # not close (3,1,0)
    moms = [moment_IEII(order[0], order[1], order[2]) for order in orders]
    moms_2 = [moment_IEII_2(order[0], order[1], order[2]) for order in orders]
    for numJ in [0, 1, 2, 3]:
        jsize_v = rng.exponential(mu_v, numJ)
        jtime_v = rng.uniform(0, h, numJ)  # unsorted
        jsize_v, jtime_v = tuple(jsize_v), tuple(sorted(jtime_v))  # set as tuples
        par['jumpsize'] = jsize_v
        par['jumptime'] = jtime_v
        n = 100 * 10000
        n_segment = [10, 50, 100]
        tm = [poly2num(mom, par) for mom in moms]
        print(f"\nnumJ = {numJ}, \njumpsize = {jsize_v}, \njumptime = {jtime_v}")
        temp = "tmean = [{: f}, {: f}, {: f}, {: f}, {: f}]"
        print(temp.format(tm[0], tm[1], tm[2], tm[3], tm[4]))

        if numJ == 0:
            tm2 = [poly2num_2(mom, par) for mom in moms_2]
            temp = "tmean2= [{: f}, {: f}, {: f}, {: f}, {: f}]"
            print(temp.format(tm2[0], tm2[1], tm2[2], tm2[3], tm2[4]))

        for i in range(len(n_segment)):
            IE, I, I2 = np.empty(n), np.empty(n), np.empty(n)
            for j in range(n):
                IE[j], I[j], I2[j] = rIEII(v0, k, theta, sigma, h, jsize_v, jtime_v, n_segment[i])
            sm = []
            for order in orders:
                sm.append(np.mean((IE ** order[0]) * (I ** order[1]) * (I2 ** order[2])))
            temp = "smean = [{: f}, {: f}, {: f}, {: f}, {: f}], n_segment = {:d}"
            print(temp.format(sm[0], sm[1], sm[2], sm[3], sm[4], n_segment[i]))

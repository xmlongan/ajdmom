import pytest
import math
from ajdmom.poly import Poly

from ajdmom.utils import comb, dbfactorial, cmnorm, simplify_rho


def test_comb():
    expected = math.comb(5, 1) * math.comb(5 - 1, 2)
    actual = comb(5, [1, 2, 2])
    assert actual == expected


def test_dbfactorial():
    n = [0, 1, 2, 3, 10, 11]
    expected = [1, 1, 2, 3, 3840, 10395]
    for i in range(len(n)):
        assert dbfactorial(n[i]) == expected[i]


def test_cmnorm():
    coef = [1, 0, 1, 0, 3, 0, 15, 0, 105]
    power = [0, 0, 2, 0, 4, 0, 6, 0, 8]
    for i in range(len(coef)):
        cmom = cmnorm(i)
        for key in cmom:
            pow_actual, coef_actual = key[0], cmom[key]
            assert coef_actual == coef[i]
            assert pow_actual == power[i]


def test_simplify_rho():
    kf = ['rho', 'sqrt(1-rho^2)']
    poly = Poly({
        (1, 0): 1,
        (0, 2): 1
    })
    poly.set_keyfor(kf)
    #
    poly_expected = Poly({
        (0,): 1,
        (1,): 1,
        (2,): -1
    })
    poly_expected.set_keyfor(kf)
    #
    poly_actual = simplify_rho(poly, 1)
    assert poly_actual == poly_expected

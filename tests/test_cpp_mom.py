import pytest

from ajdmom.poly import Poly
from ajdmom.cpp_mom import mnorm
from ajdmom.cpp_mom import mcpp
from ajdmom.cpp_mom import cmcpp


@pytest.mark.parametrize(
    "n, poly",
    [
        (0, {(0, 0): 1}),
        (1, {(1, 0): 1}),
        (2, {(2, 0): 1, (0, 1): 1}),
        (3, {(3, 0): 1, (1, 1): 3}),
        (8, {(8, 0): 1, (6, 1): 28, (4, 2): 210, (2, 3): 420, (0, 4): 105})
    ]
)
def test_mnorm(n, poly):
    """Test moment function of normal distribution"""
    kf = ('mu', 'sigma^2')
    expected = Poly(poly)
    expected.set_keyfor(kf)
    #
    assert mnorm(n) == expected


@pytest.mark.parametrize(
    "n, poly",
    [
        (0, {(0, 0, 0): 1}),
        (1, {(1, 1, 0): 1}),
        (2, {(1, 2, 0): 1, (1, 0, 2): 1, (2, 2, 0): 1}),
        (3, {(2, 3, 0): 3, (2, 1, 2): 3, (1, 3, 0): 1, (1, 1, 2): 3,
             (3, 3, 0): 1}),
        (4, {(1, 0, 4): 3,
             (1, 2, 2): 6,
             (1, 4, 0): 1,
             (2, 0, 4): 3,
             (2, 2, 2): 18,
             (2, 4, 0): 7,
             (3, 2, 2): 6,
             (3, 4, 0): 6,
             (4, 4, 0): 1})
    ]
)
def test_mcpp(n, poly):
    """Test moment of Compound Poisson Process"""
    kf = ('lambda*h', 'mu', 'sigma')
    expected = Poly(poly)
    expected.set_keyfor(kf)
    #
    assert mcpp(n) == expected


@pytest.mark.parametrize(
    "n, poly",
    [
        (0, {(0, 0, 0): 1}),
        (1, {}),
        (2, {(1, 2, 0): 1, (1, 0, 2): 1}),
        (3, {(1, 3, 0): 1, (1, 1, 2): 3}),
        (4, {(1, 0, 4): 3,
             (1, 2, 2): 6,
             (1, 4, 0): 1,
             (2, 0, 4): 3,
             (2, 2, 2): 6,
             (2, 4, 0): 3})
    ]
)
def test_cmcpp(n, poly):
    """Test central moment of Compound Poisson Process"""
    kf = ('lambda*h', 'mu', 'sigma')
    expected = Poly(poly)
    expected.set_keyfor(kf)
    #
    assert cmcpp(n) == expected

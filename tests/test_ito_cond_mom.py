import pytest
from fractions import Fraction as Frac

from ajdmom.poly import Poly
from ajdmom.ito_cond_mom import moment_IEII


@pytest.fixture
def moments():
    kf = ['e^{kt}', 't', 'k^{-}', 'v0-theta', 'theta', 'sigma']
    kf += ['l_{1:n}', 'o_{1:n}', 'p_{2:n}', 'q_{2:n}']
    moms = {}
    P0 = Poly()
    P0.set_keyfor(kf)
    #
    mom300 = Poly({
        (2, 0, 2, 1, 0, 1, (), (), (), ()): Frac(3, 2),
        (1, 0, 2, 1, 0, 1, (), (), (), ()): -Frac(3, 1),
        (0, 0, 2, 1, 0, 1, (), (), (), ()): Frac(3, 2),
        (3, 0, 2, 0, 1, 1, (), (), (), ()): Frac(1, 2),
        (0, 0, 2, 0, 1, 1, (), (), (), ()): Frac(1, 1),
        (1, 0, 2, 0, 1, 1, (), (), (), ()): -Frac(3, 2),
        (2, 0, 2, 0, 0, 1, (1,), (0,), (), ()): Frac(3, 2),
        (0, 0, 2, 0, 0, 1, (3,), (0,), (), ()): Frac(3, 2),
        (1, 0, 2, 0, 0, 1, (2,), (0,), (), ()): -Frac(3, 1),
    })
    mom300.set_keyfor(kf)
    #
    mom210 = Poly({
        (1, 1, 1, 1, 0, 1, (), (), (), ()): Frac(1, 1),
        (1, 0, 2, 0, 1, 1, (), (), (), ()): -Frac(1, 1),
        (0, 1, 1, 1, 0, 1, (), (), (), ()): -Frac(2, 1),
        (0, 1, 1, 0, 1, 1, (), (), (), ()): -Frac(1, 1),
        (1, 0, 2, 1, 0, 1, (), (), (), ()): Frac(1, 1),
        (0, 0, 2, 1, 0, 1, (), (), (), ()): -Frac(1, 1),
        (2, 0, 2, 0, 1, 1, (), (), (), ()): Frac(1, 1),
        (1, 1, 1, 0, 0, 1, (1,), (0,), (), ()): Frac(1, 1),
        (1, 0, 1, 0, 0, 1, (1,), (1,), (), ()): -Frac(1, 1),
        (1, 0, 2, 0, 0, 1, (1,), (0,), (), ()): Frac(1, 1),
        (0, 0, 2, 0, 0, 1, (2,), (0,), (), ()): -Frac(1, 1),
        (0, 1, 1, 0, 0, 1, (2,), (0,), (), ()): -Frac(2, 1),
        (0, 0, 1, 0, 0, 1, (2,), (1,), (), ()): Frac(2, 1)
    })
    mom210.set_keyfor(kf)
    #
    mom120 = Poly({
        (0, 1, 1, 1, 0, 1, (), (), (), ()): Frac(1, 1),
        (-1, 0, 2, 1, 0, 1, (), (), (), ()): Frac(1, 1),
        (0, 0, 2, 1, 0, 1, (), (), (), ()): -Frac(1, 1),
        (0, 2, 0, 1, 0, 1, (), (), (), ()): Frac(1, 1),
        (1, 0, 2, 0, 1, 1, (), (), (), ()): Frac(5, 2),
        (0, 0, 2, 0, 1, 1, (), (), (), ()): -Frac(3, 1),
        (-1, 0, 2, 0, 1, 1, (), (), (), ()): Frac(1, 2),
        (0, 1, 1, 0, 1, 1, (), (), (), ()): -Frac(2, 1),
        (0, 1, 1, 0, 0, 1, (1,), (0,), (), ()): Frac(1, 1),
        (0, 0, 1, 0, 0, 1, (1,), (1,), (), ()): -Frac(1, 1),
        (-1, 0, 2, 0, 0, 1, (2,), (0,), (), ()): Frac(1, 1),
        (0, 0, 2, 0, 0, 1, (1,), (0,), (), ()): -Frac(1, 1),
        (0, 2, 0, 0, 0, 1, (1,), (0,), (), ()): Frac(1, 1),
        (0, 0, 0, 0, 0, 1, (1,), (2,), (), ()): Frac(1, 1),
        (0, 1, 0, 0, 0, 1, (1,), (1,), (), ()): -Frac(2, 1)
    })
    mom120.set_keyfor(kf)
    #
    mom102 = Poly({
        (0, 1, 1, 1, 0, 1, (), (), (), ()): Frac(1, 1),
        (-1, 0, 2, 1, 0, 1, (), (), (), ()): Frac(1, 1),
        (0, 0, 2, 1, 0, 1, (), (), (), ()): -Frac(1, 1),
        (1, 0, 2, 0, 1, 1, (), (), (), ()): Frac(1, 2),
        (0, 0, 2, 0, 1, 1, (), (), (), ()): -Frac(1, 1),
        (-1, 0, 2, 0, 1, 1, (), (), (), ()): Frac(1, 2),
        (0, 1, 1, 0, 0, 1, (1,), (0,), (), ()): Frac(1, 1),
        (0, 0, 1, 0, 0, 1, (1,), (1,), (), ()): -Frac(1, 1),
        (-1, 0, 2, 0, 0, 1, (2,), (0,), (), ()): Frac(1, 1),
        (0, 0, 2, 0, 0, 1, (1,), (0,), (), ()): -Frac(1, 1)
    })
    mom102.set_keyfor(kf)
    #
    mom030 = Poly({
        (-1, 1, 1, 1, 0, 1, (), (), (), ()): -Frac(3, 1),
        (-1, 0, 2, 1, 0, 1, (), (), (), ()): -Frac(3, 1),
        (0, 0, 2, 1, 0, 1, (), (), (), ()): Frac(3, 1),
        (0, 1, 1, 0, 1, 1, (), (), (), ()): Frac(3, 1),
        (-1, 0, 2, 0, 1, 1, (), (), (), ()): Frac(3, 1),
        (0, 0, 2, 0, 1, 1, (), (), (), ()): -Frac(3, 1),
        (-1, 1, 1, 0, 0, 1, (1,), (0,), (), ()): -Frac(3, 1),
        (-1, 0, 2, 0, 0, 1, (1,), (0,), (), ()): -Frac(3, 1),
        (0, 0, 2, 0, 0, 1, (0,), (0,), (), ()): Frac(3, 1),
        (-1, 0, 1, 0, 0, 1, (1,), (1,), (), ()): Frac(3, 1)
    })
    mom030.set_keyfor(kf)
    #
    mom012 = Poly({
        (-1, 1, 1, 1, 0, 1, (), (), (), ()): -Frac(1, 1),
        (-1, 0, 2, 1, 0, 1, (), (), (), ()): -Frac(1, 1),
        (0, 0, 2, 1, 0, 1, (), (), (), ()): Frac(1, 1),
        (0, 1, 1, 0, 1, 1, (), (), (), ()): Frac(1, 1),
        (-1, 0, 2, 0, 1, 1, (), (), (), ()): Frac(1, 1),
        (0, 0, 2, 0, 1, 1, (), (), (), ()): -Frac(1, 1),
        (-1, 1, 1, 0, 0, 1, (1,), (0,), (), ()): -Frac(1, 1),
        (-1, 0, 2, 0, 0, 1, (1,), (0,), (), ()): -Frac(1, 1),
        (0, 0, 2, 0, 0, 1, (0,), (0,), (), ()): Frac(1, 1),
        (-1, 0, 1, 0, 0, 1, (1,), (1,), (), ()): Frac(1, 1)
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

"""
Moments of the SRJD model

Unconditional moments of the SRJD model.
"""
import math
from fractions import Fraction

from ajdmom.poly import Poly
from ajdmom.mdl_srjd.cmom import cmoment_v


def expand(poly):
    """expand the 'E[v]' in moment polys

    :param Poly poly: moment poly with ``keyfor`` =
      ('k^{-}', 'E[v]', 'sigma', 'lmbd', 'mu_v')
    :return: poly with ``keyfor`` =
      ('k^{-}', 'theta', 'sigma', 'lmbd', 'mu_v')
    :rtype: Poly
    """
    kf = ['k^{-}', 'theta', 'sigma', 'lmbd', 'mu_v']
    poln = Poly()
    poln.set_keyfor(kf)
    #
    for k, v in poly.items():
        # E[v] = theta + lmbd * mu_v / k
        n = k[1]
        if n == 0:
            poln.add_keyval(k, v)
            continue
        for i in range(n + 1):
            bino = math.comb(n, i)
            key = (k[0] + i, n - i, k[2], k[3] + i, k[4] + i)
            poln.add_keyval(key, bino * v)
    return poln

def moment_v(n, simplify = True):
    r"""moment of the state :math:`v_t`

    unconditional moment, noting that :math:`v_t = (v_t - E[v_t]) + E[v_t]`,
    the frist moment :math:`E[v] = \theta + \lambda\mu_v/k`

    :param int n: order of the unconditional moment
    :param bool simplify: whether to simplify the moment
    :return: if simplify = True, poly with ``keyfor`` =
      ('k^{-}', 'theta', 'sigma', 'lmbd', 'mu_v'),
      otherwise,
      ('k^{-}', 'E[v]', 'sigma', 'lmbd', 'mu_v')
    :rtype: Poly
    """
    if n < 0:
        raise ValueError('n must be 0 or positive')
    kf = ['k^{-}', 'E[v]', 'sigma', 'lmbd', 'mu_v']
    poly = Poly()
    poly.set_keyfor(kf)
    #
    if n == 0:
        poly.add_keyval((0, 0, 0, 0, 0), Fraction(1, 1))
        return poly if not simplify else expand(poly)
    if n == 1:
        poly.add_keyval((0, 1, 0, 0, 0), Fraction(1, 1))
        return poly if not simplify else expand(poly)
    for i in range(n + 1):
        bino = math.comb(n, i)
        poln = cmoment_v(i)
        for k, v in poln.items():
            key = list(k)
            key[1] += n - i  # (E[v_t])^{n-i}
            key = tuple(key)
            poly.add_keyval(key, bino * v)
    return poly if not simplify else expand(poly)


if __name__ == "__main__":
    from pprint import pprint

    print('\nExample usage of the module functions\n')
    n = 4
    mom = moment_v(n, simplify = False)
    print(f'moment_v({n}) = ')
    pprint(mom)
    print(f"{mom.keyfor}")

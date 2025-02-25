"""
Central Moments of :math:`IEZ_{s,t}`
"""
import math
from fractions import Fraction

from ajdmom.poly import Poly
from ajdmom.mdl_srjd.iez_mom import moment_IEZ, poly2num

def cmoment_IEZ(n):
    r"""Central moment of :math:`E[\overline{IEZ}_{s,t}^n]`

    :param int n: order of the central moment
    :return: poly with ``keyfor`` = ('e^{kt}', 'e^{ks}', 'k^{-}', 'lmbd', 'mu_v')
    :rtype: Poly
    """
    if n < 0:
        raise ValueError(f"moment_IEZ({n}) is called!")
    kf = ['e^{kt}', 'e^{ks}', 'k^{-}', 'lmbd', 'mu_v']
    poly = Poly()
    poly.set_keyfor(kf)
    #
    p0 = Poly({(0, 0, 0, 0, 0): Fraction(0, 1)})
    p0.set_keyfor(kf)
    p1 = Poly({(0, 0, 0, 0, 0): Fraction(1, 1)})
    p1.set_keyfor(kf)
    if n == 0:
        return p1
    if n == 1:
        return p0
    mean = moment_IEZ(1)
    for i in range(n + 1):
        sign = (-1) ** i
        bino = math.comb(n, i)
        pol1 = p1 if i == 0 else mean ** i
        poln = (sign * bino) * (pol1 * moment_IEZ(n - i))
        poly.merge(poln)
    return poly


def cm(n, par):
    cmoment = cmoment_IEZ(n)
    value = poly2num(cmoment)
    return value


if __name__ == '__main__':
    from pprint import pprint

    print('\nExample usage of the module functions')
    n = 3
    print(f"cmoment_IEZ({n}) = ")
    pprint(cmoment_IEZ(n))
    print("which is a poly with keyfor = ('e^{kt}', 'e^{ks}', 'k^{-}', 'lmbd', 'mu_v')")

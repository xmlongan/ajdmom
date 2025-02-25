"""
Central Moments of the SVCJ model
"""
import math
from fractions import Fraction

from ajdmom.poly import Poly
from ajdmom.mdl_svcj.mom import moment_y, poly2num

def cmoment_y(n):
    """central moment of :math:`y_t`

    Central moment :math:`E[\bar{y}_t^n] = E[(y_t - E[y_t])^n]`

    :param integer n: order of the central moment
    :return: poly with ``keyfor`` = ('e^{kt}', 't', 'k^{-}',
      'mu', 'E[v]', 'theta', 'sigma', 'rho',
      'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s')
    :rtype: Poly
    """
    if n < 0:
        raise ValueError('n must be 0 or positive')
    mean_y = moment_y(1)
    kf = ['e^{kt}', 't', 'k^{-}',
          'mu', 'E[v]', 'theta', 'sigma', 'rho',
          'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s']
    poly = Poly()
    poly.set_keyfor(kf)
    #
    p0 = Poly({(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): Fraction(0, 1)})
    p0.set_keyfor(kf)
    p1 = Poly({(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): Fraction(1, 1)})
    p1.set_keyfor(kf)
    #
    if n == 0:
        return p1
    if n == 1:
        return p0
    # typical case
    for i in range(n+1):
        bino = math.comb(n, i)
        sign = (-1) ** i
        pol1 = p1 if i == 0 else mean_y ** i
        poln = (bino * sign) * (pol1 * moment_y(n - i))
        poly.merge(poln)
    return poly

def cm_y(n, par):
    cmoment = cmoment_y(n)
    value = poly2num(cmoment, par)
    return value

def remove_mu_column(poly):
    poly.remove_zero()
    kf = ['e^{kt}', 't', 'k^{-}',
          #'mu',
          'E[v]', 'theta', 'sigma', 'rho',
          'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s']
    poln = Poly()
    poln.set_keyfor(kf)
    for k, v in poly.items():
        if k[3] == 0:
            key = tuple(k[:3]) + tuple(k[4:])
            poln.add_keyval(key, v)
        else:
            raise Exception(f'k = {k} where k[3] != 0, i.e., mu exist')
    return poln


if __name__ == '__main__':
    from pprint import pprint

    n = 4
    cmom = cmoment_y(n)
    cmom = remove_mu_column(cmom)
    print(f"cmoment_y({n}) =")
    pprint(cmom)
    print(f"{len(cmom)} rows of {cmom.keyfor}")

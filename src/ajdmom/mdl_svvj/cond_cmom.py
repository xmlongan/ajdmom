"""
Conditional Central Moments of the SVVJ model, given initial variance
"""
import math
from fractions import Fraction

from ajdmom import Poly
from ajdmom.utils import comb, simplify_rho
from ajdmom.mdl_svcj.cond_ieii_ieziziz_mom import moment_ieii_ieziziz


def b_n(n1, n2, n3, n4, n5, n6, n7):
    kf = ['e^{kt}', 't', 'k^{-}',
          'mu', 'v0-theta', 'theta', 'sigma', 'rho', 'sqrt(1-rho^2)',
          'lmbd', 'mu_v']
    poly = Poly()
    poly.set_keyfor(kf)
    for i in range(n2 + 1):
        for j in range(n6 + 1):
            bino = math.comb(n2, i) * math.comb(n6, j)
            sign = (-1) ** (i + n5 + n6 + j)
            c = sign * Fraction(bino, 2 ** (n1 + i + n4 + n5 + n6 + n7))
            key = (-(n1 + n4) - j, n7, n1 + i + n4 + n5 + 2*n6 + n7,
                   0, 0, 0, n1 + i, n2 - i, n3,
                   n6 + n7, n6 + n7)
            poly.add_keyval(key, c)
    return poly


def cmoment_y(n):
    """conditional central moment of the SVVJ model :math:`y_t|v_0`

    :param integer n: order of the conditional moment
    :return: poly with ``keyfor`` =  ('e^{kt}', 't', 'k^{-}',
      'mu', 'v0-theta', 'theta', 'sigma', 'rho',
      'lmbd', 'mu_v')
    :rtype: Poly
    """
    kf = ['e^{kt}', 't', 'k^{-}',
          'mu', 'v0-theta', 'theta', 'sigma', 'rho',
          'lmbd', 'mu_v']
    if n < 0:
        raise ValueError('n must be non-negative')
    p0 = Poly({(0, 0, 0, 0, 0, 0, 0, 0, 0, 0): Fraction(0, 1)})
    p0.set_keyfor(kf)
    p1 = Poly({(0, 0, 0, 0, 0, 0, 0, 0, 0, 0): Fraction(1, 1)})
    p1.set_keyfor(kf)
    # special case
    if n == 0:
        return p1
    elif n == 1:
        return p0
    # typical cases
    kf.insert(8, 'sqrt(1-rho^2)')
    poly = Poly()
    poly.set_keyfor(kf)
    for n1 in range(n + 1):
        for n2 in range(n - n1 + 1):
            for n3 in range(n - n1 - n2 + 1):
                for n4 in range(n - n1 - n2 - n3 + 1):
                    for n5 in range(n - n1 - n2 - n3 - n4 + 1):
                        for n6 in range(n - n1 - n2 - n3 - n4 - n5 + 1):
                            n7 = n - n1 - n2 - n3 - n4 - n5 - n6
                            c = comb(n, [n1, n2, n3, n4, n5, n6, n7])
                            pol1 = b_n(n1, n2, n3, n4, n5, n6, n7)
                            # kf = ['e^{kt}', 't', 'k^{-}',
                            #       'mu', 'v0-theta', 'theta', 'sigma', 'rho', 'sqrt(1-rho^2)',
                            #        'lmbd', 'mu_v']
                            pol2 = moment_ieii_ieziziz(n1, n2, n3, n4, n5, 0)
                            # kf = ['e^{kt}','t','k^{-}',
                            #       'v0-theta','theta','sigma',
                            #       'lmbd','mu_v','mu_s','sigma_s']
                            poln = Poly()
                            poln.set_keyfor(pol1.keyfor)
                            for k, v in pol2.items():
                                key = (k[0], k[1], k[2],
                                       0, k[3], k[4], k[5], 0, 0,
                                       k[6], k[7])
                                poln.add_keyval(key, v)
                            poly.merge(c * (pol1 * poln))
    return simplify_rho(poly, 8)


def poly2num(poly, par):
    # kf = ['e^{kt}', 't', 'k^{-}',
    #       'mu', 'v0-theta', 'theta', 'sigma', 'rho',
    #       'lmbd', 'mu_v']
    k = par['k']
    mu, v0, theta, sigma, rho = par['mu'], par['v0'], par['theta'], par['sigma'], par['rho']
    lmbd, mu_v = par['lmbd'], par['mu_v']
    h = par['h']
    #
    f = 0
    for K, V in poly.items():
        val = math.exp(K[0] * k * h) * (h ** K[1]) / (k ** K[2])
        val *= (mu ** K[3]) * ((v0 - theta) ** K[4]) * (theta ** K[5])
        val *= (sigma ** K[6]) * (rho ** K[7])
        val *= (lmbd ** K[8]) * (mu_v ** K[9])
        f += val * V
    return f


def cm_y(n, par):
    cmoment = cmoment_y(n)
    value = poly2num(cmoment, par)
    return value


if __name__ == '__main__':
    from pprint import pprint

    n = 3
    poly = cmoment_y(n)
    print(f"cmoment_y({n}) = "); pprint(poly); print(f"which is a poly with keyfor = {poly.keyfor}")

    v0 = 0.007569
    k = 3.46
    theta = 0.008
    sigma = 0.14
    rho = -0.82
    r = 0.0319
    lmbd = 0.47
    mu_v = 0.05

    mu = r
    h = 1

    par = {'v0': v0, 'mu': mu, 'k': k, 'theta': theta, 'sigma': sigma, 'rho': rho,
           'lmbd': lmbd, 'mu_v': mu_v,
           'h': h}

    for n in range(1, 4+1):
        print(f"cmoment({n}) = {cm_y(n, par)}")

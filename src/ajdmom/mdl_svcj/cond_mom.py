"""
Conditional Moments of the SVCJ model, given initial variance
"""
import math
from fractions import Fraction

from ajdmom.poly import Poly
from ajdmom.utils import comb
from ajdmom.mdl_svcj.cond_ieii_ieziziz_mom import moment_ieii_ieziziz


def b_n(n1, n2, n3, n4, n5, n6, n7, n8):
    kf = ['e^{kt}', 't', 'k^{-}',
          'mu', 'v0-theta', 'theta', 'sigma', 'rho', 'sqrt(1-rho^2)',
          'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s']
    poly = Poly()
    poly.set_keyfor(kf)
    for i1 in range(n2 + 1):
        for i2 in range(n5 + 1):
            for i3 in range(n7 + 1):
                for i4 in range(n8 + 1):
                    bino1 = math.comb(n2, i1)
                    bino2 = math.comb(n5, i2)
                    bino3 = math.comb(n7, i3)
                    bino4 = math.comb(n8, i4)
                    bino = bino1 * bino2 * bino3 * bino4
                    sign = (-1) ** (i1 + i2 + i3 + i4)
                    c = sign * Fraction(bino, 2 ** (n1 + n4 + i1 + i2 + i3 + n8))
                    #
                    key = (-(n1 + n4 + n8 - i4), n7, n1 + n4 + i1 + i2 + n8,
                           n7 - i3, n8, i3, n1 + i1, n2 - i1, n3,
                           0, 0, n5 - i2, 0, 0)
                    poly.add_keyval(key, c)
    return poly


def enlarge(poly, new_keyfor):
    # it requires n < m
    n, m = len(poly.keyfor), len(new_keyfor)
    index = []  # indexes of elements of poly.keyfor in new_keyfor
    for k in poly.keyfor:
        index.append(new_keyfor.index(k))
        # if not found, it will raise an error
    poln = Poly()
    poln.set_keyfor(new_keyfor)
    for k, v in poly.items():
        key = [0 for i in range(m)]
        for i in range(n):
            key[index[i]] = k[i]
        poln.add_keyval(tuple(key), v)
    return poln


def expand_sqrt(poly):
    """expand sqrt(1-rho^2) in poly key"""
    # kf = ['e^{kt}', 't', 'k^{-}',
    #       'mu', 'v0-theta', 'theta', 'sigma', 'rho', 'sqrt(1-rho^2)',
    #       'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s']
    kf = ['e^{kt}', 't', 'k^{-}',
          'mu', 'v0-theta', 'theta', 'sigma', 'rho',
          'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s']
    poln = Poly()
    poln.set_keyfor(kf)
    for k, v in poly.items():
        if k[8] % 2 != 0:
            raise Exception("the power of sqrt(1-rho^2) is not even!")
        for i in range(k[8] // 2 + 1):
            key = list(k)
            key[7] += 2 * i
            del key[8]
            key = tuple(key)
            sign = (-1) ** i
            val = sign * math.comb(k[8] // 2, i) * v
            poln.add_keyval(key, val)
    return poln


def moment_y(n):
    """conditional moment of :math:`y_t|v_0`

    :param integer n: order of the conditional moment
    :return: poly with ``keyfor`` = ('e^{kt}', 't', 'k^{-}',
      'mu', 'v0-theta', 'theta', 'sigma', 'rho',
      'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s')
    :rtype: Poly
    """
    kf = ['e^{kt}', 't', 'k^{-}',
          'mu', 'v0-theta', 'theta', 'sigma', 'rho',
          'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s']
    if n < 0:
        raise ValueError('n must be non-negative')
    p1 = Poly({(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): Fraction(1, 1)})
    p1.set_keyfor(kf)
    # special case
    if n == 0:
        return p1
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
                            for n7 in range(n - n1 - n2 - n3 - n4 - n5 - n6 + 1):
                                n8 = n - n1 - n2 - n3 - n4 - n5 - n6 - n7
                                c = comb(n, [n1, n2, n3, n4, n5, n6, n7, n8])
                                # print(f"({n1}, {n2}, {n3}, {n4}, {n5}, {n6}, {n7}, {n8}), c = {c}")
                                pol1 = b_n(n1, n2, n3, n4, n5, n6, n7, n8)
                                # kf = ['e^{kt}', 't', 'k^{-}',
                                #       'mu', 'v0-theta', 'theta', 'sigma', 'rho', 'sqrt(1-rho^2)',
                                #        'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s']
                                # print("b_n = "); pprint(pol1); print(f"{pol1.keyfor}")
                                pol2 = moment_ieii_ieziziz(n1, n2, n3, n4, n5, n6)
                                # kf = ['e^{kt}','t','k^{-}',
                                #       'v0-theta','theta','sigma',
                                #       'lmbd','mu_v','mu_s','sigma_s']
                                # print("moment_ieii_ieziziz = "); pprint(pol2); print(f"{pol2.keyfor}\n")
                                pol2 = enlarge(pol2, pol1.keyfor)
                                poly.merge(c * (pol1 * pol2))
    return expand_sqrt(poly)  # expand sqrt(1-rho^2)


def poly2num(poly, par):
    # kf = ['e^{kt}', 't', 'k^{-}',
    #       'mu', 'v0-theta', 'theta', 'sigma', 'rho',
    #       'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s']
    k = par['k']
    mu, v0, theta, sigma, rho = par['mu'], par['v0'], par['theta'], par['sigma'], par['rho']
    lmbd, mu_v, rhoJ, mu_s, sigma_s = par['lmbd'], par['mu_v'], par['rhoJ'], par['mu_s'], par['sigma_s']
    h = par['h']
    #
    f = 0
    for K, V in poly.items():
        val = math.exp(K[0] * k * h) * (h ** K[1]) / (k ** K[2])
        val *= (mu ** K[3]) * ((v0 - theta) ** K[4]) * (theta ** K[5])
        val *= (sigma ** K[6]) * (rho ** K[7])
        val *= (lmbd ** K[8]) * (mu_v ** K[9]) * (rhoJ ** K[10])
        val *= (mu_s ** K[11]) * (sigma_s ** K[12])
        f += val * V
    return f


def m_y(n, par):
    moment = moment_y(n)
    value = poly2num(moment, par)
    return value


if __name__ == '__main__':
    from pprint import pprint

    n = 3
    poly = moment_y(n)
    # print(f"moment_y({n}) = "); pprint(poly); print(f"which is a poly with keyfor = {poly.keyfor}")

    v0 = 0.007569
    k = 3.46
    theta = 0.008
    sigma = 0.14
    rho = -0.82
    r = 0.0319
    lmbd = 0.47
    mu_b = -0.1
    sigma_s = 0.0001
    mu_v = 0.05
    rhoJ = -0.38

    mu = r - lmbd * mu_b
    mu_s = math.log((1 + mu_b) * (1 - rhoJ * mu_v)) - sigma_s ** 2 / 2
    h = 1

    par = {'v0': v0, 'mu': mu, 'k': k, 'theta': theta, 'sigma': sigma, 'rho': rho,
           'lmbd': lmbd, 'mu_v': mu_v, 'rhoJ': rhoJ, 'mu_s': mu_s, 'sigma_s': sigma_s,
           'h': h}

    for n in range(1, 4+1):
        print(f"moment({n}) = {m_y(n, par)}")

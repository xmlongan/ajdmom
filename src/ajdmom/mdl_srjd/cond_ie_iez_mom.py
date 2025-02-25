"""
Conditional Moments of :math:`E[IE_t^{n_1} IEZ_t^{n_2}|v_0]`
"""
import math
from fractions import Fraction

from ajdmom.poly import Poly
from ajdmom.mdl_srjd.iez_mom import moment_IEZ


def iez_to_ie_iez(poly):
    # from
    # kf = ['e^{kt}', 'e^{ks}', 'k^{-}', 'lmbd', 'mu_v']
    # to (s = 0)
    kf = ['e^{kt}', 'k^{-}', 'v0-theta', 'theta', 'sigma', 'lmbd', 'mu_v']
    poln = Poly()
    poln.set_keyfor(kf)
    for k, v in poly.items():
        key = (k[0], k[2], 0, 0, 0, k[3], k[4])
        poln.add_keyval(key, v)
    return poln

def int_e_poly(c, tp, m, poly):
    poln = Poly()
    poln.set_keyfor(['e^{kt}', 'k^{-}', 'v0-theta', 'theta', 'sigma', 'lmbd', 'mu_v'])
    for k, v in poly.items():
        n = k[0] + m  # \int_0^t e^{n ks} ds = e^{n kt} /(nk) - 1/(nk)
        if n == 0:
            raise ValueError(r"n = 0, the integral should be \int_0^t ds = t!")
        key = list(k)
        key[0] = n
        key[1] += 1
        if tp in [1, 2, 3]: key[tp + 1] += 1
        key = tuple(key)
        poln.add_keyval(key, c * v * Fraction(1, n))
        #
        key = list(k)
        key[0] = 0
        key[1] += 1
        if tp in [1, 2, 3]: key[tp + 1] += 1
        key = tuple(key)
        poln.add_keyval(key, c * v * Fraction(-1, n))
    return poln

def recursive_ie(n, IE, IE_IEZ):
    poly = Poly()
    poly.set_keyfor(['e^{kt}', 'k^{-}', 'v0-theta', 'theta', 'sigma', 'lmbd', 'mu_v'])
    if n >= 2:
        c = Fraction(n * (n - 1), 2)
        poly.merge(int_e_poly(c, 1, 1, IE[n - 2]))
        poly.merge(int_e_poly(c, 2, 2, IE[n - 2]))
        poly.merge(int_e_poly(c, 3, 1, IE[n - 1]))
        poly.merge(int_e_poly(c, 4, 1, IE_IEZ[(n - 2, 1)]))
    return poly

def moment_ie(n):
    """Moment of :math:`E[IE_t^n]`

    :param integer n: order of the moment

    :return: poly with ``keyfor`` =
      ('e^{kt}', 'k^{-}', 'v0-theta', 'theta', 'sigma', 'lmbd', 'mu_v')
    :rtype: Poly
    """
    if n < 0:
        raise ValueError(f"moment_ie({n}) is called!")
    #
    # IE: a dict of moments of E[IE^n]
    #
    IE = {}
    #
    kf = ['e^{kt}', 'k^{-}', 'v0-theta', 'theta', 'sigma', 'lmbd', 'mu_v']
    #
    # special poly constants, analog to 0 and 1
    #
    P0 = Poly({(0, 0, 0, 0, 0, 0, 0): Fraction(0, 1)})
    P0.set_keyfor(kf)
    P1 = Poly({(0, 0, 0, 0, 0, 0, 0): Fraction(1, 1)})
    P1.set_keyfor(kf)
    #
    # n = 0: special case
    #
    IE[0] = P1  # equiv to constant 1
    #
    # n = 1
    #
    IE[1] = P0  # equiv to constant 0
    #
    P2 = Poly({
        (1, 1, 1, 0, 0, 0, 0): Fraction(1, 1),
        (0, 1, 1, 0, 0, 0, 0): Fraction(-1, 1),
        (2, 1, 0, 1, 0, 0, 0): Fraction(1, 2),
        (0, 1, 0, 1, 0, 0, 0): Fraction(-1, 2),
        (2, 2, 0, 0, 0, 1, 1): Fraction(1, 2),
        (1, 2, 0, 0, 0, 1, 1): Fraction(-1, 1),
        (0, 2, 0, 0, 0, 1, 1): Fraction(1, 2)
    })
    IE[2] = P2
    #
    if n <= 2:
        return IE[n]
    #
    IE_IEZ = {}
    #
    IE_IEZ[(0, 1)] = iez_to_ie_iez(moment_IEZ(1))
    #
    IE_IEZ[(1, 1)] = P0
    #
    # n >= 3: typical cases
    #
    if n > 3:
        # compute all lower-order moments to get ready
        for i in range(3, n):
            # print(f"moment_ie({n}), i = {i}")
            if i > 3: IE_IEZ[(i - 2, 1)] = moment_ie_iez(i - 2, 1)
            poly = recursive_ie(i, IE, IE_IEZ)
            poly.remove_zero()
            IE[i] = poly
            # delete polys no more needed
            del IE[i - 2]
            del IE_IEZ[(i - 2, 1)]
    # the last one
    if n > 3: IE_IEZ[(n - 2, 1)] = moment_ie_iez(n - 2, 1)
    poly = recursive_ie(n, IE, IE_IEZ)
    poly.remove_zero()
    return poly

def key_times_poly(k, poly):
    # k: ['e^{kt}', 'e^{ks}', 'k^{-}', 'lmbd', 'mu_v']
    kf = ['e^{kt}', 'k^{-}', 'v0-theta', 'theta', 'sigma', 'lmbd', 'mu_v']
    poln = Poly()
    poln.set_keyfor(kf)
    for key, val in poly.items():
        knw = (k[0] + key[0], k[2] + key[1], key[2], key[3], key[4],
               k[3] + key[5], k[4] + key[6])
        poln.add_keyval(knw, val)
    return poln

def recursive_ie_iez(n1, n2, IE_IEZ):
    poly = Poly()
    poly.set_keyfor(['e^{kt}', 'k^{-}', 'v0-theta', 'theta', 'sigma', 'lmbd', 'mu_v'])
    if n1 >= 2 and n2 >= 1:
        for i in range(n2 + 1):
            bino = math.comb(n2, i)
            pol1 = moment_IEZ(n2 - i)  # ['e^{kt}', 'e^{ks}', 'k^{-}', 'lmbd', 'mu_v']
            for k, v in pol1.items():
                coef = bino * Fraction(n1 * (n1 - 1), 2) * v
                pol2 = int_e_poly(coef, 1, 1 + k[1], IE_IEZ[(n1 - 2, i)])
                pol2 += int_e_poly(coef, 2, 2 + k[1], IE_IEZ[(n1 - 2, i)])
                pol2 += int_e_poly(coef, 3, 1 + k[1], IE_IEZ[(n1 - 1, i)])
                pol2 += int_e_poly(coef, 4, 1 + k[1], IE_IEZ[(n1 - 2, i + 1)])
                poly.merge(key_times_poly(k, pol2))
    if n1 >= 2 and n2 == 0:
        poly = moment_ie(n1)
    if n1 == 1:  # no matter what value n2 take, the result is 0
        key, val = (0, 0, 0, 0, 0, 0, 0), Fraction(0, 1)
        poly.add_keyval(key, val)
    if n1 == 0:  # works even n2 = 0
        poly = iez_to_ie_iez(moment_IEZ(n2))
    return poly

def moment_ie_iez(n1, n2):
    """Moment of :math:`E[IE_t^{n_1} IEZ_t^{n_2}|v_0]`

    :param integer n1: order of :math:`IE_t`
    :param integer n2: order of :math:`IEZ_t`

    :return: poly with ``keyfor`` =
        ('e^{kt}', 'k^{-}', 'v0-theta', 'theta', 'sigma', 'lmbd', 'mu_v')
    :rtype: Poly
    """
    if not (n1 >= 0 and n2 >= 0):
        raise ValueError(f'moment_ie_iez({n1}, {n2}) is called!')
    #
    # IE_IEZ: a dict of moments of E[IE^{n1} IEZ^{n2}]
    #
    IE_IEZ = {}
    #
    kf = ['e^{kt}', 'k^{-}', 'v0-theta', 'theta', 'sigma', 'lmbd', 'mu_v']
    #
    # special poly constants, analog to 0 and 1
    #
    P0 = Poly({(0, 0, 0, 0, 0, 0, 0): Fraction(0, 1)})
    P0.set_keyfor(kf)
    P1 = Poly({(0, 0, 0, 0, 0, 0, 0): Fraction(1, 1)})
    P1.set_keyfor(kf)
    #
    # n1 + n2 = 0: special case
    #
    IE_IEZ[(0, 0)] = P1  # equiv to constant 1
    #
    # n1 + n2 = 1
    #
    IE_IEZ[(1, 0)] = P0  # equiv to constant 0
    IE_IEZ[(0, 1)] = iez_to_ie_iez(moment_IEZ(1))
    #
    # n1 + n2 = 2
    #
    IE_IEZ[(2, 0)] = moment_ie(2)
    IE_IEZ[(1, 1)] = P0
    IE_IEZ[(0, 2)] = iez_to_ie_iez(moment_IEZ(2))
    #
    if n1 + n2 <= 2:
        return IE_IEZ[(n1, n2)]
    #
    # n1 + n2 >= 3: typical cases
    #
    if n1 + n2 > 3:
        # compute all lower-order moments to get ready
        for n in range(3, n1 + n2):
            for i in range(n + 1):
                poly = recursive_ie_iez(i, n - i, IE_IEZ)
                poly.remove_zero()
                IE_IEZ[(i, n - i)] = poly
            # seems almost all polys should be reserved
    # the last one
    poly = recursive_ie_iez(n1, n2, IE_IEZ)
    poly.remove_zero()
    return poly

def poly2num(poly, par):
    # 'e^{kt}', 'k^{-}', 'v0-theta', 'theta', 'sigma', 'lmbd', 'mu_v'
    v0, k, theta, sigma = par['v0'], par['k'], par['theta'], par['sigma']
    lmbd, mu_v, h = par['lmbd'], par['mu_v'], par['h']
    f = 0
    for K, v in poly.items():
        val = math.exp(K[0] * k * h) / (k ** K[1])
        val *= ((v0-theta) ** K[2]) * (theta ** K[3]) * (sigma ** K[4])
        val *= (lmbd ** K[5]) * (mu_v ** K[6])
        f += val * v
    return f

def m_ie(n, par):
    moment = moment_ie(n)
    value = poly2num(moment, par)
    return value

def m_ie_iez(n1, n2, par):
    moment = moment_ie_iez(n1, n2)
    value = poly2num(moment, par)
    return value


if __name__ == '__main__':
    from pprint import pprint

    # pprint(iez_to_ie_iez(moment_IEZ(n)))

    # n = 6
    # print(f"moment_ie(n) = ")
    # moment_ie(10)
    # pprint(moment_ie(n))

    print('\nExample usage of the module functions')
    kf = ['e^{kt}', 'k^{-}', 'v0-theta', 'theta', 'sigma', 'lmbd', 'mu_v']
    n1 = 3
    n2 = 2
    print(f"\nmoment_ie_iez(n1, n2) returns a poly with keyfor = \n{kf}\n")
    print(f'moment_ie_iez({n1}, {n2}) = ')
    pprint(moment_ie_iez(n1, n2))

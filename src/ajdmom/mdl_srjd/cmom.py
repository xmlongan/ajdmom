"""
Central Moments of the SRJD model

Unconditional central moments of the SRJD model.
"""
import math
from fractions import Fraction

from ajdmom.poly import Poly
from ajdmom.mdl_srjd.cond_ie_iez_cmom import cmoment_ie_iez

def cond_cmoment_v(n):
    """Conditional central moment :math:`E[(v_t - E[v])^n|v0]`

    Different from the cmoment_v() in 'cond_cmom.py' in the ``keyfor``
    attribute.

    :param int n: order of the conditional central moment
    :return: poly with ``keyfor`` = ('e^{kt}', 'k^{-}', 'v0-E[v]',
      'E[v]', 'sigma', 'lmbd', 'mu_v')
    :rtype: Poly
    """
    kf = ['e^{kt}', 'k^{-}', 'v0-E[v]', 'E[v]', 'sigma', 'lmbd', 'mu_v']
    poly = Poly()
    poly.set_keyfor(kf)
    #
    p1 = Poly({(0, 0, 0, 0, 0, 0, 0): Fraction(1, 1)})
    p1.set_keyfor(kf)
    if n == 0:
        return p1
    if n == 1:
        key = (-1, 0, 1, 0, 0, 0, 0)
        poly.add_keyval(key, Fraction(1, 1))
        return poly
    for i1 in range(n + 1):
        for i2 in range(n - i1 + 1):
            i3 = n - i1 - i2
            bino = math.comb(n, i1) * math.comb(n - i1, i2)
            poln = cmoment_ie_iez(i1, i2)
            for k, v in poln.items():
                key = list(k)
                key[0] -= n   # e^{-nkt}
                key[2] += i3  # v0-E[v]
                key[4] += i1  # sigma
                key = tuple(key)
                poly.add_keyval(key, v * bino)
    return poly

def find_same_order(poly, n):
    """Check the poly does have one and only one monomial with order n

    :param Poly poly: conditional central moment of the state with
      ``keyfor`` = ('e^{kt}', 'k^{-}', 'v0-E[v]', 'E[v]', 'sigma',
      'lmbd', 'mu_v')
    :param integer n: order of :math:`v0-E[v]`
    :return: True or raise an error
    :rtype: bool
    """
    count = 0
    target_key = (-n, 0, n, 0, 0, 0, 0)
    target_order = n
    for k, v in poly.items():
        if k[2] == target_order:
            count += 1
    if count == 1 and poly[target_key] ==  Fraction(1, 1):
        return True
    else:
        msg = "Either multiple of monomials (v0-E[v])^n, or coefficient not correct"
        raise Exception(msg)

def key_times_poly(coef, k, poly):
    # k   0        1          2        3        4        5       6
    # ['e^{kt}', 'k^{-}', 'v0-E[v]', 'E[v]', 'sigma', 'lmbd', 'mu_v']
    #                     to expand
    #
    # poly: expanded E[(v0-E[v])^i] with keyfor =
    # ['k^{-}', 'E[v]', 'sigma', 'lmbd', 'mu_v']
    #
    kf = ['e^{kt}', 'k^{-}', 'E[v]', 'sigma', 'lmbd', 'mu_v']
    poln = Poly()
    poln.set_keyfor(kf)
    for K, V in poly.items():
        key = (k[0],
               k[1] + K[0],
               k[3] + K[1],
               k[4] + K[2],
               k[5] + K[3],
               k[6] + K[4])
        poln.add_keyval(key, coef * V)
    return poln

def last_expectation(poly, uncond_cm):
    # poly with keyfor =
    # ['e^{kt}', 'k^{-}', 'v0-E[v]', 'E[v]', 'sigma', 'lmbd', 'mu_v']
    #                    expectation
    #
    kf = ['e^{kt}', 'k^{-}', 'E[v]', 'sigma', 'lmbd', 'mu_v']
    poln = Poly()
    poln.set_keyfor(kf)
    #
    for k, v in poly.items():
        order = k[2]
        if order == 0:   # constant
            key = k[0:2] + k[3:]
            poln.add_keyval(key, v)
        # order == 1, with expectation = 0
        elif order > 1:
            poln.merge(key_times_poly(v, k, uncond_cm[order]))
    return poln

def divide(n, poly):
    # poly with keyfor =
    # ['e^{kt}', 'k^{-}', 'E[v]', 'sigma', 'lmbd', 'mu_v']
    kf = ['k^{-}', 'E[v]', 'sigma', 'lmbd', 'mu_v']
    #
    pol1 = Poly()
    pol1.set_keyfor(kf)
    pol2 = Poly()
    pol2.set_keyfor(kf)
    #
    for k, v in poly.items():
        if v == 0: continue  # works for Fraction(0, 1)
        #
        if k[0] == 0:
            pol1.add_keyval(k[1:], v)
        elif k[0] == -n:
            pol2.add_keyval(k[1:], v)
        else:
            raise ValueError("k[0] is neither 0 nor -n")
    #
    if len(pol1) != len(pol2):
        raise ValueError("len(pol1) != len(pol2)")
    #
    for k in pol1:
        if pol1[k] != -pol2[k]:
            raise ValueError("pol1[k] != -pol2[k]")
    # double check
    for k in pol2:
        if pol2[k] != -pol1[k]:
            raise ValueError("pol2[k] != -pol1[k]")
    #
    return pol1

def solve_polynomial(n, uncond_cm):
    #
    # get the conditional central moment as a polynomial of v0-E[v]
    #
    cond_cm = cond_cmoment_v(n)
    # kf = ['e^{kt}', 'k^{-}', 'v0-E[v]', 'E[v]', 'sigma', 'lmbd', 'mu_v']
    #
    # pick up the special monomial: e^{-nkt} (v0-E[v])^n
    #
    if find_same_order(cond_cm, n):
        del cond_cm[(-n, 0, n, 0, 0, 0, 0)]
    else:
        raise Exception("error with the conditional central moment")
    #
    # get the unconditional central moment in the left hand of the equation
    #
    uncond_left = last_expectation(cond_cm, uncond_cm)
    #
    # divide the left by 1 - e^{-nkt}
    #
    poly = divide(n, uncond_left) # ['k^{-}', 'E[v]', 'sigma', 'lmbd', 'mu_v']
    return poly

def cmoment_v(n):
    """central moment :math:`E[(v_t - E[v])^m]`

    unconditional central moment

    :param int n: order of the unconditional central moment
    :return: poly with ``keyfor`` = ('k^{-}', 'E[v]', 'sigma', 'lmbd', 'mu_v')
    :rtype: Poly
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    kf = ['k^{-}', 'E[v]', 'sigma', 'lmbd', 'mu_v']
    p1 = Poly({(0 ,0 ,0, 0, 0): Fraction(1, 1)})
    p1.set_keyfor(kf)
    p0 = Poly({(0 ,0 ,0, 0, 0): Fraction(0, 1)})
    p0.set_keyfor(kf)
    #
    uncond_cm = []
    uncond_cm.append(p1)  # E[(v_0 - E[v])^0]
    uncond_cm.append(p0)  # E[(v_0 - E[v])^1]
    #
    if n < 2:
        return uncond_cm[n]
    #
    for i in range(2, n):
        poly = solve_polynomial(i, uncond_cm)
        uncond_cm.append(poly)
    poly = solve_polynomial(n, uncond_cm)
    return poly


if __name__ == "__main__":
    from pprint import pprint

    print('\nExample usage of the module functions\n')
    kf = ['k^{-}', 'E[v]', 'sigma', 'lmbd', 'mu_v']
    n = 4
    print(f'cmoment_v({n}) = ')
    pprint(cmoment_v(n))
    print(f"{kf}")

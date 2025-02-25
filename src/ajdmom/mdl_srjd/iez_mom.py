"""
Moments of :math:`IEZ_{s,t}`
"""
import math
from fractions import Fraction

from ajdmom.poly import Poly
from ajdmom.cpp_mom import dmgf_cpp

def moment_EJ(n):
    # kf = ['e^{kt}-e^{ks}', 't-s', 'k^{-}', 'mu_v']
    kf = ['e^{kt}', 'e^{ks}', 't-s', 'k^{-}', 'mu_v']
    poly = Poly()
    poly.set_keyfor(kf)
    if n < 0:
        raise ValueError(f'moment_EJ({n}) is called!')
    if n == 0:
        key = (0, 0, 0, 0, 0)
        val = Fraction(1, 1)
        poly.add_keyval(key, val)
        return poly
    key = (n, 0, -1, 1, n)
    val = Fraction(math.factorial(n-1), 1)
    poly.add_keyval(key, val)
    #
    key = (0, n, -1, 1, n)
    val = Fraction(-math.factorial(n-1), 1)
    poly.add_keyval(key, val)
    return poly

def decode(poly):
    poln = Poly()
    poln.set_keyfor(['e^{kt}', 'e^{ks}', 't-s', 'k^{-}', 'lmbd', 'mu_v'])
    for k in poly:
        i = k[0]
        #
        pol1 = Poly({(0, 0, 0, 0, 0): Fraction(1, 1)})
        pol1.set_keyfor(['e^{kt}', 'e^{ks}', 't-s', 'k^{-}', 'mu_v'])
        for j in range(1, len(k)):
            n_j, m_j = k[j]
            pol1 = pol1 * (moment_EJ(n_j) ** m_j)
        #
        for key in pol1:
            knw = (key[0], key[1], key[2] + i, key[3], i, key[4])
            val = poly[k] * pol1[key]
            poln.add_keyval(knw, val)
    poln.remove_zero()
    #
    pol2 = Poly()
    pol2.set_keyfor(['e^{kt}', 'e^{ks}', 'k^{-}', 'lmbd', 'mu_v'])
    for key in poln:
        if key[2] != 0:
            raise ValueError("power of 't-s' is not 0 in moment_IEZ(n)!")
    for k, v in poln.items():
        key = k[0:2] + k[3:]
        pol2.add_keyval(key, v)
    return pol2

def moment_IEZ(n):
    """Moment of :math:`E[IEZ_{s,t}^n]`

    :param int n: order of the moment
    :return: poly with ``keyfor`` = ('e^{kt}', 'e^{ks}', 'k^{-}', 'lmbd', 'mu_v')
    :rtype: Poly
    """
    if n < 0:
        raise ValueError(f"moment_IEZ({n}) is called!")
    #
    M = []
    kf = ['[lmbd(t-s)]^{i} M^{(n_1)m_1}(s) ... M^{(n_l)m_l}(s)']
    # n = 0
    # kf = ['e^{kt}-e^{ks}', 't-s', 'k^{-}', 'lmbd', 'mu_v']
    poly = Poly({(0,): Fraction(1, 1)})
    poly.set_keyfor(kf)
    M.append(poly)
    #
    if n == 0:
        poln = Poly({(0, 0, 0, 0, 0): Fraction(1, 1)})
        poln.set_keyfor(['e^{kt}', 'e^{ks}', 'k^{-}', 'lmbd', 'mu_v'])
        return poln
    # n = 1
    poly = Poly({(1, (1, 1)): 1})
    poly.set_keyfor(kf)
    M.append(poly)
    #
    if n == 1:
        poln = Poly({
            (1, 0, 1, 1, 1): Fraction(1, 1),
            (0, 1, 1, 1, 1): Fraction(-1, 1)
        })
        poln.set_keyfor(['e^{kt}', 'e^{ks}', 'k^{-}', 'lmbd', 'mu_v'])
        return poln
    #
    for i in range(2, n+1):
        M.append(dmgf_cpp(M[-1]))
    return decode(M[-1])

def poly2num(poly, par):
    # ['e^{kt}', 'e^{ks}', 'k^{-}', 'lmbd', 'mu_v']
    k, lmbd, mu_v = par['k'], par['lmbd'], par['mu_v']
    s, t = par['lb'], par['ub']
    f = 0
    for K, v in poly.items():
        val = math.exp(K[0] * k * t) * math.exp(K[1] * k * s)
        val *= (k ** (-K[2])) * (lmbd ** K[3]) * (mu_v ** K[4])
        f += val * v
    return f

def m(n, par):
    moment = moment_IEZ(n)
    value = poly2num(moment, par)
    return value


if __name__ == '__main__':
    from pprint import pprint

    print('\nExample usage of the module functions')

    n = 3
    print(f"moment_EJ({n}) = ")
    pprint(moment_EJ(n))
    print("which is a poly with keyfor = ('e^{kt}', 'e^{ks}', 't-s', 'k^{-}', 'mu_v')\n")
    #
    # n = 1
    print(f"moment_IEZ({n}) = ")
    pprint(moment_IEZ(n))
    print("which is a poly with keyfor = ('e^{kt}', 'e^{ks}', 'k^{-}', 'lmbd', 'mu_v')")

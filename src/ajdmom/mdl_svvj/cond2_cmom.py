"""
Conditional Central Moments for the SVVJ model, given the initial state of
the variance and the realized jumps in the variance.

Conditional central moments are derived simultaneously because the one-by-one
procedure is not efficient.
"""
import math
from fractions import Fraction as Frac

from ajdmom.poly import Poly
from ajdmom.utils import simplify_rho, comb, fZ
from ajdmom.ito_cond2_mom import recursive_IEII


def simplify(poly, tp=0):
    """Simplify the final poly

    :param Poly poly: poly with ``keyfor`` =
      ('e^{kt}','t','k^{-}','v0-theta','theta','sigma',
      'l_{1:n}', 'o_{1:n}',
      'sigma/2k','rho-sigma/2k','sqrt(1-rho^2)')
    :param int tp: type of the simplification:
      (1) ``tp = 0``: only changes from 'e^{kt}' to 'e^{-kt}'.
      (2) ``tp = 1``: changes in addition from 'sigma/2k', 'rho-sigma/2k',
      'sqrt(1-rho^2)' to 'rho'.
    :return: poly with ``keyfor`` =
      ('e^{-kt}','t','k^{-}','v0-theta','theta','sigma',
      'l_{1:n}', 'o_{1:n}', 'rho') when ``tp = 1``;
      poly with ``keyfor`` changes only from 'e^{kt}' to 'e^{-kt}'
      if ``tp = 0``.
    :rtype: Poly
    """
    keyfor = ('e^{-kt}',) + poly.keyfor[1:]
    poln = Poly()
    #
    # tp = 0: only changes from e^{kt} to e^{-kt}
    #
    if tp == 0:
        poln.set_keyfor(keyfor)
        for k, v in poly.items():
            poln.add_keyval((-k[0],) + k[1:], v)
        return poln
    #
    # tp = 1: simplify more
    #   from 'sigma/2k' (8), 'rho-sigma/2k' (9) to 'rho'
    kf = keyfor[0:8] + ('rho',) + keyfor[10:]
    poln.set_keyfor(kf)
    for k in poly:
        for i in range(0, k[9] + 1):
            p = k[8] + i  # power of 'sigma/2k'
            key = (-k[0], k[1], k[2] + p, k[3], k[4], k[5] + p,
                   k[6], k[7], k[9] - i, k[10])
            val = math.comb(k[9], i) * ((-1) ** i) * Frac(1, 2 ** p) * poly[k]
            poln.add_keyval(key, val)
    #   from 'rho' (9), 'sqrt(1-rho^2)' (10) to 'rho'
    poln = simplify_rho(poln, 10)
    return poln


def cmoments_y_to(l, show=False):
    """Derive the central moments with orders from 0 to l simultaneously

    :param int l: highest order of the conditional central moments to derive, l >= 1
    :param bool show: show the in-process message or not, defaults to False
    :return: a list of the conditional central moments with ``keyfor`` =
      ('e^{kt}','t','k^{-}','v0-theta','theta','sigma',
      'l_{1:n}', 'o_{1:n}',
      'sigma/2k','rho-sigma/2k','sqrt(1-rho^2)')
    :rtype: list
    """
    # IEII: a dict of moments of E[IE_t^{n1}I_t^{n2}I_t^{*n3}]
    IEII = {}
    kf = ['e^{kt}', 't', 'k^{-}', 'v0-theta', 'theta', 'sigma', 'l_{1:n}', 'o_{1:n}']
    #
    P0 = Poly({(0, 0, 0, 0, 0, 0, (), ()): Frac(0, 1)})
    P0.set_keyfor(kf)
    P1 = Poly({(0, 0, 0, 0, 0, 0, (), ()): Frac(1, 1)})
    P1.set_keyfor(kf)
    #
    # n1 + n2 + n3 = 0
    IEII[(0, 0, 0)] = P1  # support for special case, constant 1
    # n1 + n2 + n3 = 1
    IEII[(1, 0, 0)] = P0  # support for special case, constant 0
    IEII[(0, 1, 0)] = P0
    IEII[(0, 0, 1)] = P0
    # n1 + n2 +n3 = 2
    P200 = Poly({
        (1, 0, 1, 1, 0, 0, (), ()): Frac(1, 1),
        (0, 0, 1, 1, 0, 0, (), ()): -Frac(1, 1),
        (2, 0, 1, 0, 1, 0, (), ()): Frac(1, 2),
        (0, 0, 1, 0, 1, 0, (), ()): -Frac(1, 2),
        (1, 0, 1, 0, 0, 0, (0,), (0,)): Frac(1, 1),
        (0, 0, 1, 0, 0, 0, (1,), (0,)): -Frac(1, 1)
    })
    P200.set_keyfor(kf)
    P110 = Poly({
        (0, 1, 0, 1, 0, 0, (), ()): Frac(1, 1),
        (1, 0, 1, 0, 1, 0, (), ()): Frac(1, 1),
        (0, 0, 1, 0, 1, 0, (), ()): -Frac(1, 1),
        (0, 1, 0, 0, 0, 0, (0,), (0,)): Frac(1, 1),
        (0, 0, 0, 0, 0, 0, (0,), (1,)): -Frac(1, 1)
    })
    P110.set_keyfor(kf)
    P020 = Poly({
        (-1, 0, 1, 1, 0, 0, (), ()): -Frac(1, 1),
        (0, 0, 1, 1, 0, 0, (), ()): Frac(1, 1),
        (0, 1, 0, 0, 1, 0, (), ()): Frac(1, 1),
        (-1, 0, 1, 0, 0, 0, (0,), (0,)): -Frac(1, 1),
        (0, 0, 1, 0, 0, 0, (-1,), (0,)): Frac(1, 1)
    })
    P020.set_keyfor(kf)
    #
    IEII[(2, 0, 0)] = P200
    IEII[(1, 1, 0)] = P110
    IEII[(1, 0, 1)] = P0
    IEII[(0, 2, 0)] = P020
    IEII[(0, 1, 1)] = P0
    IEII[(0, 0, 2)] = P020
    #
    kf += ['sigma/2k', 'rho-sigma/2k', 'sqrt(1-rho^2)']
    #
    cmoms = []
    #
    # special case: 0-th central moment
    #
    P1 = Poly({(0, 0, 0, 0, 0, 0, (), (), 0, 0, 0): Frac(1, 1)})
    P1.set_keyfor(kf)
    cmoms.append(P1)  # equiv to constant 1
    #
    # special case: 1-th central moment
    #
    P0 = Poly({(0, 0, 0, 0, 0, 0, (), (), 0, 0, 0): Frac(0, 1)})
    P0.set_keyfor(kf)
    cmoms.append(P0)  # equiv to constant 0
    #
    # typical cases
    #
    for n in range(2, l + 1):
        # container for the n-th conditional central moment of y
        poly = Poly()
        poly.set_keyfor(kf)
        # iter over all combinations of i + j + (n-i-j) = n
        for i in range(n, -1, -1):
            for j in range(n - i, -1, -1):
                # derive for this exact combination
                c = comb(n, [i, j, n - i - j])  # constant
                if n < 3:
                    poln = IEII[(i, j, n - i - j)]  # already in IEII
                else:
                    poln = recursive_IEII(i, j, n - i - j, IEII)
                # add this exact combination into poly
                # c * b * poln
                for k in poln:
                    key = list(k)
                    key[0] -= i
                    key = tuple(key)  # × e^{-ikt}
                    key = key + (i, j, n - i - j)  # enlarge with last 3
                    poly.add_keyval(key, c * poln[k])
                # reserve poln for further use or delete it otherwise
                if n >= 3 and n < l:
                    IEII[(i, j, n - i - j)] = poln
                else:
                    del poln
        # delete polys when no more needed
        if n == l:
            del IEII
        else:  # only reserve the most recent polys
            index = [key for key in IEII if key[0] + key[1] + key[2] == n - 2]
            for key in index: del IEII[key]
        poly.remove_zero()
        cmoms.append(poly)
        if show: print(f"complete the derivation for the {n}-th central moment.")
    return cmoms


def cmoment_y(l):
    """Derive the conditional central moment

    :param int l: order of the conditional central moment to derive
    :return: Poly object with ``keyfor`` =
      ('e^{kt}','t','k^{-}','v0-theta','theta','sigma',
      'l_{1:n}', 'o_{1:n}',
      'sigma/2k','rho-sigma/2k','sqrt(1-rho^2)')
    :rtype: Poly
    """
    cmoments = cmoments_y_to(l)
    return cmoments[-1]


def classify(poly):
    """Classify the key-value pairs inside poly according to levels of summation

    :param Poly poly: poly with the first part of its ``keyfor`` =
     ('e^{-kt}','t','k^{-}','v0-theta','theta','sigma',
     'l_{1:n}', 'o_{1:n}')
     and the second part may be ('sigma/2k','rho-sigma/2k','sqrt(1-rho^2)')
     or ('rho')
    :return: list of subpolys
    :rtype: list
    """
    # record levels of summations
    N_sum = []
    for k in poly:
        if len(k[6]) not in N_sum: N_sum.append(len(k[6]))
    N_sum = sorted(N_sum)  # 0,1,2,...
    #
    # excludes l_{1:n}, o_{1:n}
    kf0 = poly.keyfor[:6] + poly.keyfor[8:]
    subpolys = [Poly() for n in N_sum]
    #
    subpolys[0].set_keyfor(kf0 if N_sum[0] == 0 else poly.keyfor)
    # exclude the sub-keyfor if the sub-key corresponding to 'l_{1:n}' is empty
    for i in range(1, len(N_sum)): subpolys[i].set_keyfor(poly.keyfor)
    #
    # classify
    for k, v in poly.items():
        subpoly = subpolys[len(k[6])]
        subpoly[k[0:6] + k[8:] if len(k[6]) == 0 else k] = v
        # exclude the empty subkey (),()
    return subpolys


def write_to_subpolys(poly, nth):
    """Write poly into different subpolys according to levels of summation

    :param Poly poly: poly with ``keyfor`` =
      ('e^{kt}','t','k^{-}','v0-theta','theta','sigma',
      'l_{1:n}', 'o_{1:n}',
      'sigma/2k','rho-sigma/2k','sqrt(1-rho^2)')
    :param int nth: order of the moment
    :return: nothing
    :rtype: None
    """
    # only changes from e^{kt} to e^{-kt}
    poly = simplify(poly, tp=0)  # do not over simplify
    subpolys = classify(poly)
    for n in range(len(subpolys)):
        subpoly = subpolys[n]
        fname = f"svvj-cond-cmoment-{nth}-formula-{n}.txt"
        subpoly.write_to_txt(fname)


##########
# scalar
##########
def poly2num(poly, par):
    """Decode poly back to scalar

    :param Poly poly: poly to be decoded with attribute ``keyfor`` =
      ('e^{kt}','t','k^{-}','v0-theta','theta','sigma',
      'l_{1:n}', 'o_{1:n}',
      'sigma/2k','rho-sigma/2k','sqrt(1-rho^2)')
    :param dict par: parameters in dict, ``jumptime`` (tuple) and ``jumpsize`` (tuple)
      must also be included
    :return: scalar of the poly
    :rtype: float
    """
    v0 = par['v0']
    k = par['k']
    theta = par['theta']
    sigma = par['sigma']
    rho = par['rho']
    h = par['h']  # t = h
    J = par['jumpsize']  # vector of jump sizes
    s = par['jumptime']  # vector of jump time points
    #
    f = 0
    for K in poly:
        val = math.exp(K[0] * k * h) * (h ** K[1]) * (k ** (-K[2]))
        val *= ((v0 - theta) ** K[3]) * (theta ** K[4]) * (sigma ** K[5])
        #
        l, o = K[6], K[7]
        val *= fZ(l, o, k, s, J)
        #
        val *= (sigma / (2 * k)) ** K[8]
        val *= (rho - sigma / (2 * k)) ** K[9]
        val *= (1 - rho ** 2) ** (K[10] / 2)
        #
        f += val * poly[K]
    return f


def cm(l, par):
    """conditional central moment in scalar

    :param int l: order of the conditional central moment
    :param dict par: parameters in dict, ``jumptime`` (tuple) and ``jumpsize`` (tuple)
      must also be included
    :return: scalar of the central moment
    :rtype: float
    """
    cmoments = cmoments_y_to(l)
    cmoment = cmoments[-1]  # need only the last one
    value = poly2num(cmoment, par)
    return value


if __name__ == "__main__":
    import sys
    from pprint import pprint

    print('\nExample usage of the module function\n')
    args = sys.argv[1:]
    l = 3 if len(args) == 0 else int(args[0])
    cmoms = cmoments_y_to(l, show=True)  # 0-th to l-th central moments
    kf = cmoms[0].keyfor
    print(f"\ncmoments_y_to(l) returns a list of polys with keyfor = \n{kf}")
    # for i in range(1, l+1):
    #   cmom = cmoms[i]
    #   write_to_subpolys(cmom, i)
    print(f"\ncmoment_y({l}) = ")
    pprint(cmoms[l])
    print(f"which is a poly with keyfor = \n{kf}")

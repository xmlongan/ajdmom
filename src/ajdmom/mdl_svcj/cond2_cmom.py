"""
Conditional Central Moments for the SVCJ model, given :math:`v_0`
and the realized jumps in the variance.
"""
import math
from fractions import Fraction as Frac

from ajdmom.poly import Poly
from ajdmom.utils import cmnorm, comb, dbfactorial, fZ
from ajdmom.mdl_svvj.cond2_cmom import cmoments_y_to as cmoments_y_to_svvj


def cmoment_IZs(m, J):
    r"""conditional central moment :math:`\mathbb{E}[(\overline{IZ_t^{s}})^{m}
    |z^v(u), 0\le u \le t]`

    :param int m: power of :math:`\overline{IZ^s}_t`
    :param tuple J: jump sizes in the variance
    :return: poly with attribute ``keyfor`` = ('sigma_s'), noting that key is
      still a tuple (of one element)
    :rtype: Poly
    """
    kf = ['sigma_s']
    poly = Poly()
    poly.set_keyfor(kf)
    #
    # special cases
    #
    if m == 0:  # return 1
        poly.add_keyval((0,), Frac(1, 1))
        return (poly)
    if len(J) == 0:  # return 0
        poly.add_keyval((0,), Frac(0, 1))
        return (poly)
    #
    # typical cases
    #
    if len(J) == 1:  # single normal variable
        poly = cmnorm(m)
        poly.set_keyfor(kf)
        return (poly)
    if len(J) == 2:
        for i in range(0, m + 1):
            c = math.comb(m, i)
            pol1 = cmnorm(i)
            pol2 = cmnorm(m - i)  # cmnorm(0) supported
            poln = pol1 * pol2
            poln.set_keyfor(kf)
            poly.merge(c * poln)
        return (poly)
    #
    # len(J) > 2
    #
    stack = []
    nextstep = "forward"
    while len(stack) < len(J):
        if nextstep == "forward":
            if len(stack) < len(J) - 2:
                stack.append(0)
                nextstep = "forward"
            else:
                left = m - sum(stack)
                for i in range(0, left + 1):
                    stack.extend([i, left - i])
                    index = stack.copy()
                    #
                    c = comb(m, index)
                    poln = Poly({(0,): Frac(1, 1)})
                    poln.set_keyfor(['sigma'])
                    for j in range(len(J)):
                        poln *= cmnorm(index[j])
                    poln.set_keyfor(['sigma_s'])
                    poly += c * poln
                    #
                    stack.pop()
                    stack.pop()
                # checking whether all indexes traversed
                if stack[0] == m: break
                #
                nextstep = "backward"
        else:
            left = m - sum(stack)
            if left == 0:
                stack.pop()
                nextstep = "backward"
            else:
                stack[-1] += 1
                nextstep = "forward"
    return poly


def cmoments_y_to(l, J):
    """conditional central moments of :math:`y_t` of orders :math:`0:l`

    :param int l: highest order of the conditional central moments to derive, >= 1
    :param tuple J: jump sizes in the variance
    :return: a list of polys with attribute ``keyfor`` =
      ('e^{kt}','t','k^{-}','v0-theta','theta','sigma',
      'l_{1:n}', 'o_{1:n}',
      'sigma/2k','rho-sigma/2k','sqrt(1-rho^2)',
      'sigma_s')
    :rtype: list
    """
    cmoms = []
    cmoms_svvj = cmoments_y_to_svvj(l)  # 0-th to l-th central moments
    #
    kf = cmoms_svvj[0].keyfor + ('sigma_s',)
    #
    # special case: 0-th central moment
    #
    P1 = Poly({(0, 0, 0, 0, 0, 0, (), (), 0, 0, 0, 0): Frac(1, 1)})
    P1.set_keyfor(kf)
    cmoms.append(P1)  # equiv to constant 1
    #
    # special case: 1-th central moment
    #
    P0 = Poly({(0, 0, 0, 0, 0, 0, (), (), 0, 0, 0, 0): Frac(0, 1)})
    P0.set_keyfor(kf)
    cmoms.append(P0)  # equiv to constant 0
    #
    # typical cases
    #
    for n in range(2, l + 1):
        poly = Poly()
        poly.set_keyfor(kf)
        for n1 in range(n, -1, -1):
            c = math.comb(n, n1)
            pol1 = cmoms_svvj[n1]  # 0-th central moment supported
            pol2 = cmoment_IZs(n - n1, J)  # kf = ['sigma_s']
            for k1 in pol1:
                for k2 in pol2:
                    key = k1 + k2
                    poly.add_keyval(key, c * pol1[k1] * pol2[k2])
        poly.remove_zero()
        cmoms.append(poly)
    return cmoms


def cmoment_y(l, J):
    cmoms = cmoments_y_to(l, J)
    return cmoms[-1]


##########
# scalar
##########
def poly2num(poly, par):
    """Decode poly back to scalar

    :param Poly poly: poly to be decoded with attibute ``keyfor`` =
      ('e^{kt}','t','k^{-}','v0-theta','theta','sigma',
      'l_{1:n}', 'o_{1:n}',
      'sigma/2k','rho-sigma/2k','sqrt(1-rho^2)',
      'sigma_s')
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
    J = par['jumpsize']  # list of jump sizes
    s = par['jumptime']  # list of jump time points
    sigma_s = par['sigma_s']
    #
    f = 0
    for K in poly:
        val = math.exp(K[0] * k * h) * (h ** K[1]) * (k ** (-K[2]))
        val *= ((v0 - theta) ** K[3]) * (theta ** K[4]) * (sigma ** K[5])
        #
        l, o= K[6], K[7]
        val *= fZ(l, o, k, s, J)
        #
        val *= (sigma / (2 * k)) ** K[8]
        val *= (rho - sigma / (2 * k)) ** K[9]
        val *= (1 - rho ** 2) ** (K[10] / 2)
        #
        val *= sigma_s ** K[11]
        #
        f += val * poly[K]
    return f


def cm(l, par):
    """conditional central moment in scalar

    :param int l: order of the conditional central moment
    :param dict par: parameters in dict, ``jumptime`` (tuple) and ``jumpsize`` (tuple)
      must also be included
    :return: scalar of the conditional central moment
    :rtype: float
    """
    J = par['jumpsize']  # list of jump sizes
    cmoments = cmoments_y_to(l, J)
    cmoment = cmoments[-1]
    value = poly2num(cmoment, par)
    return value


if __name__ == "__main__":
    from pprint import pprint
    import sys
    import random

    print('Example usage of the module function\n')
    numJ = 5  # special cases: 0, 1, 2
    jumpsize = [random.expovariate(lambd=1.0) for i in range(numJ)]
    args = sys.argv[1:]
    l = 2 if len(args) == 0 else int(args[0])
    cmoms = cmoments_y_to(l, jumpsize)  # 0-th to l-th central moments
    kf = cmoms[0].keyfor
    print(f"cmoment_y({l}, J) = \n")
    pprint(cmoms[l])
    print(f"\nwhich is a poly with keyfor = \n{kf}")

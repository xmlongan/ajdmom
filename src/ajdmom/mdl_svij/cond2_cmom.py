"""
Conditional Central Moments for the SVIJ model, given the initial variance
and jump time points and jump sizes of the CPP in the variance
"""
import math
from fractions import Fraction as Frac

from ajdmom.poly import Poly
from ajdmom.cpp_mom import cmcpp
from ajdmom.utils import fZ
from ajdmom.mdl_svvj.cond2_cmom import cmoments_y_to as cmoments_y_to_svvj


def cmoments_y_to(l):
    """conditional central moments of :math:`y_t` of orders :math:`0:l`

    :param int l: highest order of the central moments to derive, >= 1.
    :return: a list of polys with attribute ``keyfor`` =
      ('e^{kt}','t','k^{-}','v0-theta','theta','sigma',
      'l_{1:n}', 'o_{1:n}', 'p_{2:n}', 'q_{2:n}',
      'sigma/2k','rho-sigma/2k','sqrt(1-rho^2)',
      'lmbd_s', 'mu_s', 'sigma_s')
    :rtype: list
    """
    cmoms = []
    cmoms_svvj = cmoments_y_to_svvj(l)  # 0-th to l-th central moments
    #
    kf = cmoms_svvj[0].keyfor + ('lmbd_s', 'mu_s', 'sigma_s')
    #
    # special case: 0-th central moment
    #
    P1 = Poly({(0, 0, 0, 0, 0, 0, (), (), (), (), 0, 0, 0, 0, 0, 0): Frac(1, 1)})
    P1.set_keyfor(kf)
    cmoms.append(P1)  # equiv to constant 1
    #
    # special case: 1-th central moment
    #
    P0 = Poly({(0, 0, 0, 0, 0, 0, (), (), (), (), 0, 0, 0, 0, 0, 0): Frac(0, 1)})
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
            pol2 = cmcpp(n - n1)  # kf = ['lmbd_s*h','mu_s','sigma_s']
            for k1 in pol1:
                for k2 in pol2:
                    key = (k1[0], k1[1] + k2[0]) + k1[2:] + k2  # t = h
                    poly.add_keyval(key, c * pol1[k1] * pol2[k2])
        poly.remove_zero()
        cmoms.append(poly)
    return cmoms


def cmoment_y(l):
    cmoms = cmoments_y_to(l)
    return cmoms[-1]


##########
# scalar
##########
def poly2num(poly, par):
    """Decode poly back to scalar

    :param Poly poly: poly to be decoded with attribute ``keyfor`` =
      ('e^{kt}','t','k^{-}','v0-theta','theta','sigma',
      'l_{1:n}', 'o_{1:n}', 'p_{2:n}', 'q_{2:n}',
      'sigma/2k','rho-sigma/2k','sqrt(1-rho^2)',
      'lmbd_s', 'mu_s', 'sigma_s')
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
    lmbd_s = par['lmbd_s']
    mu_s = par['mu_s']
    sigma_s = par['sigma_s']
    #
    f = 0
    for K in poly:
        val = math.exp(K[0] * k * h) * (h ** K[1]) * (k ** (-K[2]))
        val *= ((v0 - theta) ** K[3]) * (theta ** K[4]) * (sigma ** K[5])
        #
        l, o, p, q = K[6], K[7], K[8], K[9]
        val *= fZ(l, o, p, q, k, s, J)
        #
        val *= (sigma / (2 * k)) ** K[10]
        val *= (rho - sigma / (2 * k)) ** K[11]
        val *= (1 - rho ** 2) ** (K[12] / 2)
        #
        val *= (lmbd_s ** K[13]) * (mu_s ** K[14]) * (sigma_s ** K[15])
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
    cmoment = cmoment_y(l)
    value = poly2num(cmoment, par)
    return value


if __name__ == "__main__":
    from pprint import pprint
    import sys

    print('Example usage of the module function\n')
    args = sys.argv[1:]
    l = 2 if len(args) == 0 else int(args[0])
    cmoms = cmoments_y_to(l)  # 0-th to l-th central moments
    kf = cmoms[0].keyfor
    print(f"cmoment_y({l}) = \n")
    pprint(cmoms[l])
    print(f"\nwhich is a poly with keyfor = \n{kf}")

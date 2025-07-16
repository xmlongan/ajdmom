"""
Conditional Moments for the SVIJ model, given :math:`v_0`
and the realized jumps in the variance.
"""
import math
from fractions import Fraction as Frac

from ajdmom.poly import Poly
from ajdmom.cpp_mom import mcpp
from ajdmom.utils import fZ
from ajdmom.mdl_svvj.cond2_mom import moments_y_to as moments_y_to_svvj


def moments_y_to(l):
    r"""conditional moments of :math:`y_t` of orders :math:`0:l`

    :param int l: highest order of the conditional moments to derive, >= 1
    :return: a list of polys with attribute ``keyfor`` =
      ('e^{kt}','t','k^{-}','beta_t','mu-theta/2','v0-theta','theta','sigma',
      'l_{1:n}', 'o_{1:n}', 'p',
      'sigma/2k','rho-sigma/2k','sqrt(1-rho^2)',
      'lmbd_s', 'mu_s', 'sigma_s'), now 'p' encoding power of :math:`I\!Z_t`.
    :rtype: Poly
    """
    moms = []
    moms_svvj = moments_y_to_svvj(l)  # 0-th to l-th moments
    #
    kf = moms_svvj[0].keyfor + ('lmbd_s', 'mu_s', 'sigma_s')
    #
    # special case: 0-th moment
    #
    P1 = Poly({(0, 0, 0, 0, 0, 0, 0, 0, (), (), 0, 0, 0, 0, 0, 0, 0): Frac(1, 1)})
    P1.set_keyfor(kf)
    moms.append(P1)  # equiv to constant 1
    #
    # typical cases
    #
    for n in range(1, l + 1):
        poly = Poly()
        poly.set_keyfor(kf)
        for n1 in range(n, -1, -1):
            c = math.comb(n, n1)
            pol1 = moms_svvj[n1]  # 0-th moment supported
            pol2 = mcpp(n - n1)  # kf = ['lmbd_s*h','mu_s','sigma_s']
            for k1 in pol1:
                for k2 in pol2:
                    key = (k1[0], k1[1] + k2[0]) + k1[2:] + k2  # t = h
                    poly.add_keyval(key, c * pol1[k1] * pol2[k2])
        poly.remove_zero()
        moms.append(poly)
    return moms


def moment_y(l):
    moms = moments_y_to(l)
    return moms[-1]


##########
# scalar
##########

def poly2num(poly, par):
    """Decode poly back to scalar

    :param Poly poly: poly to be decoded with attribute ``keyfor`` =
      ('e^{kt}','t','k^{-}','beta_t','mu-theta/2','v0-theta','theta','sigma',
      'l_{1:n}', 'o_{1:n}', 'p',
      'sigma/2k','rho-sigma/2k','sqrt(1-rho^2)',
      'lmbd_s', 'mu_s', 'sigma_s')
    :param dict par: parameters in dict, ``jumptime`` (tuple) and ``jumpsize`` (tuple)
      must also be included

    :return: scalar of the poly
    :rtype: float
    """
    v0 = par['v0']
    mu = par['mu']
    k = par['k']
    theta = par['theta']
    sigma = par['sigma']
    rho = par['rho']
    h = par['h']  # t= h
    J = par['jumpsize']  # vector of jump sizes
    s = par['jumptime']  # vector of jump time points
    lmbd_s = par['lmbd_s']
    mu_s = par['mu_s']
    sigma_s = par['sigma_s']
    #
    beta_t = (1 - math.exp(-k * h)) / (2 * k)
    #
    f = 0
    for K in poly:
        val = math.exp(K[0] * k * h) * (h ** K[1]) * (k ** (-K[2]))
        val *= (beta_t ** K[3]) * ((mu - theta / 2) ** K[4]) * ((v0 - theta) ** K[5])
        val *= (theta ** K[6]) * (sigma ** K[7])
        #
        l, o, p = K[8], K[9], K[10]
        #
        val *= fZ(l, o, k, s, J)
        #
        IZ = sum(J)
        val *= IZ ** p
        #
        val *= ((sigma / (2 * k)) ** K[11]) * ((rho - sigma / (2 * k)) ** K[12])
        val *= ((1 - rho ** 2) ** (K[13] / 2))
        #
        val *= (lmbd_s ** K[14]) * (mu_s ** K[15]) * (sigma_s ** K[16])
        #
        f += val * poly[K]
    return f


def m(l, par):
    """conditional moment in scalar

    :param int l: order of the conditional moment
    :param dict par: parameters in dict, ``jumptime`` (tuple) and ``jumpsize`` (tuple)
      must also be included
    :return: scalar of the moment
    :rtype: float
    """
    moment = moment_y(l)
    value = poly2num(moment, par)
    return value


if __name__ == "__main__":
    # Example usage of the module, see 'tests/test_mdl_svcj.py' for more test
    from pprint import pprint
    import sys

    print('Example usage of the module function\n')
    args = sys.argv[1:]
    l = 1 if len(args) == 0 else int(args[0])
    moms = moments_y_to(l)  # 0-th to l-th moments
    kf = moms[0].keyfor
    print(f"moment_y({l}) = \n")
    pprint(moms[l])
    print(f"\nwhich is a poly with keyfor = \n{kf}")

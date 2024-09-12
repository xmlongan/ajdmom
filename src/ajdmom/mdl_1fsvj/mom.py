"""
Moments for One-Factor SV with jumps
"""
import math

from ajdmom.poly import Poly
from ajdmom.mdl_1fsv.mom import moment_y as m_y
# ('e^{-kh}','h','k^{-}','mu','theta','sigma_v','rho','sqrt(1-rho^2)')
from ajdmom.cpp_mom import mcpp


# ('lambda*h','mu','sigma^2')

def moment_y(l):
    """Moment of :math:`y_n` of order :math:`l`

    :param int l: order of the moment.
    :return: poly with attribute ``keyfor`` =
       ('e^{-kh}','h','k^{-}','mu','theta','sigma_v','rho','sqrt(1-rho^2)',
       'lambda','mu_j','sigma_j').
    :rtype: Poly
    """
    poly = Poly()
    kf = ['e^{-kh}', 'h', 'k^{-}', 'mu', 'theta', 'sigma_v', 'rho', 'sqrt(1-rho^2)',
          'lambda', 'mu_j', 'sigma_j']
    poly.set_keyfor(kf)
    #
    for i in range(l + 1):
        coef = math.comb(l, i)
        pol1 = m_y(i)
        # ('e^{-kh}','h','k^{-}','mu','theta','sigma_v','rho','sqrt(1-rho^2)')
        pol2 = mcpp(l - i)
        # ('lambda*h','mu','sigma')
        keyIndexes = [(0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1), (-1, 0, -1, -1, -1, -1, -1, -1, 0, 1, 2)]
        poln = pol1.mul_poly(pol2, keyIndexes, kf)
        poly.merge(coef * poln)
    return poly


##########
# scalar and (partial) derivative
##########

def dpoly(poly, wrt):
    """Partial derivative of central moment w.r.t. parameter wrt

    :param Poly poly: poly with attribute ``keyfor`` =
       ('e^{-kh}','h','k^{-}','mu','theta','sigma_v','rho','sqrt(1-rho^2)',
       'lambda','mu_j','sigma_j')
    :param str wrt: with respect to.
    :return: poly with attribute ``keyfor`` =
       ('e^{-kh}','h','k^{-}','mu','theta','sigma_v','rho','sqrt(1-rho^2)',
       'lambda','mu_j','sigma_j').
    :rtype: Poly
    """
    pold = Poly()
    kf = ('e^{-kh}', 'h', 'k^{-}', 'mu', 'theta', 'sigma_v', 'rho', 'sqrt(1-rho^2)',
          'lambda', 'mu_j', 'sigma_j')
    pold.set_keyfor(kf)
    #
    # partial derivative w.r.t. k
    if wrt == 'k':
        for k in poly:
            if k[0] != 0:
                knw = list(k)
                knw[1] += 1
                val = (-k[0]) * poly[k]
                pold.add_keyval(tuple(knw), val)
            if k[2] != 0:
                knw = list(k)
                knw[2] += 1
                val = (-k[2]) * poly[k]
                pold.add_keyval(tuple(knw), val)
    # partial derivative w.r.t. rho
    elif wrt == 'rho':
        for k in poly:
            if k[6] != 0:
                knw = list(k)
                knw[6] -= 1
                val = k[6] * poly[k]
                pold.add_keyval(tuple(knw), val)
            if k[7] != 0:
                knw = list(k)
                knw[6] += 1
                knw[7] -= 2
                val = (-k[7]) * poly[k]
                pold.add_keyval(tuple(knw), val)
    elif wrt in ['mu', 'theta', 'sigma_v', 'lambda', 'mu_j', 'sigma_j']:
        if wrt == 'mu': i = 3
        if wrt == 'theta': i = 4
        if wrt == 'sigma_v': i = 5
        if wrt == 'lambda': i = 8
        if wrt == 'mu_j': i = 9
        if wrt == 'sigma_j': i = 10
        for k in poly:
            if k[i] != 0:
                knw = list(k)
                knw[i] -= 1
                val = k[i] * poly[k]
                pold.add_keyval(tuple(knw), val)
    else:
        candidates = "'k','mu','theta','sigma_v','rho','lambda','mu_j','sigma_j'"
        raise ValueError(f"wrt must be one of {candidates}.")
    return pold


def poly2num(poly, par):
    """Decode poly back to scalar

    :param Poly poly: poly to be decoded with attribute ``keyfor`` =
       ('e^{-kh}','h','k^{-}','mu','theta','sigma_v','rho','sqrt(1-rho^2)',
       'lambda','mu_j','sigma_j').
    :param dict par: parameters in dict.
    :return: scalar of the poly.
    :rtype: float
    """
    k = par['k']
    h = par['h']
    mu = par['mu']
    theta = par['theta']
    sigma_v = par['sigma_v']
    rho = par['rho']
    lmbd = par['lambda']
    mu_j = par['mu_j']
    sigma_j = par['sigma_j']
    #
    value = 0
    for K in poly:
        val = poly[K] * math.exp(-K[0] * k * h) * (h ** K[1]) * (k ** (-K[2]))
        val *= (mu ** K[3])
        val *= (theta ** K[4]) * (sigma_v ** K[5]) * (rho ** K[6])
        val *= (1 - rho ** 2) ** (K[7] / 2)
        val *= (lmbd ** K[8]) * (mu_j ** K[9]) * (sigma_j ** K[10])
        value += val
    return value


def m(l, par):
    """Moment in scalar

    :param int l: order of the moment.
    :param dict par: parameters in dict.
    :return: scalar of the moment.
    :rtype: float
    """
    moment = moment_y(l)
    value = poly2num(moment, par)
    return value


def dm(l, par, wrt):
    """Partial derivative of moment w.r.t. parameter wrt

    :param int l: order of the moment.
    :param dict par: parameters in dict.
    :param str wrt: with respect to.
    :return: scalar of the partial derivative.
    :rtype: float
    """
    moment = moment_y(l)
    pold = dpoly(moment, wrt)
    value = poly2num(pold, par)
    return value


if __name__ == "__main__":
    # Example usage of the module, see 'tests/test_mdl_1fsvj.py' for more test
    from pprint import pprint
    import sys

    print('\nExample usage of the module function\n')
    kf = ('e^{-kh}', 'h', 'k^{-}', 'mu', 'theta', 'sigma_v', 'rho', 'sqrt(1-rho^2)',
          'lambda', 'mu_j', 'sigma_j^2')
    print(f"moment_y(l) returns a poly with keyfor = \n{kf}")
    args = sys.argv[1:]
    n = 1 if len(args) == 0 else int(args[0])
    print(f"moment_y({n}) = ")
    pprint(moment_y(n))

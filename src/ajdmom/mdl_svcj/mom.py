"""
Conditional Moments for SVCJ model
"""
import math
from fractions import Fraction as Frac

from ajdmom.poly import Poly
from ajdmom.cpp_mom import mnorm
from ajdmom.utils import comb, fZ
from ajdmom.mdl_svvj.mom import moments_y_to as moments_y_to_svvj


def mnorm_cond(n):
    """conditional normal distribution moment :math:`J_i^s|J_i^v \sim
    \mathcal{N} (\mu_s+rho_J J_i^v, \sigma_s^2)`

    :param int n: order of the moment, n >= 0
    :return: poly with ``keyfor`` = ('mu_s', 'rho_J', 'sigma_s', 'J_i')
    :rtype: Poly
    """
    kf = ['mu_s', 'rho_J', 'sigma_s', 'J_i']
    # special case
    if n == 0:
        P1 = Poly({(0, 0, 0, 0): Frac(1, 1)})
        P1.set_keyfor(kf)
        return (P1)
    # n >=1
    poly = Poly()
    poly.set_keyfor(kf)
    #
    poln = mnorm(n)  # kf = ['mu','sigma^2']
    for k in poln:
        for i in range(k[0] + 1):
            c = math.comb(k[0], i)
            key = (i, k[0] - i, 2 * k[1], k[0] - i)
            val = c * poln[k]
            poly.add_keyval(key, val)
    return poly


def comb_poly(index):
    """multiply together those conditional normal distribution moments

    :param list index: a combination of the Compound Poisson Process power :math:`m`,
       :math:`(m_1,\cdots,m_{N(t)})`
    :return: poly with attribute ``keyfor`` =
      ('mu_s', 'rho_J', 'sigma_s', 'J_{1:n}')
    :rtype: Poly
    """
    kf = ['mu_s', 'rho_J', 'sigma_s', 'J_{1:n}']
    poly = Poly()
    poly.set_keyfor(kf)
    #
    poly.add_keyval((0, 0, 0, ()), Frac(1, 1))
    for i in index:
        # mnorm_cond(0) supported
        pol1 = mnorm_cond(i)  # ['mu_s', 'rho_J', 'sigma_s', 'J_i']
        poln = Poly()
        poln.set_keyfor(kf)
        for k1 in pol1:
            for k2 in poly:
                key = (k1[0] + k2[0], k1[1] + k2[1], k1[2] + k2[2], k2[3] + (k1[3],))
                val = pol1[k1] * poly[k2]
                poln.add_keyval(key, val)
        poly = poln
    return poly


def moment_IZs(m, J):
    """moment :math:`\mathbb{E}[(IZ_t^{s})^{m}|z^v(u), 0\le u \le t]`

    :param int m: power of :math:`IZ^s_t`
    :param tuple J: jump sizes in the variance
    :return: poly with attribute ``keyfor`` =
      ('mu_s', 'rho_J', 'sigma_s', 'J_{1:n}')
    :rtype: Poly
    """
    kf = ['mu_s', 'rho_J', 'sigma_s', 'J_{1:n}']
    poly = Poly()
    poly.set_keyfor(kf)
    #
    numJ = len(J)
    #
    # special cases
    #
    if m == 0:  # return(1)
        poly.add_keyval((0, 0, 0, tuple(0 for i in range(numJ))), Frac(1, 1))
        return (poly)
    if numJ == 0:  # return(0)
        poly.add_keyval((0, 0, 0, ()), Frac(0, 1))
        return (poly)
    #
    # typical cases
    #
    if numJ == 1:  # single normal variable
        poln = mnorm_cond(m)  # kf = ['mu_s', 'rho_J', 'sigma_s', 'J_i']
        for k in poln:
            key = (k[0], k[1], k[2], (k[3],))
            poly.add_keyval(key, poln[k])
        return (poly)
    if numJ == 2:
        for i in range(0, m + 1):
            c = math.comb(m, i)
            pol1 = mnorm_cond(i)
            pol2 = mnorm_cond(m - i)  # mnorm_cond(0) supported
            # kf = ['mu_s', 'rho_J', 'sigma_s', 'J_i']
            for k1 in pol1:
                for k2 in pol2:
                    key = (k1[0] + k2[0], k1[1] + k2[1], k1[2] + k2[2], (k1[3], k2[3]))
                    val = c * pol1[k1] * pol2[k2]
                    poly.add_keyval(key, val)
        return (poly)
    # numJ > 2
    stack = []
    nextstep = "forward"
    while len(stack) < numJ:
        if nextstep == "forward":
            if len(stack) < numJ - 2:
                stack.append(0)
                nextstep = "forward"
            else:
                left = m - sum(stack)
                for i in range(left + 1):
                    stack.extend([i, left - i])
                    index = stack.copy()
                    #
                    c = comb(m, index)
                    poln = comb_poly(index)
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


def moments_y_to(l, J):
    """conditional moments of :math:`y_t` of orders :math:`0:l`

    :param int l: highest order of the conditional moments to derive, l >= 1
    :param tuple J: jump sizes in the variance
    :return: a list of polys with attribute ``keyfor`` =
      ('e^{kt}','t','k^{-}','beta_t','mu-theta/2','v0-theta','theta','sigma',
      'l_{1:n}', 'o_{1:n}', 'p_{2:n}', 'q_{2:n}',
      'sigma/2k','rho-sigma/2k','sqrt(1-rho^2)',
      'mu_s', 'rho_J', 'sigma_s', 'J_{1:n}')
    :rtype: list
    """
    moms = []
    moms_svvj = moments_y_to_svvj(l)  # 0-th to l-th moments
    #
    kf = moms_svvj[0].keyfor + ('mu_s', 'rho_J', 'sigma_s', 'J_{1:n}')
    #
    # special case: 0-th moment
    #
    P1 = Poly({(0, 0, 0, 0, 0, 0, 0, 0, (), (), (), (), 0, 0, 0, 0): Frac(1, 1)})
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
            pol2 = moment_IZs(n - n1, J)  # kf = ['mu_s', 'rho_J', 'sigma_s', 'J_{1:n}']
            for k1 in pol1:
                for k2 in pol2:
                    key = k1 + k2
                    poly.add_keyval(key, c * pol1[k1] * pol2[k2])
        poly.remove_zero()
        moms.append(poly)
    return moms


def moment_y(l, J):
    moms = moments_y_to(l, J)
    return moms[-1]


##########
# scalar
##########

def poly2num(poly, par):
    """Decode poly back to scalar

    :param Poly poly: poly to be decoded with attibute ``keyfor`` =
      ('e^{kt}','t','k^{-}','beta_t','mu-theta/2','v0-theta','theta','sigma',
      'l_{1:n}', 'o_{1:n}', 'p_{2:n}', 'q_{2:n}',
      'sigma/2k','rho-sigma/2k','sqrt(1-rho^2)',
      'mu_s', 'rho_J', 'sigma_s', 'J_{1:n}')
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
    mu_s = par['mu_s']
    rho_J = par['rho_J']
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
        l, o, p, q = K[8], K[9], K[10], K[11]
        #
        val *= fZ(l, o, p, q, k, s, J)
        #
        val *= ((sigma / (2 * k)) ** K[12]) * ((rho - sigma / (2 * k)) ** K[13])
        val *= ((1 - rho ** 2) ** (K[14] / 2))
        #
        val *= (mu_s ** K[15]) * (rho_J ** K[16]) * (sigma_s ** K[17])
        #
        f_J = 1
        for i in range(len(J)):
            f_J *= (J[i]) ** (K[18][i])
        val *= f_J
        #
        f += val * poly[K]
    return f


def m(l, par):
    """conditional moment in scalar

    :param int l: order of the conditional moment
    :param dict par: parameters in dict, ``jumptime`` (tuple) and ``jumpsize`` (tuple)
      must also be included
    :return: scalar of the conditional central moment
    :rtype: float
    """
    J = par['jumpsize']  # list of jump sizes
    moments = moments_y_to(l, J)
    moment = moments[-1]
    value = poly2num(moment, par)
    return value


if __name__ == "__main__":
    from pprint import pprint
    import sys
    import random

    print('Example usage of the module function\n')
    args = sys.argv[1:]
    numJ = 2 if len(args) == 0 else int(args[0])  # special cases: 0, 1, 2
    jumpsize = [random.expovariate(lambd=1.0) for i in range(numJ)]
    print(f"jumpsize = ")
    pprint(jumpsize)
    #
    l = 2 if len(args) <= 1 else int(args[1])
    moms = moments_y_to(l, jumpsize)  # 0-th to l-th moments
    kf = moms[0].keyfor
    print(f"\nmoment_y({l}, J) = \n")
    pprint(moms[l])
    print(f"\nwhich is a poly with keyfor = \n{kf}")

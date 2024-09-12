"""
Conditional Moments for SVVJ model
"""
import math
from fractions import Fraction as Frac

from ajdmom.poly import Poly
from ajdmom.utils import comb, fZ
from ajdmom.mdl_svvj.cmom import cmoments_y_to


def moment_comb(n, n1, n2, n3, n4, PY):
    """moment for this combination in the expansion of :math:`y_t^n`

    :param int n: n in :math:`E[y_t^n]`
    :param int n1: power of :math:`(\mu -\\theta/2)t`
    :param int n2: power of :math:`(v_0 - \\theta)\\beta_t`
    :param int n3: power of :math:`I\!E\!Z_t`
    :param int n4: power of :math:`I\!Z_t`
    :param Poly PY: poly of :math:`E[\overline{y}_t^{n_5}|v_0, z_s, 0\le s\le t]`
    :return: poly with attribute ``keyfor`` =
      ('e^{kt}','t','k^{-}','beta_t','mu-theta/2','v0-theta','theta','sigma',
      'l_{1:n}', 'o_{1:n}', 'p_{2:n}', 'q_{2:n}',
      'sigma/2k','rho-sigma/2k','sqrt(1-rho^2)')
    :rtype: Poly
    """
    kf = PY.keyfor[0:3] + ('beta_t', 'mu-theta/2') + PY.keyfor[3:]
    poly = Poly()
    poly.set_keyfor(kf)
    #
    n5 = n - n1 - n2 - n3 - n4
    c = comb(n, [n1, n2, n3, n4, n5])
    c *= ((-1) ** (n2 + n4)) * Frac(1, 2 ** (n3 + n4))
    #
    # PY.keyfor = ('e^{kt}','t','k^{-}','v0-theta','theta','sigma',
    #       'l_{1:n}', 'o_{1:n}', 'p_{2:n}', 'q_{2:n}',
    #       'sigma/2k','rho-sigma/2k','sqrt(1-rho^2)')
    for k in PY:
        key1 = (k[0] - n3, k[1] + n1, k[2] + n3 + n4, n2, n1, k[3] + n2, k[4], k[5])
        key3 = k[10:]
        #
        k6, k7, k8, k9 = k[6], k[7], k[8], k[9]
        # × IEZ_t^n3 IZ_t^n4
        #
        # expand k6: e^{ks_i} J_i, k7: s_i
        #
        k6 += tuple(1 for i in range(n3)) + tuple(0 for i in range(n4))
        k7 += tuple(0 for i in range(n3 + n4))
        #
        # expand k8: e^{0k(s_1 v ... v s_n)}, k9: (s_1 v ... v s_n)^0
        #
        n_to_expand = n3 + n4 - 1 if len(k[6]) == 0 else n3 + n4
        k8 += tuple(0 for i in range(n_to_expand))
        k9 += tuple(0 for i in range(n_to_expand))
        #
        key = key1 + (k6, k7, k8, k9) + key3
        poly.add_keyval(key, c * PY[k])
    return poly


def moments_y_to(l):
    """conditional moments of :math:`y_t` with orders :math:`0:l`

    :param int l: highest order of the conditional moments to derive, l >= 1
    :return: a list of polys with attribute ``keyfor`` =
      ('e^{kt}','t','k^{-}','beta_t','mu-theta/2','v0-theta','theta','sigma',
      'l_{1:n}', 'o_{1:n}', 'p_{2:n}', 'q_{2:n}',
      'sigma/2k','rho-sigma/2k','sqrt(1-rho^2)')
    :rtype: Poly
    """
    moms = []
    cmoms = cmoments_y_to(l)  # get 0:l-th central moments at once
    #
    kf = cmoms[0].keyfor[0:3] + ('beta_t', 'mu-theta/2') + cmoms[0].keyfor[3:]
    #
    # special case: 0-th moment
    #
    P1 = Poly({(0, 0, 0, 0, 0, 0, 0, 0, (), (), (), (), 0, 0, 0): Frac(1, 1)})
    P1.set_keyfor(kf)
    moms.append(P1)  # equiv to constant 1
    #
    # typical cases
    #
    for n in range(1, l + 1):
        poly = Poly()
        poly.set_keyfor(kf)
        for n1 in range(n, -1, -1):
            for n2 in range(n - n1, -1, -1):
                for n3 in range(n - n1 - n2, -1, -1):
                    for n4 in range(n - n1 - n2 - n3, -1, -1):
                        n5 = n - n1 - n2 - n3 - n4
                        poly.merge(moment_comb(n, n1, n2, n3, n4, cmoms[n5]))
        poly.remove_zero()
        moms.append(poly)
    return moms


def moment_y(l):
    moments = moments_y_to(l)
    return moments[-1]


##########
# scalar
##########

def poly2num(poly, par):
    """Decode poly back to scalar

    :param Poly poly: poly to be decoded with attibute ``keyfor`` =
      ('e^{kt}','t','k^{-}','beta_t','mu-theta/2','v0-theta','theta','sigma',
      'l_{1:n}', 'o_{1:n}', 'p_{2:n}', 'q_{2:n}',
      'sigma/2k','rho-sigma/2k','sqrt(1-rho^2)')
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
    moments = moments_y_to(l)
    moment = moments[-1]
    value = poly2num(moment, par)
    return value


if __name__ == "__main__":
    import sys
    from pprint import pprint

    print('Example usage of the module function\n')
    args = sys.argv[1:]
    l = 1 if len(args) == 0 else int(args[0])
    moms = moments_y_to(l)  # 0-th - l-th moments
    kf = moms[0].keyfor
    print(f"moment_y({l}) = ")
    pprint(moms[l])
    print(f"\nwhich is a poly with keyfor = \n{kf}")

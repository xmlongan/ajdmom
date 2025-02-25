"""
Central Moments for Two-Factor SV model
"""
import math
from fractions import Fraction as Frac

from ajdmom.poly import Poly
from ajdmom.ito_mom import moment_v
from ajdmom.itos_mom import moment_IEI_IEII, t_mul_t0
from ajdmom.utils import comb


def b_n(n1, n2, n3, n4, n5, n6, n7):
    """Coefficient corresponding to the combination

    :param int n1: times of constant being selected.
    :param int n2: times of :math:`v_{1,n-1}` being selected.
    :param int n3: times of :math:`v_{2,n-1}` being selected.
    :param int n4: times of :math:`eI_{1,n}` being selected.
    :param int n5: times of :math:`I_{1,n}` being selected.
    :param int n6: times of :math:`eI_{2,n}` being selected.
    :param int n7: times of :math:`I_{2,n}` being selected.
    :return: poly with attribute ``keyfor`` =
       ('e^{-k1h}','k1^{-}','theta1','sigma_v1',
       'e^{-k2h}','k2^{-}','theta2','sigma_v2').
    :rtype: Poly
    """
    b = Poly()
    kf = ['e^{-k1h}', 'k1^{-}', 'theta1', 'sigma_v1',
          'e^{-k2h}', 'k2^{-}', 'theta2', 'sigma_v2']
    b.set_keyfor(kf)
    #
    for i1 in range(n1, -1, -1):  # i1: 1/k1 * theta1
        for i2 in range(n1 - i1, -1, -1):  # i2: -e^{-k1h}/k1 * theta1
            for i3 in range(n1 - i1 - i2, -1, -1):  # i3: 1/k2 * theta2
                i4 = n1 - i1 - i2 - i3  # i4: -e^{-k2h}/k2 * theta2
                for j in range(n2 + 1):  # j: -e^{-k1h}/k1
                    for q in range(n3 + 1):  # q: -e^{-k2h}/k2
                        num = math.comb(n1, i1)
                        num *= math.comb(n1 - i1, i2)
                        num *= math.comb(n1 - i1 - i2, i3)
                        num *= math.comb(n2, j)
                        num *= math.comb(n3, q)
                        key = (i2 + j, i1 + i2 + n2 + n4 + n5, i1 + i2, n4 + n5,
                               i4 + q, i3 + i4 + n3 + n6 + n7, i3 + i4, n6 + n7)
                        num *= (-1) ** (i2 + i4 + (n2 + j) + (n3 + q) + n5 + n7)
                        den = 2 ** (n1 + n2 + n3 + n4 + n5 + n6 + n7)  # l - n8
                        b.add_keyval(key, Frac(num, den))
    return b


def vvee_IEI_IEII(n2, n3, n4, n5, n6, n7, n8):
    r"""Multiply :math:`IEI\_IEII` by :math:`v_{1,n-1}v_{2,n-1}e^{-k_1nh}e^{-k_2nh}`

    :math:`E[v_{1,n-1}^{n_2}v_{2,n-1}^{n_3}e^{-n_4k_1 nh}e^{-n_6k_2 nh} IEI\_IEII]`
       where

       .. math::

          IEI\_IEII = E[eI_{1,n}^{n_4} I_{1,n}^{n_5} eI_{2,n}^{n_6} I_{2,n}^{n_7}
          I_{n}^{*n_8}|v_{1,n-1}v_{2,n-1}]

    :param int n2: times of :math:`v_{1,n-1}` being selected.
    :param int n3: times of :math:`v_{2,n-1}` being selected.
    :param int n4: times of :math:`eI_{1,n}` being selected.
    :param int n5: times of :math:`I_{1,n}` being selected.
    :param int n6: times of :math:`eI_{2,n}` being selected.
    :param int n7: times of :math:`I_{2,n}` being selected.
    :param int n8: times of :math:`I_{n}^{*}` being selected.
    :return: poly with attribute ``keyfor`` =
       ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
       'e^{-(n1*k1+n2*k2)h}','h',
       'v_{1,n-1}','theta1','sigma_v1', 'v_{2,n-1}','theta2','sigma_v2').
    :rtype: Poly
    """
    poln = Poly()
    kf = ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
          'e^{-(n1*k1+n2*k2)h}', 'h',
          'v_{1,n-1}', 'theta1', 'sigma_v1', 'v_{2,n-1}', 'theta2', 'sigma_v2')
    poln.set_keyfor(kf)
    #
    poly = moment_IEI_IEII(n4, n5, n6, n7, n8)
    # keyfor =
    # ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
    # 'e^{(m_4*k1+m_6*k2)(n-1)h}','e^{(j_1*k1+j_2*k2)[t-(n-1)h]}','[t-(n-1)h]',
    # 'v_{1,n-1}','theta1','sigma_v1', 'v_{2,n-1}','theta2','sigma_v2')
    for k in poly:
        # * e^{-n4*k1*nh} e^{-n6*k2*nh}  * v_{1,n-1}^{n2} v_{2,n-1}^{n3}
        t1 = (n4 - k[2][0], n6 - k[2][1])
        key = (k[0], t1, k[3], k[4] + n2, k[5], k[6], k[7] + n3, k[8], k[9])
        poln.add_keyval(key, poly[k])
    return poln


def moment_comb(n, n1, n2, n3, n4, n5, n6, n7, n8):
    r"""Moment for this combination of given n1 to n8 with n.

    :param int n: l in :math:`E[\overline{y}_{n}^l]`.
    :param int n1: times of constant being selected.
    :param int n2: times of :math:`v_{1,n-1}` being selected.
    :param int n3: times of :math:`v_{2,n-1}` being selected.
    :param int n4: times of :math:`eI_{1,n}` being selected.
    :param int n5: times of :math:`I_{1,n}` being selected.
    :param int n6: times of :math:`eI_{2,n}` being selected.
    :param int n7: times of :math:`I_{2,n}` being selected.
    :param int n8: times of :math:`I_{n}^{*}` being selected.
    :return: poly with attribute ``keyfor`` =
       ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
       'e^{-(n1*k1+n2*k2)h}','h',
       'v_{1,n-1}','theta1','sigma_v1', 'v_{2,n-1}','theta2','sigma_v2').
    :rtype: Poly
    """
    poly = vvee_IEI_IEII(n2, n3, n4, n5, n6, n7, n8)
    # ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
    # 'e^{-(n1*k1+n2*k2)h}','h', 'v_{1,n-1}','theta1','sigma_v1',
    # 'v_{2,n-1}','theta2','sigma_v2')
    b = b_n(n1, n2, n3, n4, n5, n6, n7)
    # ['e^{-k1h}','k1^{-}','theta1','sigma_v1',
    #  'e^{-k2h}','k2^{-}','theta2','sigma_v2']
    kf = ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
          'e^{-(n1*k1+n2*k2)h}', 'h',
          'v_{1,n-1}', 'theta1', 'sigma_v1', 'v_{2,n-1}', 'theta2', 'sigma_v2')
    #
    poln = Poly()
    poln.set_keyfor(kf)
    #
    for k1 in poly:
        for k2 in b:
            # t0 = list(k1[0]); t0.insert(0,(1,0,k2[1])); t0.insert(0,(0,1,k2[5]))
            # t0 = tuple(t0)
            t0 = t_mul_t0((1, 0, k2[1]), k1[0])
            t0 = t_mul_t0((0, 1, k2[5]), t0)
            #
            t1 = (k1[1][0] + k2[0], k1[1][1] + k2[4])
            #
            key = (t0, t1, k1[2], k1[3], k1[4] + k2[2], k1[5] + k2[3],
                   k1[6], k1[7] + k2[6], k1[8] + k2[7])
            val = poly[k1] * b[k2]
            poln.add_keyval(key, val)
    # c = c_n(n, n1, n2, n3, n4, n5, n6, n7, n8)
    c = comb(n, [n1, n2, n3, n4, n5, n6, n7, n8])
    return c * poln


def sub_v(poly):
    """Substitute :math:`v_{1,n-1}` and :math:`v_{2,n-1}` with their moments

    :param Poly poly: poly with attribute ``keyfor`` =
       ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
       'e^{-(n1*k1+n2*k2)h}','h',
       'v_{1,n-1}','theta1','sigma_v1', 'v_{2,n-1}','theta2','sigma_v2').
    :return: poly with attribute ``keyfor`` =
       ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
       'e^{-(n1*k1+n2*k2)h}','h', 'theta1','sigma_v1', 'theta2','sigma_v2').
    :rtype: Poly
    """
    poly_sum = Poly()
    kf = ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
          'e^{-(n1*k1+n2*k2)h}', 'h', 'theta1', 'sigma_v1', 'theta2', 'sigma_v2')
    poly_sum.set_keyfor(kf)
    #
    poln = Poly()
    for k in poly:
        pol1 = moment_v(k[3])  # [theta1, sigma_v1^2/k1]
        for key in pol1:
            i, j = key
            # t0 = list(k[0]); t0.insert(0, (1,0,j)); t0 = tuple(t0)
            t0 = t_mul_t0((1, 0, j), k[0])
            #
            knw = (t0, k[1], k[2], k[4] + i, k[5] + 2 * j, k[6], k[7], k[8])
            val = poly[k] * pol1[key]
            poln.add_keyval(knw, val)
    for k in poln:
        pol2 = moment_v(k[5])  # [theta2, sigma_v2^2/k2]
        for key in pol2:
            i, j = key
            # t0 = list(k[0]); t0.insert(0, (0,1,j)); t0 = tuple(t0)
            t0 = t_mul_t0((0, 1, j), k[0])
            #
            knw = (t0, k[1], k[2], k[3], k[4], k[6] + i, k[7] + 2 * j)
            val = poln[k] * pol2[key]
            poly_sum.add_keyval(knw, val)
    return poly_sum


def cmoment_y(l):
    """Central moment of :math:`y_n` with order :math:`l`

    :param int l: order of the central moment.
    :return: poly with attribute ``keyfor`` =
       ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
       'e^{-(n1*k1+n2*k2)h}','h', 'theta1','sigma_v1', 'theta2','sigma_v2').
    :rtype: Poly
    """
    poly = Poly()
    kf = ['(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
          'e^{-(n1*k1+n2*k2)h}', 'h',
          'v_{1,n-1}', 'theta1', 'sigma_v1', 'v_{2,n-1}', 'theta2', 'sigma_v2']
    poly.set_keyfor(kf)
    #
    for n1 in range(l, -1, -1):
        for n2 in range(l - n1, -1, -1):
            for n3 in range(l - n1 - n2, -1, -1):
                for n4 in range(l - n1 - n2 - n3, -1, -1):
                    for n5 in range(l - n1 - n2 - n3 - n4, -1, -1):
                        for n6 in range(l - n1 - n2 - n3 - n4 - n5, -1, -1):
                            for n7 in range(l - n1 - n2 - n3 - n4 - n5 - n6, -1, -1):
                                n8 = l - n1 - n2 - n3 - n4 - n5 - n6 - n7
                                poly.merge(moment_comb(l, n1, n2, n3, n4, n5, n6, n7, n8))
    poly_sum = sub_v(poly)
    poly_sum.remove_zero()
    return poly_sum


##########
# scalar and (partial) derivative
##########

def dt0(t0, wrt):
    """Partial derivative of the tuple encoding k1 and k2

    :param tuple t0: tuple encoding k1 and k2 as
       '(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}'.
    :param str wrt: with respect to.
    :return: list of tuple
       ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}', val).
    :rtype: list
    """
    deriv = []
    if wrt == 'k1':
        for i in range(len(t0)):
            key = list(t0)
            k = t0[i]
            if k[2] != 0 and k[0] != 0:
                K = (k[0], k[1], k[2] + 1)
                key[i] = K
                val = -k[2] * k[0]
                deriv.append((tuple(key), val))
    elif wrt == 'k2':
        for i in range(len(t0)):
            key = list(t0)
            k = t0[i]
            if k[2] != 0 and k[1] != 0:
                K = (k[0], k[1], k[2] + 1)
                key[i] = K
                val = -k[2] * k[1]
                deriv.append((tuple(key), val))
    else:
        msg = f"wrt must be either 'k1' or 'k2' while {wrt} is supplied!"
        raise ValueError(msg)
    return deriv


def dpoly(poly, wrt):
    """Partial derivative of moment w.r.t. parameter wrt

    :param Poly poly: poly with attribute ``keyfor`` =
       ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
       'e^{-(n1*k1+n2*k2)h}','h', 'theta1','sigma_v1','theta2','sigma_v2').
    :param str wrt: with respect to.
    :return: poly with attribute ``keyfor`` =
       ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
       'e^{-(n1*k1+n2*k2)h}','h', 'theta1','sigma_v1','theta2','sigma_v2').
    :rtype: Poly
    """
    pold = Poly()
    kf = ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
          'e^{-(n1*k1+n2*k2)h}', 'h', 'theta1', 'sigma_v1', 'theta2', 'sigma_v2')
    pold.set_keyfor(kf)
    #
    # partial derivative w.r.t. k1 or k2
    if wrt == 'mu':
        return (pold)
    elif wrt == 'k1' or wrt == 'k2':
        for k in poly:
            t0 = k[0]
            if len(t0) != 0:
                deriv = dt0(t0, wrt)
                if len(deriv) != 0:
                    for t in deriv:
                        t0_new, val_new = t
                        knw = list(k)
                        knw[0] = t0_new
                        val = val_new * poly[k]
                        pold.add_keyval(tuple(knw), val)
        for k in poly:
            if wrt == 'k1':
                n = k[1][0]
            else:
                n = k[1][1]
            if n != 0:
                knw = list(k)
                knw[2] += 1
                val = (-n) * poly[k]
                pold.add_keyval(tuple(knw), val)
    # partial derivative w.r.t. other parameters
    elif wrt in ['theta1', 'sigma_v1', 'theta2', 'sigma_v2']:
        if wrt == 'theta1': i = 3
        if wrt == 'sigma_v1': i = 4
        if wrt == 'theta2': i = 5
        if wrt == 'sigma_v2': i = 6
        for k in poly:
            if k[i] != 0:
                knw = list(k)
                knw[i] -= 1
                val = k[i] * poly[k]
                pold.add_keyval(tuple(knw), val)
    else:
        candidates = "'k1','k2','theta1','sigma_v1','theta2','sigma_v2'"
        raise ValueError(f"wrt must be one of {candidates}!")
    return pold


def poly2num(poly, par):
    """Decode poly back to scalar

    :param Poly poly: poly to be decoded with attribute ``keyfor`` =
       ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
       'e^{-(n1*k1+n2*k2)h}','h', 'theta1','sigma_v1','theta2','sigma_v2').
    :param dict par: parameters in dict.
    :return: scalar of the poly.
    :rtype: float
    """
    k1 = par['k1']
    k2 = par['k2']
    h = par['h']
    theta1 = par['theta1']
    theta2 = par['theta2']
    sigma_v1 = par['sigma_v1']
    sigma_v2 = par['sigma_v2']
    #
    value = 0
    for k in poly:
        val = poly[k]
        t0 = k[0]
        for t in t0:
            val *= (t[0] * k1 + t[1] * k2) ** (-t[2])
        t1 = k[1]
        val *= math.exp(-(t1[0] * k1 + t1[1] * k2) * h)
        val *= h ** k[2]
        val *= theta1 ** k[3]
        val *= sigma_v1 ** k[4]
        val *= theta2 ** k[5]
        val *= sigma_v2 ** k[6]
        value += val
    return value


def cm(l, par):
    """Central moment in scalar

    :param int l: order of the central moment.
    :param dict par: parameters in dict.
    :return: scalar of the central moment.
    :rtype: float
    """
    cmoment = cmoment_y(l)
    value = poly2num(cmoment, par)
    return value


def dcm(l, par, wrt):
    """Partial derivative of central moment w.r.t. parameter wrt

    :param int l: order of the central moment.
    :param dict par: parameters in dict.
    :param str wrt: with respect to.
    :return: scalar of the partial derivative.
    :rtype: float
    """
    cmoment = cmoment_y(l)
    pold = dpoly(cmoment, wrt)
    value = poly2num(pold, par)
    return value


if __name__ == "__main__":
    # Example usage of the module, see 'tests/test_mdl_2fsv.py' for more test
    from pprint import pprint
    import sys

    print('\nExample usage of the module function\n')
    args = sys.argv[1:]
    n = 2 if len(args) == 0 else int(args[0])
    cmom = cmoment_y(n)
    print(f"cmoment_y({n}) = \n")
    pprint(cmom)
    print(f"\nwhich is a Poly with attribute keyfor = \n{cmom.keyfor}")

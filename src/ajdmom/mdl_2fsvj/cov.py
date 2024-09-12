"""
Covariance for the Two-Factor SV with jumps
"""
import math

from ajdmom.poly import Poly
from ajdmom.cpp_mom import mcpp
from ajdmom.itos_mom import t_mul_t0
from ajdmom.mdl_2fsv.cov import moment_yy as m_yy
from ajdmom.mdl_2fsvj.mom import (
    moment_y,
    dpoly,
    poly2num
)


def moment_yy(l1, l2):
    """Co-Moment :math:`E[y_n^{l_1}y_{n+1}^{l_2}]`

    :param int l1: :math:`l_1` in :math:`E[y_n^{l_1}y_{n+1}^{l_2}]`.
    :param int l2: :math:`l_2` in :math:`E[y_n^{l_1}y_{n+1}^{l_2}]`.
    :return: poly with attribute ``keyfor`` =
       ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
       'e^{-(n1*k1+n2*k2)h}','h','mu',
       'theta1','sigma_v1','theta2','sigma_v2', 'lambda','mu_j','sigma_j').
    :rtype: Poly
    """
    poly = Poly()
    kf = ['(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
          'e^{-(n1*k1+n2*k2)h}', 'h', 'mu',
          'theta1', 'sigma_v1', 'theta2', 'sigma_v2', 'lambda', 'mu_j', 'sigma_j']
    poly.set_keyfor(kf)
    for i in range(l1, -1, -1):
        for j in range(l2, -1, -1):
            pol1 = m_yy(i, j)
            # keyfor = '(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
            #   'e^{-(n1*k1+n2*k2)h}','h','mu',
            #   'theta1','sigma_v1','theta2','sigma_v2')
            pol2 = mcpp(l1 - i)
            pol3 = mcpp(l2 - j)
            # keyfor = ('lambda*h','mu_j','sigma_j')
            poln = pol2 * pol3
            c = math.comb(l1, i) * math.comb(l2, j)
            for k1 in pol1:
                for k2 in poln:
                    key = (k1[0], k1[1], k1[2] + k2[0], k1[3], k1[4], k1[5], k1[6], k1[7],
                           k2[0], k2[1], k2[2])
                    val = c * pol1[k1] * poln[k2]
                    poly.add_keyval(key, val)
    poly.remove_zero()
    return poly


def cov_yy(l1, l2):
    """Covariance :math:`cov(y_n^{l_1},y_{n+1}^{l_2})`

    :param int l1: :math:`l_1` in :math:`E[y_n^{l_1}y_{n+1}^{l_2}]`.
    :param int l2: :math:`l_2` in :math:`E[y_n^{l_1}y_{n+1}^{l_2}]`.
    :return: poly with attribute ``keyfor`` =
       ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
       'e^{-(n1*k1+n2*k2)h}','h','mu',
       'theta1','sigma_v1','theta2','sigma_v2', 'lambda','mu_j','sigma_j').
    :rtype: Poly
    """
    poly = Poly()
    kf = ['(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
          'e^{-(n1*k1+n2*k2)h}', 'h', 'mu',
          'theta1', 'sigma_v1', 'theta2', 'sigma_v2', 'lambda', 'mu_j', 'sigma_j']
    poly.set_keyfor(kf)
    #
    pol1 = moment_y(l1)  # keyfor = kf
    pol2 = moment_y(l2)  # keyfor = kf
    for k1 in pol1:
        for k2 in pol2:
            t0 = k1[0]  # simply: t0 = k1[0] + k2[0]    # first special one
            for t in k2[0]: t0 = t_mul_t0(t, t0)  # carefully treat
            t1 = (k1[1][0] + k2[1][0], k1[1][1] + k2[1][1])  # second special one
            #
            key = (t0, t1) + tuple(k1[i] + k2[i] for i in range(2, 11))
            val = pol1[k1] * pol2[k2]
            poly.add_keyval(key, val)
    #
    cov = moment_yy(l1, l2) - poly  # moment_yy(l1,l2) with keyfor = kf
    cov.remove_zero()
    return cov


##########
# scalar and (partial) derivative
##########

def cov(l1, l2, par):
    """Covariance in scalar

    :param int l1: *l1* in :math:`cov(y_n^{l_1},y_{n+1}^{l_2})`.
    :param int l2: *l2* in :math:`cov(y_n^{l_1},y_{n+1}^{l_2})`.
    :param dict par: parameters in dict.
    :return: scalar of the covariance.
    :rtype: float
    """
    covariance = cov_yy(l1, l2)
    value = poly2num(covariance, par)
    return value


def dcov(l1, l2, par, wrt):
    """Partial derivative of covariance w.r.t. parameter wrt

    :param int l1: *l1* in :math:`cov(y_n^{l_1},y_{n+1}^{l_2})`.
    :param int l2: *l2* in :math:`cov(y_n^{l_1},y_{n+1}^{l_2})`.
    :param dict par: parameters in dict.
    :param str wrt: with respect to.
    :return: scalar of the partial derivative.
    :rtype: float
    """
    covariance = cov_yy(l1, l2)
    pold = dpoly(covariance, wrt)
    value = poly2num(pold, par)
    return value


if __name__ == "__main__":
    # Example usage of the module, see 'tests/test_mdl_2fsvj.py' for more test
    from pprint import pprint
    import sys

    print('\nExample usage of the module function\n')
    args = sys.argv[1:]
    l1 = 1 if len(args) == 0 else int(args[0])
    l2 = 1 if len(args) <= 1 else int(args[1])
    cov_l1_l2 = cov_yy(l1, l2)
    print(f"cov_yy({l1},{l2}) = \n")
    pprint(cov_l1_l2)
    print(f"\nwhich is a Poly with attribute keyfor =\n{cov_l1_l2.keyfor}")

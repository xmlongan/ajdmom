"""
Covariance for One-Factor SV
"""
from ajdmom.poly import Poly
from ajdmom.utils import comb
from ajdmom.mdl_1fsv.cmom import ve_IEII
from ajdmom.mdl_1fsv.mom import (
    b_n,
    sub_v,
    moment_y,
    dpoly,
    poly2num
)


def exp_vn(l):
    """Expand :math:`v_n^l`

    :param int l: power
    :return: poly with attribute ``keyfor`` =
       ('e^{-knh}eI_n', 'e^{-kh}', 'v_{n-1}', 'theta', 'sigma_v').
    :rtype: Poly
    """
    polv = Poly()
    kf = ['e^{-knh}eI_n', 'e^{-kh}', 'v_{n-1}', 'theta', 'sigma_v']
    polv.set_keyfor(kf)
    for i in range(l, -1, -1):  # e^{-kh}v_{n-1}
        for j in range(l - i, -1, -1):  # 1*theta
            for p in range(l - i - j, -1, -1):  # -e^{-kh}*theta
                q = l - i - j - p  # sigma_v e^{-knh}eI_n
                key = (q, i + p, i, j + p, q)
                val = comb(l, [i, j, p, q]) * ((-1) ** p)
                polv.add_keyval(key, val)
    return polv


def ve_IEII_vn(n2, n3, n4, n5):
    """Compute :math:`v_n^{n_2}e^{-n_3k(n+1)h} IEII` and expand :math:`v_n`

    :param int n2: :math:`n_2` in above equation.
    :param int n3: :math:`n_3` in :math:`IEII` which is
       :math:`E[eI_{n+1}^{n_3}I_{n+1}^{n_4}I_{n+1}^{*n_5}|v_n]`.
    :param int n4: :math:`n_4` in :math:`IEII`.
    :param int n5: :math:`n_5` in :math:`IEII`.
    :return: poly with attribute ``keyfor`` =
       ('e^{-knh}eI_n','e^{-kh}','h','v_{n-1}','k^{-}','theta','sigma_v').
    :rtype: Poly
    """
    poly = ve_IEII(n2, n3, n4, n5)
    # ['e^{-kh}','h','v_n','k^{-}','theta','sigma_v']
    #
    # expand v_n
    poly_eIv = Poly()
    kf = ['e^{-knh}eI_n', 'e^{-kh}', 'h', 'v_{n-1}', 'k^{-}', 'theta', 'sigma_v']
    poly_eIv.set_keyfor(kf)
    #
    for k1 in poly:
        polv = exp_vn(k1[2])
        # ['e^{-knh}eI_n', 'e^{-kh}', 'v_{n-1}', 'theta', 'sigma_v']
        for k2 in polv:
            key = (k2[0], k1[0] + k2[1], k1[1], k2[2], k1[3], k1[4] + k2[3], k1[5] + k2[4])
            val = poly[k1] * polv[k2]
            poly_eIv.add_keyval(key, val)
    return poly_eIv


def moment_inner_comb(l1, m1, m2, m3, m4, m5, poly_eIv):
    """Moment for this inner combination in expansion of :math:`y_n^{l_1}`

    :param int l1: *l1* in :math:`y_n^{l_1}`.
    :param int m1: times of the constant in :math:`y_n` being selected.
    :param int m2: times of :math:`v_{n-1}` being selected.
    :param int m3: times of :math:`eI_{n}` being selected.
    :param int m4: times of :math:`I_{n}` being selected.
    :param int m5: times of :math:`I_{n}^{*}` being selected.
    :param Poly poly_eIv: poly with attribute ``keyfor`` =
       ('e^{-knh}eI_n','e^{-kh}','h','v_{n-1}','k^{-}','theta','sigma_v').
    :return: poly with attribute ``keyfor`` =
       ('e^{-kh}','h','v_{n-1}','k^{-}','mu','theta','sigma_v','rho',
       'sqrt(1-rho^2)').
    :rtype: Poly
    """
    poly = Poly()
    kf = ['e^{-kh}', 'h', 'v_{n-1}', 'k^{-}', 'theta', 'sigma_v']
    poly.set_keyfor(kf)
    #
    # combine and compute
    # k1: ['e^{-knh}eI_n','e^{-kh}','h','v_{n-1}','k^{-}','theta','sigma_v']
    # k2: ['e^{-kh}','h','v_{n-1}','k^{-}','theta','sigma_v']
    for k1 in poly_eIv:
        poln = ve_IEII(m2 + k1[3], m3 + k1[0], m4, m5)
        for k2 in poln:
            key = (k1[1] + k2[0], k1[2] + k2[1], k2[2], k1[4] + k2[3], k1[5] + k2[4], k1[6] + k2[5])
            val = poly_eIv[k1] * poln[k2]
            poly.add_keyval(key, val)
    #
    b = b_n(m1, m2, m3, m4, m5)
    # ['e^{-kh}','h','k^{-}','mu','theta','sigma_v','rho','sqrt(1-rho^2)']
    #
    keyfor = ['e^{-kh}', 'h', 'v_{n-1}', 'k^{-}', 'mu', 'theta', 'sigma_v', 'rho',
              'sqrt(1-rho^2)']
    keyIndexes = ([0, 1, 2, 3, -1, 4, 5, -1, -1], [0, 1, -1, 2, 3, 4, 5, 6, 7])
    poly = poly.mul_poly(b, keyIndexes, keyfor)
    #
    c = comb(l1, [m1, m2, m3, m4, m5])
    return c * poly


def moment_outer_comb(l2, n1, n2, n3, n4, n5, l1):
    r"""Moment for this outer combination in expansion of :math:`y_{n+1}^{l_2}`

    :param int l2: *l2* in :math:`y_{n+1}^{l_2}`.
    :param int n1: times of :math:`\theta` being selected.
    :param int n2: times of :math:`v_{n}` being selected.
    :param int n3: times of :math:`eI_{n+1}` being selected.
    :param int n4: times of :math:`I_{n+1}` being selected.
    :param int n5: times of :math:`I_{n+1}^{*}` being selected.
    :param int l1: *l1* in :math:`y_{n}^{l_1}`.
    :return: poly with attibute ``keyfor`` =
       ('e^{-kh}','h','v_{n-1}','k^{-}','mu','theta','sigma_v','rho',
       'sqrt(1-rho^2)').
    :rtype: Poly
    """
    # c = c_n(l2, n1, n2, n3, n4, n5)
    c = comb(l2, [n1, n2, n3, n4, n5])
    b = b_n(n1, n2, n3, n4, n5)
    # ['e^{-kh}','h','k^{-}','mu','theta','sigma_v','rho','sqrt(1-rho^2)']
    #
    poly_eIv = ve_IEII_vn(n2, n3, n4, n5)
    # ['e^{-knh}eI_n','e^{-kh}','h','v_{n-1}','k^{-}','theta','sigma_v']
    #
    poly = Poly()
    kf = ['e^{-kh}', 'h', 'v_{n-1}', 'k^{-}', 'mu', 'theta', 'sigma_v', 'rho',
          'sqrt(1-rho^2)']
    poly.set_keyfor(kf)
    #
    for m1 in range(l1, -1, -1):
        for m2 in range(l1 - m1, -1, -1):
            for m3 in range(l1 - m1 - m2, -1, -1):
                for m4 in range(l1 - m1 - m2 - m3, -1, -1):
                    m5 = l1 - m1 - m2 - m3 - m4
                    poly.merge(moment_inner_comb(l1, m1, m2, m3, m4, m5, poly_eIv))
    keyIndexes = ([0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, -1, 2, 3, 4, 5, 6, 7])
    poly = poly.mul_poly(b, keyIndexes, kf)
    return c * poly


def moment_yy(l1, l2):
    """Moment :math:`E[y_n^{l_1}y_{n+1}^{l_2}]` as in equation :eq:`moment_yy`

    :param int l1: *l1* in :math:`E[y_n^{l_1}y_{n+1}^{l_2}]`.
    :param int l2: *l2* in :math:`E[y_n^{l_1}y_{n+1}^{l_2}]`.
    :return: poly with attribute ``keyfor`` =
       ('e^{-kh}','h','k^{-}','mu','theta','sigma_v','rho','sqrt(1-rho^2)').
    :rtype: Poly
    """
    poly = Poly()
    kf = ['e^{-kh}', 'h', 'v_{n-1}', 'k^{-}', 'mu', 'theta', 'sigma_v', 'rho',
          'sqrt(1-rho^2)']
    poly.set_keyfor(kf)
    #
    for n1 in range(l2, -1, -1):
        for n2 in range(l2 - n1, -1, -1):
            for n3 in range(l2 - n1 - n2, -1, -1):
                for n4 in range(l2 - n1 - n2 - n3, -1, -1):
                    n5 = l2 - n1 - n2 - n3 - n4
                    poly.merge(moment_outer_comb(l2, n1, n2, n3, n4, n5, l1))
    poly = sub_v(poly)
    poly.remove_zero()
    return poly


def cov_yy(l1, l2):
    """Covariance :math:`cov(y_n^{l_1},y_{n+1}^{l_2})` in equation :eq:`cov_yy`

    :param int l1: *l1* in :math:`cov(y_n^{l_1},y_{n+1}^{l_2})`.
    :param int l2: *l2* in :math:`cov(y_n^{l_1},y_{n+1}^{l_2})`.
    :return: poly with attribute ``keyfor`` =
       ('e^{-kh}','h','k^{-}','mu','theta','sigma_v','rho','sqrt(1-rho^2)').
    :rtype: Poly
    """
    #
    # moment_yy(l1,l2), moment_y(l1), moment_y(l2)
    # e^{-kh}, h, k^{-1}, mu, theta,  sigma_v, rho,  sqrt(1-rho^2)
    cov = moment_yy(l1, l2) - (moment_y(l1) * moment_y(l2))
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
    :return: scalar of the partial derivative.
    :rtype: float
    """
    covariance = cov_yy(l1, l2)
    pold = dpoly(covariance, wrt)
    value = poly2num(pold, par)
    return value


if __name__ == "__main__":
    # Example usage of the module, see 'tests/test_mdl_1fsv.py' for more test
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

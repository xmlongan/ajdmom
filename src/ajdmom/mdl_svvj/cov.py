"""
Covariance for the SVVJ model
"""
import math
from fractions import Fraction

from ajdmom import Poly
from ajdmom.utils import comb, simplify_rho
from ajdmom.mdl_svcj.cond_ieii_ieziziz_mom import moment_ieii_ieziziz
from ajdmom.mdl_svcj.mom import moment_v_theta
from ajdmom.mdl_svvj.mom import key_times_poly, poly2num, moment_y, expand_Ev


def exp_vn_theta(l):
    r"""Expand :math:`(v_n - \theta)^l`

    :param int l: power of :math:`(v_n - \theta)`
    :return: poly with attribute ``keyfor`` =
      ('e^{-knh}IE_n', 'e^{-knh}IEZ_n', 'e^{-kh}', 'v_{n-1}-theta', 'sigma_v')
    :rtype: Poly
    """
    polv = Poly()
    kf = ['e^{-knh}IE_n', 'e^{-knh}IEZ_n', 'e^{-kh}', 'v_{n-1}-theta', 'sigma_v']
    polv.set_keyfor(kf)
    for i in range(l + 1):
        for j in range(l - i + 1):
            p = l - i - j
            bino = comb(l, [i, j, p])
            key = (j, p, i, i, j)
            polv.add_keyval(key, Fraction(bino, 1))
    return polv

def ve_IEII_IEZIZ(n0, n1, n2, n3, n4, n5):
    r"""
    Derive :math:`(v_n-\theta)^{n_0} e^{-(n_1+n_4)k(n+1)h}
    \mathbb{E}[I\!E_{n+1}^{n_1} I_{n+1}^{n_2} I_{n+1}^{*n_3}
    I\!E\!Z_{n+1}^{n_4} I\!Z_{n+1}^{n_5}]`
    """
    ve_poly = Poly()
    kf = ['e^{-kh}', 'h', 'k^{-}', 'v_n-theta', 'theta', 'sigma_v', 'lmbd', 'mu_v']
    ve_poly.set_keyfor(kf)
    #
    poly = moment_ieii_ieziziz(n1, n2, n3, n4, n5, 0)
    # with keyfor
    # 'e^{knh}' + ('e^{kh}', 'h', 'k^{-}', 'v_n-theta', 'theta', 'sigma', 'lmbd', 'mu_v',
    # 'mu_s', 'sigma_s')
    # and the power for the eliminated 'e^{knh}' always = m1 + m4
    # may error: further check!
    for k, v in poly.items():
        # × e^{-(n_1+n_4)k(n+1)h}          × (v_n-theta)^{m_0}
        key = (n1 + n4 - k[0], k[1], k[2], n0 + k[3], k[4], k[5], k[6], k[7])
        ve_poly.add_keyval(key, v)
    return ve_poly

def ve_IEII_IEZIZ_vn(n0, n1, n2, n3, n4, n5):
    r"""Derive :math:`(v_n-\theta)^{n_0}e^{-(n_1+n_4)k(n+1)h} IEII_IEZIZ` and expand :math:`v_n-\theta`

    :param int n0: order of :math:`v_n-\theta`
    :param int n1: order of :math:`IE_{n+1}`
    :param int n2: order of :math:`I_{n+1}`
    :param int n3: order of :math:`I_{n+1}^{*}`
    :param int n4: order of :math:`IEZ_{n+1}`
    :param int n5: order of :math:`IZ_{n+1}`
    :return: poly with attribute ``keyfor`` =
      ('e^{-knh}IE_n', 'e^{-knh}IEZ_n', 'e^{-kh}', 'h', 'k^{-}', 'v_{n-1}-theta',
      'theta', 'sigma_v', 'lmbd', 'mu_v')
    :rtype: Poly
    """
    ve_poly = ve_IEII_IEZIZ(n0, n1, n2, n3, n4, n5)
    #
    # expand (v_n - theta)
    #
    poly = Poly()
    kf = ['e^{-knh}IE_n', 'e^{-knh}IEZ_n',
          'e^{-kh}', 'h', 'k^{-}', 'v_{n-1}-theta', 'theta', 'sigma_v', 'lmbd', 'mu_v']
    poly.set_keyfor(kf)
    #
    for k1 in ve_poly:
        # ['e^{-kh}', 'h', 'k^{-}', 'v_n-theta', 'theta', 'sigma_v', 'lmbd', 'mu_v']
        polv = exp_vn_theta(k1[3])
        # ['e^{-knh}IE_n', 'e^{-knh}IEZ_n', 'e^{-kh}', 'v_{n-1}-theta', 'sigma_v']
        for k2 in polv:
            key = (        k2[0],  # 'e^{-knh}IE_n'
                           k2[1],  # 'e^{-knh}IEZ_n'
                   k1[0] + k2[2],  # 'e^{-kh}'
                   k1[1],          # 'h'
                   k1[2],          # 'k^{-}'
                           k2[3],  # 'v_{n-1}-theta'
                   k1[4],          # 'theta'
                   k1[5] + k2[4],  # 'sigma_v'
                   k1[6],          # 'lmbd'
                   k1[7])          # 'mu_v'
            val = ve_poly[k1] * polv[k2]
            poly.add_keyval(key, val)
    return poly

def b_n(m0, m1, m2, m3, m4, m5, m6):
    kf = ['e^{-kh}', 'h', 'k^{-}', 'mu', 'theta', 'sigma_v',
          'rho', 'sqrt(1-rho^2)', 'lmbd', 'mu_v']
    poly = Poly()
    poly.set_keyfor(kf)
    for i in range(m0 + 1):         # (1 - e^{-kh})/(2k)
        for j in range(m2 + 1):     # (rho - sigma_v/2k)
            for o in range(m6 + 1): # (mu - theta/2)h
                bino = math.comb(m0, i) * math.comb(m2, j) * math.comb(m6, o)
                sign = (-1) ** (m0 + m5 + i + j + o)
                key = (i,       # 'e^{-kh}'
                       m6,      # 'h'
                       m0 + m1 + m4 + m5 + j, # 'k^{-}'
                       m6 - o,  # 'mu'
                       o,       # theta
                       m1 + j,  # sigma_v
                       m2 - j,  # rho
                       m3,      # sqrt(1-rho^2)
                       0, 0)
                val = sign * bino * Fraction(1, 2 ** (m0 + m1 + m4 + m5 + j + o))
                poly.add_keyval(key, val)
    return poly

def moment_inner_comb(l1, m0, m1, m2, m3, m4, m5, m6, poly_eIv):
    r"""Moment for this inner combination in expansion of :math:`y_n^{l_1}`

    :param int l1: order in :math:`y_n^{l_1}`.
    :param int m0: times of :math:`(v_{n-1}-\theta)` being selected.
    :param int m1: times of :math:`I\!E_n` being selected.
    :param int m2: times of :math:`I_n` being selected.
    :param int m3: times of :math:`I_n^{*}` being selected.
    :param int m4: times of :math:`I\!E\!Z_n` being selected.
    :param int m5: times of :math:`I\!Z_n` being selected.
    :param int m6: times of :math:`(\mu - \theta/2)h` being selected.
    :param Poly poly_eIv: poly with attribute ``keyfor`` =
      ('e^{-knh}IE_n', 'e^{-knh}IEZ_n', 'e^{-kh}', 'h', 'k^{-}', 'v_{n-1}-theta',
      'theta', 'sigma_v', 'lmbd', 'mu_v')
    :return: poly with attribute ``keyfor`` =
      (e^{-kh}', 'h', 'k^{-}', 'mu', 'v_{n-1}-theta',
      'theta', 'sigma_v', 'rho', 'sqrt(1-rho^2)', 'lmbd', 'mu_v')
    :rtype: Poly
    """
    poly = Poly()
    kf = ['e^{-kh}', 'h', 'k^{-}', 'v_{n-1}-theta', 'theta', 'sigma_v', 'lmbd', 'mu_v']
    poly.set_keyfor(kf)
    #
    # combine and compute
    # k1: ['e^{-knh}IE_n', 'e^{-knh}IEZ_n', 'e^{-kh}', 'h', 'k^{-}', 'v_{n-1}-theta',
    #      'theta', 'sigma_v', 'lmbd', 'mu_v']
    # k2: ['e^{-kh}', 'h', 'k^{-}', 'v_{n-1}-theta', 'theta', 'sigma_v', 'lmbd', 'mu_v']
    for k1 in poly_eIv:
        poln = ve_IEII_IEZIZ(m0 + k1[5], m1 + k1[0], m2, m3, m4 + k1[1], m5)
        for k2 in poln:
            key = (k1[2] + k2[0],  # 'e^{-kh}'
                   k1[3] + k2[1],  # 'h'
                   k1[4] + k2[2],  # k^{-}
                   # k1[5] + k2[3],  # v_{n-1}-theta
                           k2[3],  # v_{n-1}-theta
                   k1[6] + k2[4],  # theta
                   k1[7] + k2[5],  # sigma_v
                   k1[8] + k2[6],  # lmbd
                   k1[9] + k2[7])  # mu_v
            val = poly_eIv[k1] * poln[k2]
            poly.add_keyval(key, val)
    b = b_n(m0, m1, m2, m3, m4, m5, m6)
    # kf = ['e^{-kh}', 'h', 'k^{-}', 'mu', 'theta', 'sigma_v',
    #       'rho', 'sqrt(1-rho^2)', 'lmbd', 'mu_v']
    keyfor = ['e^{-kh}', 'h', 'k^{-}', 'mu', 'v_{n-1}-theta', 'theta', 'sigma_v',
              'rho', 'sqrt(1-rho^2)', 'lmbd', 'mu_v']
    keyIndexes = (
        [0, 1, 2, -1, 3, 4, 5, -1, -1, 6, 7],  # poly index
        [0, 1, 2, 3, -1, 4, 5, 6, 7, 8, 9]     # b index
    )
    poly = poly.mul_poly(b, keyIndexes, keyfor)
    c = comb(l1, [m0, m1, m2, m3, m4, m5, m6])
    return c * poly

def moment_outer_comb(l2, n0, n1, n2, n3, n4, n5, n6, l1):
    """Moment for this outer combination in expansion of :math:`y_{n+1}^{l_2}`"""
    c = comb(l2, [n0, n1, n2, n3, n4, n5, n6])
    b = b_n(n0, n1, n2, n3, n4, n5, n6)
    # kf = ['e^{-kh}', 'h', 'k^{-}', 'mu', 'theta', 'sigma_v',
    #       'rho', 'sqrt(1-rho^2)', 'lmbd', 'mu_v']
    poly_eIv = ve_IEII_IEZIZ_vn(n0, n1, n2, n3, n4, n5)
    # kf = ['e^{-knh}IE_n', 'e^{-knh}IEZ_n',
    #       'e^{-kh}', 'h', 'k^{-}', 'v_{n-1}-theta', 'theta', 'sigma_v', 'lmbd', 'mu_v']
    poly = Poly()
    kf = ['e^{-kh}', 'h', 'k^{-}', 'mu', 'v_{n-1}-theta', 'theta', 'sigma_v',
          'rho', 'sqrt(1-rho^2)', 'lmbd', 'mu_v']
    poly.set_keyfor(kf)
    #
    for m0 in range(l1 + 1):
        for m1 in range(l1 - m0 + 1):
            for m2 in range(l1 - m0 - m1 + 1):
                for m3 in range(l1 - m0 - m1 - m2 + 1):
                    for m4 in range(l1 - m0 - m1 - m2 - m3 + 1):
                        for m5 in range(l1 - m0 - m1 - m2 - m3 - m4 + 1):
                            m6 = l1 - m0 - m1 - m2 - m3 - m4 - m5
                            poly.merge(moment_inner_comb(l1, m0, m1, m2, m3, m4, m5, m6, poly_eIv))
    keyIndexes = (
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # poly index
        [0, 1, 2, 3, -1, 4, 5, 6, 7, 8, 9]   # b index
    )
    poly = poly.mul_poly(b, keyIndexes, kf)
    return c * poly

def moment_yy(l1, l2):
    poly = Poly()
    kf = ['e^{-kh}', 'h', 'k^{-}', 'mu', 'v_{n-1}-theta', 'theta', 'sigma_v',
          'rho', 'sqrt(1-rho^2)', 'lmbd', 'mu_v']
    poly.set_keyfor(kf)
    #
    for n0 in range(l2 + 1):
        for n1 in range(l2 - n0 + 1):
            for n2 in range(l2 - n0 - n1 + 1):
                for n3 in range(l2 - n0 - n1 - n2 + 1):
                    for n4 in range(l2 - n0 - n1 - n2 - n3 + 1):
                        for n5 in range(l2 - n0 - n1 - n2 - n3 - n4 + 1):
                            n6 =l2 - n0 - n1 - n2 - n3 - n4 - n5
                            poly.merge(moment_outer_comb(l2, n0, n1, n2, n3, n4, n5, n6, l1))
    poly = simplify_rho(poly, 8)
    # ['e^{-kh}', 'h', 'k^{-}', 'mu', 'v_{n-1}-theta', 'theta', 'sigma_v', 'rho',
    #  'lmbd', 'mu_v']
    #
    # be careful of the changes, 'e^{-kh}' -> 'e^{kt}', 'h' -> 't', 'sigma_v' -> 'sigma'
    #
    kf = ['e^{kt}', 't', 'k^{-}', 'mu', 'E[v]', 'theta', 'sigma', 'rho',
          'lmbd', 'mu_v']
    poln = Poly()
    poln.set_keyfor(kf)
    for k, v in poly.items():
        k = (-k[0],) + k[1:]         # from 'e^{-kh}' to 'e^{kt}'
        # k: ['e^{kt}', 't', 'k^{-}', 'mu', 'v0-theta', 'theta', 'sigma', 'rho',
        #      'lmbd', 'mu_v']
        pol1 = moment_v_theta(k[4])  # ['k^{-}', 'E[v]', 'sigma', 'lmbd', 'mu_v']
        pol2 = key_times_poly(v, k, pol1)
        poln.merge(pol2)
    return poln


def cov_yy(l1, l2):
    # moment_yy: ['e^{kt}', 't', 'k^{-}', 'mu', 'E[v]', 'theta', 'sigma', 'rho', 'lmbd', 'mu_v']
    # moment_y:  ['e^{kt}', 't', 'k^{-}', 'mu', 'E[v]', 'theta', 'sigma', 'rho', 'lmbd', 'mu_v']
    cov = moment_yy(l1, l2) - (moment_y(l1) * moment_y(l2))
    cov.remove_zero()
    return cov

def cov(l1, l2, par):
    covariance = cov_yy(l1, l2)
    value = poly2num(covariance, par)
    return value


if __name__ == "__main__":
    # Example usage of the module, see 'tests/test_mdl_svvj.py' for more test
    from pprint import pprint
    import sys

    print('\nExample usage of the module function\n')
    args = sys.argv[1:]
    l1 = 2 if len(args) == 0 else int(args[0])
    l2 = 1 if len(args) <= 1 else int(args[1])
    cov_l1_l2 = cov_yy(l1, l2)
    cov_l1_l2 = expand_Ev(cov_l1_l2)
    print(f"cov_yy({l1},{l2}) = ")
    pprint(cov_l1_l2)
    print(f"which is a Poly with attribute keyfor =\n{cov_l1_l2.keyfor}")
    print(f'total {len(cov_l1_l2)} key-values\n')

    cov12 = expand_Ev(cov_yy(1, 2))
    cov21 = expand_Ev(cov_yy(2, 1))
    diff = cov12 - cov21
    pprint(diff)
    print(f'cov12 - cov21: total {len(diff)} key-values')

r"""
Conditional Joint Central Moment

Conditional joint central moment of volatility and return,
i.e., :math:`\mathbb{E}[\bar{v}_t^n \bar{y}_t^m|v_0]`.
The derivation reduces to
:math:`\mathbb{E}[I\!E_t^{m_1+n_1} I_t^{m_2} I_t^{*m_3}
I\!E\!Z_t^{m_4+n_2} I\!Z_t^{m_5} I\!Z_t^{*m_6}|v_0]`.
"""
import math
from fractions import Fraction
from pathlib import Path

from ajdmom import Poly
from ajdmom.utils import comb, simplify_rho

from ajdmom.mdl_svcj.cond_ieii_ieziziz_mom import moment_ieziziz
from ajdmom.mdl_svcj.cond_ieii_ieziziz_mom import ieziziz_to_ieii_ieziziz
from ajdmom.mdl_svcj.cond_ieii_ieziziz_mom import recursive_ieii_ieziziz


# Prepare moments of ieii_ieziziz in an efficient way

def ieii_ieziziz_comb(n: int, m: int) -> list:
    """Breakdown :math:`\bar{v}_t^n \bar{y}_t^m`

    :param n: power of :math:`\bar{v}_t^n`
    :param m: power of :math:`\bar{y}_t^m`
    :return: a list of the combinations
    :rtype: list
    """
    comb = []
    for m1 in range(m + 1):
        for m2 in range(m - m1 + 1):
            for m3 in range(m - m1 - m2 + 1):
                for m4 in range(m - m1 - m2 - m3 + 1):
                    for m5 in range(m - m1 - m2 - m3 - m4 + 1):
                        m6 = m - m1 - m2 - m3 - m4 - m5
                        for n1 in range(n + 1):
                            n2 = n - n1
                            item = (m1 + n1, m2, m3, m4 + n2, m5, m6)
                            if item not in comb: comb.append(item)
    return comb


def ieii_ieziziz_comb_to(n: int) -> list:
    """Collection up to power

    :param int n: sum of powers of :math:`\bar{v}` and :math:`\bar{y}_t`
    :return: a list of the combinations in ascending order
    :rtype: list
    """
    comb_all = []
    for i in range(n + 1):
        for j in range(n - i + 1):
            comb = ieii_ieziziz_comb(i, j)
            for item in comb:
                if item not in comb_all: comb_all.append(item)
    # sort in ascending order
    comb_all = sorted(comb_all, key=lambda t: (sum(t), t))  # very important
    return comb_all


def moment_ieii_ieziziz_comb(combs: list) -> dict:
    """Conditional joint moment for power in combs

    :param list combs: list of combinations, [(n1,...,n6), ...], in ascending order
    :return: dict of polys with ``keyfor`` =
     ('e^{kt}','t','k^{-}','v0-theta','theta','sigma',
     'lmbd','mu_v','mu_s','sigma_s')
    :rtype: dict
    """
    #
    # ieii_ieziziz: a dict of moments of E[IE_t^n1 I_t^n2 I_t^{*n3} IEZ_t^n4 IZ_t^n5 IZ_t^{*n6}]
    #
    ieii_ieziziz = {}
    #
    kf = ['e^{kt}', 't', 'k^{-}', 'v0-theta', 'theta', 'sigma',
          'lmbd', 'mu_v', 'mu_s', 'sigma_s']
    #
    # n1 + n2 + n3 + n4 + n5 + n6 = 0: special case
    #
    ieii_ieziziz[(0, 0, 0, 0, 0, 0)] = Poly.const_one(kf)  # equiv to constant 1
    #
    # n1 + n2 + n3 + n4 + n5 + n6 = 1
    #
    ieii_ieziziz[(1, 0, 0, 0, 0, 0)] = Poly.const_zero(kf)  # equiv to constant 0
    ieii_ieziziz[(0, 1, 0, 0, 0, 0)] = Poly.const_zero(kf)
    ieii_ieziziz[(0, 0, 1, 0, 0, 0)] = Poly.const_zero(kf)
    #
    ieii_ieziziz[(0, 0, 0, 1, 0, 0)] = ieziziz_to_ieii_ieziziz(moment_ieziziz(1, 0, 0))
    ieii_ieziziz[(0, 0, 0, 0, 1, 0)] = ieziziz_to_ieii_ieziziz(moment_ieziziz(0, 1, 0))
    ieii_ieziziz[(0, 0, 0, 0, 0, 1)] = ieziziz_to_ieii_ieziziz(moment_ieziziz(0, 0, 1))
    #
    # n1 + n2 + n3 + n4 + n5 + n6 >= 2
    #
    # Create the subfolder, including any parent folders if they don't exist
    # The `parents=True` argument creates parent directories as needed
    # The `exist_ok=True` argument prevents an error if the folder already exists
    subdir = Path('ieii_ieziziz')
    subdir.mkdir(parents=True, exist_ok=True)
    print(f'Creating subdir: {subdir} if it does not exist.')
    #
    for comb in combs:
        n1, n2, n3, n4, n5, n6 = comb
        poly = recursive_ieii_ieziziz(n1, n2, n3, n4, n5, n6, ieii_ieziziz)
        # poly.remove_zero()
        ieii_ieziziz[(n1, n2, n3, n4, n5, n6)] = poly
        fname = f'{subdir}/{n1}-{n2}-{n3}-{n4}-{n5}-{n6}.csv'
        poly.write_to_csv(fname)
    return ieii_ieziziz


def moment_ieii_ieziziz_csv(n1, n2, n3, n4, n5, n6):
    """Read conditional joint moment from csv file

    :param int n1: order of :math:`IE_t`
    :param int n2: order of :math:`I_t`
    :param int n3: order of :math:`I_t^{*}`
    :param int n4: order of :math:`IEZ_t`
    :param int n5: order of :math:`IZ_t`
    :param int n6: order of :math:`IZ_t^{*}`
    :return: poly with ``keyfor`` = ('e^{kt}','t','k^{-}','v0-theta','theta','sigma',
     'lmbd','mu_v','mu_s','sigma_s')
    :rtype: Poly
    """
    fname = f'ieii_ieziziz/{n1}-{n2}-{n3}-{n4}-{n5}-{n6}.csv'
    return Poly.read_csv(fname)


# conditional joint moment E[v_t^n y_t^m|v_0]

def b_mn(m1, m2, m3, m4, m5, m6, n1, n2):
    """Coefficient for this combination"""
    kf = ['e^{kt}', 't', 'k^{-}',
          'v0-theta', 'theta', 'sigma', 'rho', 'sqrt(1-rho^2)',
          'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s']
    poly = Poly()
    poly.set_keyfor(kf)
    for i1 in range(m2 + 1):
        for i2 in range(m5 + 1):
            bino = math.comb(m2, i1) * math.comb(m5, i2)
            sign = (-1) ** (i1 + i2)
            val = bino * sign * Fraction(1, 2 ** (m1 + i1 + m4 + i2))
            key = (-(m1 + m4 + n1 + n2), 0, m1 + i1 + m4 + i2,
                   0, 0, m1 + i1 + n1, m2 - i1, m3,
                   0, 0, m5 - i2, 0, 0)
            poly.add_keyval(key, val)
    return poly


def align(poly, keyfor):
    """Align poly to keyfor

    :param Poly poly: Poly object with ``keyfor`` =
     ('e^{kt}', 't', 'k^{-}', 'v0-theta', 'theta', 'sigma',
     'lmbd', 'mu_v', 'mu_s', 'sigma_s')
    :param list keyfor: the list
     ['e^{kt}', 't', 'k^{-}', 'v0-theta', 'theta', 'sigma',
     'rho', 'sqrt(1-rho^2)',
     'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s']
    :return: Poly object with ``keyfor`` = keyfor
    :rtype: Poly
    """
    poln = Poly()
    poln.set_keyfor(keyfor)
    for k, v in poly.items():
        key = k[:6] + (0, 0) + k[6:8] + (0,) + k[8:]
        poln.add_keyval(key, v)
    return poln


def joint_cmom(n, m):
    """Conditional joint central moment of volatility and return

    :param int n: power of :math:`\bar{v}_t`
    :param int m: power of :math:`\bar{y}_t`
    :return: Poly object with ``keyfor`` =
     ('e^{kt}', 't', 'k^{-}', 'v0-theta', 'theta', 'sigma', 'rho',
     'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s')
    :rtype: Poly
    """
    kf = ['e^{kt}', 't', 'k^{-}',
          'v0-theta', 'theta', 'sigma', 'rho', 'sqrt(1-rho^2)',
          'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s']
    poly = Poly()
    poly.set_keyfor(kf)
    for m1 in range(m + 1):
        for m2 in range(m - m1 + 1):
            for m3 in range(m - m1 - m2 + 1):
                for m4 in range(m - m1 - m2 - m3 + 1):
                    for m5 in range(m - m1 - m2 - m3 - m4 + 1):
                        m6 = m - m1 - m2 - m3 - m4 - m5
                        bino1 = comb(m, [m1, m2, m3, m4, m5, m6])
                        for n1 in range(n + 1):
                            n2 = n - n1
                            bino2 = math.comb(n, n1)
                            bino = bino1 * bino2
                            b = b_mn(m1, m2, m3, m4, m5, m6, n1, n2)
                            poln = moment_ieii_ieziziz_csv(m1 + n1, m2, m3, m4 + n2, m5, m6)
                            # kf = ['e^{kt}', 't', 'k^{-}',
                            #       'v0-theta', 'theta', 'sigma',
                            #       'lmbd', 'mu_v', 'mu_s', 'sigma_s']
                            poln = align(poln, kf)
                            poln = bino * (b * poln)
                            poly.merge(poln)
    return simplify_rho(poly, 7)  # expand sqrt(1-rho^2)


def poly2num(poly, par):
    """Evaluate poly to numerical value"""
    # kf = ['e^{kt}', 't', 'k^{-}',
    #       'v0-theta', 'theta', 'sigma', 'rho',
    #       'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s']
    k = par['k']
    mu, v0, theta, sigma, rho = par['mu'], par['v0'], par['theta'], par['sigma'], par['rho']
    lmbd, mu_v, rhoJ, mu_s, sigma_s = par['lmbd'], par['mu_v'], par['rhoJ'], par['mu_s'], par['sigma_s']
    h = par['h']
    #
    f = 0
    for K, V in poly.items():
        val = math.exp(K[0] * k * h) * (h ** K[1]) / (k ** K[2])
        val *= ((v0 - theta) ** K[3]) * (theta ** K[4])
        val *= (sigma ** K[5]) * (rho ** K[6])
        val *= (lmbd ** K[7]) * (mu_v ** K[8]) * (rhoJ ** K[9])
        val *= (mu_s ** K[10]) * (sigma_s ** K[11])
        f += val * V
    return f


def joint_cmom_to(n, par):
    """Conditional joint central moments to order (n, n)"""
    return [[poly2num(joint_cmom(i, j), par) for j in range(n + 1)] for i in range(n + 1)]


if __name__ == '__main__':
    from pprint import pprint

    subdir = Path('ieii_ieziziz')
    if not subdir.exists():
        print('prepare the ieii_ieziziz moments first, it will take a while...')
        combs = ieii_ieziziz_comb_to(10)
        pprint(combs)
        moments = moment_ieii_ieziziz_comb(combs)
        print(f'There are {len(combs)} items in total.')
    poly = moment_ieii_ieziziz_csv(0, 0, 0, 0, 0, 2)
    print(poly)

    print(r'Conditional joint moment E[\bar{v}_t^1 \bar{y}_t^2|v_0]: ')
    print(joint_cmom(1, 2))

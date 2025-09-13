r"""
Conditional Joint Moment

Conditional joint moment of volatility and return,
i.e., :math:`\mathbb{E}[v_t^n y_t^m|v_0]`.
"""
import math
from fractions import Fraction
from pathlib import Path

from ajdmom import Poly
from ajdmom.mdl_svcj.cond_joint_cmom import joint_cmom


def cond_mean_v():
    """Conditional mean of volatility

    :return: poly with ``keyfor`` =
     ('e^{kt}', 't', 'k^{-}',
     'mu', 'v0-theta', 'theta', 'sigma', 'rho',
     'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s')
    :rtype: Poly
    """
    kf = ['e^{kt}', 't', 'k^{-}',
          'mu', 'v0-theta', 'theta', 'sigma', 'rho',
          'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s']
    pol1 = Poly()
    pol1.set_keyfor(kf)
    k1 = (-1, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 0, 0, 0)
    k2 = (0, 0, 0,
          0, 0, 1, 0, 0,
          0, 0, 0, 0, 0)
    v1, v2 = Fraction(1, 1), Fraction(1, 1)
    pol1.add_keyval(k1, v1)
    pol1.add_keyval(k2, v2)
    return pol1


def cond_mean_y():
    """Conditional mean of return

    :return: poly with ``keyfor`` =
     ('e^{kt}', 't', 'k^{-}',
     'mu', 'v0-theta', 'theta', 'sigma', 'rho',
     'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s')
    :rtype: Poly
    """
    kf = ['e^{kt}', 't', 'k^{-}',
          'mu', 'v0-theta', 'theta', 'sigma', 'rho',
          'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s']
    pol2 = Poly()
    pol2.set_keyfor(kf)
    k1 = (0, 1, 0,
          1, 0, 0, 0, 0,
          0, 0, 0, 0, 0)
    k2 = (0, 1, 0,
          0, 0, 1, 0, 0,
          0, 0, 0, 0, 0)
    v1, v2 = Fraction(1, 1), Fraction(-1, 2)
    k3 = (0, 0, 1,
          0, 1, 0, 0, 0,
          0, 0, 0, 0, 0)
    k4 = (-1, 0, 1,
          0, 1, 0, 0, 0,
          0, 0, 0, 0, 0)
    v3, v4 = Fraction(-1, 2), Fraction(1, 2)
    pol2.add_keyval(k1, v1)
    pol2.add_keyval(k2, v2)
    pol2.add_keyval(k3, v3)
    pol2.add_keyval(k4, v4)
    return pol2


def joint_mom(n, m):
    """Conditional joint moment of volatility and return

    :param int n: power of volatility.
    :param int m: power of return.
    :return: Poly object with ``keyfor`` =
     ('e^{kt}', 't', 'k^{-}',
     'mu', 'v0-theta', 'theta', 'sigma', 'rho',
     'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s')
    :rtype: Poly
    """
    kf = ['e^{kt}', 't', 'k^{-}',
          'mu', 'v0-theta', 'theta', 'sigma', 'rho',
          'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s']
    poly = Poly()
    poly.set_keyfor(kf)
    mean_y = cond_mean_y()
    mean_v = cond_mean_v()
    for i in range(n + 1):
        for j in range(m + 1):
            bino = math.comb(n, i) * math.comb(m, j)
            pol1 = mean_v ** (n - i)
            pol2 = mean_y ** (m - j)
            pol3 = joint_cmom(i, j)
            #  ('e^{kt}', 't', 'k^{-}',
            # 'v0-theta', 'theta', 'sigma', 'rho',
            # 'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s')
            poln = Poly.multiply(pol1 * pol2, pol3)
            poly.merge(bino * poln)
    return poly


def poly2num(poly, par):
    """Evaluate poly to numerical value"""
    # kf = ['e^{kt}', 't', 'k^{-}',
    #       'mu', 'v0-theta', 'theta', 'sigma', 'rho',
    #       'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s']
    k = par['k']
    mu, v0, theta, sigma, rho = par['mu'], par['v0'], par['theta'], par['sigma'], par['rho']
    lmbd, mu_v, rhoJ, mu_s, sigma_s = par['lmbd'], par['mu_v'], par['rhoJ'], par['mu_s'], par['sigma_s']
    h = par['h']
    #
    f = 0
    for K, V in poly.items():
        val = math.exp(K[0] * k * h) * (h ** K[1]) / (k ** K[2])
        val *= (mu ** K[3]) * ((v0 - theta) ** K[4]) * (theta ** K[5])
        val *= (sigma ** K[6]) * (rho ** K[7])
        val *= (lmbd ** K[8]) * (mu_v ** K[9]) * (rhoJ ** K[10])
        val *= (mu_s ** K[11]) * (sigma_s ** K[12])
        f += val * V
    return f


def joint_mom_to(n, par):
    """Conditional joint moments to order (n, n)"""
    return [[poly2num(joint_mom(i, j), par) for j in range(n + 1)] for i in range(n + 1)]


if __name__ == '__main__':
    from pprint import pprint
    from ajdmom.mdl_svcj.cond_joint_cmom import ieii_ieziziz_comb_to, moment_ieii_ieziziz_comb

    v0 = 0.007569
    k = 3.46
    theta = 0.008
    sigma = 0.14
    rho = -0.82
    r = 0.0319
    lmbd = 0.47
    mu_b = -0.1
    sigma_s = 0.0001
    mu_v = 0.05
    rhoJ = -0.38

    mu = r - lmbd * mu_b
    mu_s = math.log((1 + mu_b) * (1 - rhoJ * mu_v)) - sigma_s ** 2 / 2
    h = 1

    par = {'v0': v0, 'mu': mu, 'k': k, 'theta': theta, 'sigma': sigma, 'rho': rho,
           'lmbd': lmbd, 'mu_v': mu_v, 'rhoJ': rhoJ, 'mu_s': mu_s, 'sigma_s': sigma_s,
           'h': h}

    subdir = Path('ieii_ieziziz')
    if not subdir.exists():
        print('prepare the ieii_ieziziz moments first, it will take a while...')
        combs = ieii_ieziziz_comb_to(10)
        pprint(combs)
        moments = moment_ieii_ieziziz_comb(combs)
        print(f'There are {len(combs)} items in total.')

    jmoms = joint_mom_to(4, par)
    for i in range(4 + 1):
        print(jmoms[i])

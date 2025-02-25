r"""
Moments of the SVCJ model
"""
import math
from fractions import Fraction

from ajdmom.poly import Poly
from ajdmom.mdl_svcj.cond_mom import moment_y as cond_moment_y
from ajdmom.mdl_srjd.cmom import cmoment_v

def moment_v_theta(n):
    """moment of :math:`E[(v0-theta)^n]`

    :param int n: order of the moment
    :return: poly with ``keyfor`` = ('k^{-}', 'E[v]', 'sigma', 'lmbd', 'mu_v')
    :rtype: Poly
    """
    kf = ['k^{-}', 'E[v]', 'sigma', 'lmbd', 'mu_v']
    #
    # v0 - theta = (v0 - E[v]) + lmbd * mu_v / k
    # E[(v0 - theta)^n] = \sum_{i=0}^n \binom{n}{i} (lmbd * mu_v / k)^i E[(v0 - E[v])^{n-i}]
    #
    poly = Poly()
    poly.set_keyfor(kf)
    for i in range(n + 1):
        bino = math.comb(n, i)
        uncmom = cmoment_v(n - i)
        # 'k^{-}', 'E[v]', 'sigma', 'lmbd', 'mu_v'
        for k, v in uncmom.items():
            key = list(k)
            key[0] += i
            key[3] += i
            key[4] += i
            key = tuple(key)
            poly.add_keyval(key, v * bino)
    return poly

def key_times_poly(coef, k, poly):
    # k: ['e^{kt}', 't', 'k^{-}',
    #     'mu', 'v0-theta', 'theta', 'sigma', 'rho',
    #            substituted
    #     'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s']
    #
    # poly: ['k^{-}', 'E[v]', 'sigma', 'lmbd', 'mu_v']
    #
    kf = ['e^{kt}', 't', 'k^{-}',
          'mu', 'E[v]', 'theta', 'sigma', 'rho',
          'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s']
    #
    poln = Poly()
    poln.set_keyfor(kf)
    for K, V in poly.items():
        key = (k[0],         # e^{kt}
               k[1],         # t
               k[2] + K[0],  # k^{-}
               k[3],         # mu
                      K[1],  # E[v]
               k[5],         # theta
               k[6] + K[2],  # sigma
               k[7],         # rho
               k[8] + K[3],  # lmbd
               k[9] + K[4],  # mu_v
               k[10],        # rhoJ
               k[11],        # mu_s
               k[12])        # sigma_s
        poln.add_keyval(key, coef * V)
    return poln

def moment_y(n):
    """moment of :math:`y_t`

    :param int n: order of the moment
    :return: poly with ``keyfor`` = ('e^{kt}', 't', 'k^{-}',
      'mu', 'E[v]', 'theta', 'sigma', 'rho',
      'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s')
    :rtype: Poly
    """
    poly = cond_moment_y(n)
    # kf = ['e^{kt}', 't', 'k^{-}',
    #       'mu', 'v0-theta', 'theta', 'sigma', 'rho',
    #       'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s']
    kf = ['e^{kt}', 't', 'k^{-}',
          'mu', 'E[v]', 'theta', 'sigma', 'rho',
          'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s']
    poln = Poly()
    poln.set_keyfor(kf)
    #
    for k, v in poly.items():
        pol1 = moment_v_theta(k[4])  # ['k^{-}', 'E[v]', 'sigma', 'lmbd', 'mu_v']
        pol2 = key_times_poly(v, k, pol1)
        poln.merge(pol2)
    return poln

def poly2num(poly, par):
    # kf = ['e^{kt}', 't', 'k^{-}',
    #       'mu', 'E[v]', 'theta', 'sigma', 'rho',
    #       'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s']
    k = par['k']
    mu, theta, sigma, rho = par['mu'], par['theta'], par['sigma'], par['rho']
    lmbd, mu_v, rhoJ, mu_s, sigma_s = par['lmbd'], par['mu_v'], par['rhoJ'], par['mu_s'], par['sigma_s']
    h = par['h']
    #
    mean_v = theta + lmbd * mu_v / k
    #
    f = 0
    for K, V in poly.items():
        val  = math.exp(K[0]*k*h) * (h ** K[1]) / (k ** K[2])
        val *= (mu ** K[3]) * (mean_v ** K[4]) * (theta ** K[5])
        val *= (sigma ** K[6]) * (rho ** K[7])
        val *= (lmbd ** K[8]) * (mu_v ** K[9]) * (rhoJ ** K[10])
        val *= (mu_s ** K[11]) * (sigma_s ** K[12])
        f += val * V
    return f

def unm_y(n, par):
    moment = moment_y(n)
    value = poly2num(moment, par)
    return value


if __name__ == '__main__':
    from pprint import pprint

    n = 4
    moment = moment_y(n)
    print(f"moment_y({n}) =")
    pprint(moment)
    print(f"{len(moment)} rows of {moment.keyfor}")

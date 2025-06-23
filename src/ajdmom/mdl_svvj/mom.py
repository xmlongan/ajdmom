"""
Moments of the SVVJ model
"""
import math

from ajdmom import Poly
from ajdmom.mdl_svvj.cond_mom import moment_y as cond_moment_y
from ajdmom.mdl_svcj.mom import moment_v_theta

def key_times_poly(coef, k, poly):
    # k: ['e^{kt}', 't', 'k^{-}',
    #     'mu', 'v0-theta', 'theta', 'sigma', 'rho',
    #            substituted
    #     'lmbd', 'mu_v']
    #
    # poly: ['k^{-}', 'E[v]', 'sigma', 'lmbd', 'mu_v']
    #
    kf = ['e^{kt}', 't', 'k^{-}',
          'mu', 'E[v]', 'theta', 'sigma', 'rho',
          'lmbd', 'mu_v']
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
               k[9] + K[4]   # mu_v
               )
        poln.add_keyval(key, coef * V)
    return poln

def moment_y(n):
    """moment of :math:`y_t`

    :param int n: order of the moment
    :return: poly with ``keyfor`` = ('e^{kt}', 't', 'k^{-}',
      'mu', 'E[v]', 'theta', 'sigma', 'rho',
      'lmbd', 'mu_v')
    :rtype: Poly
    """
    poly = cond_moment_y(n)
    # kf = ['e^{kt}', 't', 'k^{-}',
    #       'mu', 'v0-theta', 'theta', 'sigma', 'rho',
    #       'lmbd', 'mu_v']
    kf = ['e^{kt}', 't', 'k^{-}',
          'mu', 'E[v]', 'theta', 'sigma', 'rho',
          'lmbd', 'mu_v']
    poln = Poly()
    poln.set_keyfor(kf)
    #
    for k, v in poly.items():
        pol1 = moment_v_theta(k[4])  # ['k^{-}', 'E[v]', 'sigma', 'lmbd', 'mu_v']
        pol2 = key_times_poly(v, k, pol1)
        poln.merge(pol2)
    return poln

def expand_Ev(poly):
    # kf = ['e^{kt}', 't', 'k^{-}', 'mu', 'E[v]', 'theta', 'sigma', 'rho', 'lmbd', 'mu_v']
    kf = ['e^{kt}', 't', 'k^{-}', 'mu',           'theta', 'sigma', 'rho', 'lmbd', 'mu_v']
    poln = Poly()
    poln.set_keyfor(kf)
    for k, v in poly.items():
        for i in range(k[4] + 1):
            j = k[4] - i
            bino = math.comb(k[4], i)
            key = k[0:2] + (k[2]+j, k[3]) + k[5:8] + (k[8]+j, k[9]+j)
            val = bino * v
            poln.add_keyval(key, val)
    return poln

def poly2num(poly, par):
    # kf = ['e^{kt}', 't', 'k^{-}',
    #       'mu', 'E[v]', 'theta', 'sigma', 'rho',
    #       'lmbd', 'mu_v']
    k = par['k']
    mu, theta, sigma, rho = par['mu'], par['theta'], par['sigma'], par['rho']
    lmbd, mu_v = par['lmbd'], par['mu_v']
    h = par['h']
    #
    mean_v = theta + lmbd * mu_v / k
    #
    f = 0
    for K, V in poly.items():
        val  = math.exp(K[0]*k*h) * (h ** K[1]) / (k ** K[2])
        val *= (mu ** K[3]) * (mean_v ** K[4]) * (theta ** K[5])
        val *= (sigma ** K[6]) * (rho ** K[7])
        val *= (lmbd ** K[8]) * (mu_v ** K[9])
        f += val * V
    return f

def unm_y(n, par):
    moment = moment_y(n)
    value = poly2num(moment, par)
    return value

if __name__ == '__main__':
    from pprint import pprint

    n = 2
    moment = moment_y(n)
    moment = expand_Ev(moment)
    print(f"moment_y({n}) =")
    pprint(moment)
    print(f"{len(moment)} rows of {moment.keyfor}")

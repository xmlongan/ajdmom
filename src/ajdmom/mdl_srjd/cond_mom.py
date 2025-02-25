"""
Conditional Moments (I)

Conditional moments of the SRJD model, given the initial variance.
"""
import math
from ajdmom import Poly
from ajdmom.mdl_srjd.cond_ie_iez_mom import moment_ie_iez

def moment_x(n):
    r"""conditional moment of :math:`\sigma IE_t + IEZ_t`

    Note x is defined as :math:`x_t \equiv e^{kt}(v_t - \theta) - (v_0 - \theta)`,
    which equals to :math:`\sigma IE_t + IEZ_t`.

    :param integer n: order of the conditional moment
    :return: poly with ``keyfor`` = ('e^{kt}', 'k^{-}', 'v0-theta', 'theta',
      'sigma', 'lmbd', 'mu_v')
    :rtype: Poly
    """
    kf = ['e^{kt}', 'k^{-}', 'v0-theta', 'theta', 'sigma', 'lmbd', 'mu_v']
    poly = Poly()
    poly.set_keyfor(kf)
    for i in range(n+1):
        bino = math.comb(n, i)
        poln = moment_ie_iez(i, n-i)
        for k, v in poln.items():
            key = list(k)
            key[4] += i
            key = tuple(key)
            val = bino * v
            poly.add_keyval(key, val)
    poly.remove_zero()
    return poly

def moment_v(n):
    """conditional moment of the state :math:`v_t`

    :param integer n: order of the conditional moment
    :return: poly with ``keyfor`` = ('e^{kt}', 'k^{-}', 'v0-theta', 'theta',
      'sigma', 'lmbd', 'mu_v')
    :rtype: Poly
    """
    kf = ['e^{kt}', 'k^{-}', 'v0-theta', 'theta', 'sigma', 'lmbd', 'mu_v']
    poly = Poly()
    poly.set_keyfor(kf)
    for i in range(n+1):
        bino1 = math.comb(n, i)
        for j in range(n-i+1):
            bino2 = math.comb(n - i, j)
            poln = moment_x(n - i - j)
            for k, v in poln.items():
                key = list(k)
                key[0] -= n - j
                key[2] += i
                key[3] += j
                key = tuple(key)
                val = bino1 * bino2 * v
                poly.add_keyval(key, val)
    poly.remove_zero()
    return poly

def poly2num(poly, par):
    v0, k, theta = par['v0'], par['k'], par['theta']
    sigma, lmbd, mu_v = par['sigma'], par['lmbd'], par['mu_v']
    h = par['h']
    f = 0
    for K, v in poly.items():
        val = math.exp(K[0] * k * h) / (k ** K[1])
        val *= ((v0 - theta) ** K[2]) * (theta ** K[3])
        val *= (sigma ** K[4]) * (lmbd ** K[5]) * (mu_v ** K[6])
        #
        f += val * v  # v is a Fraction, produce a floating-point number
    return f

def m_x(n, par):
    moment = moment_x(n)
    value = poly2num(moment, par)
    return value

def m(n, par):
    moment = moment_v(n)
    value = poly2num(moment, par)
    return value


if __name__ == '__main__':
    from pprint import pprint
    n = 3
    # print(f"\nmoment_x({n}) = "); pprint(moment_x(n))
    print(f"\nmoment_v({n}) = "); pprint(moment_v(n))
    #
    v0, h = 0.5, 1  # failed
    k, theta, sigma = 0.1, 0.5, 0.05
    lmbd, mu_v = 1, 0.3
    par = {'v0': v0, 'h': h, 'k': k, 'theta': theta,
           'sigma': sigma, 'mu_v': mu_v, 'lmbd': lmbd}
    print(f"\nm({n}) = {m(n, par)}")

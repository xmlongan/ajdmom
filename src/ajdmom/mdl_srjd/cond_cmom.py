"""
Conditional Central Moments (I)

Conditional central moments of the SRJD model, given the initial variance.
"""
import math
from ajdmom import Poly
from ajdmom.mdl_srjd.cond_mom import moment_x, poly2num

def cmoment_x(n):
    r"""conditional central moment of :math:`\sigma IE_t + IEZ_t`

    Note x is defined as :math:`x_t \equiv e^{kt}(v_t - \theta) - (v_0 - \theta)`,
    which equals to :math:`\sigma IE_t + IEZ_t`.

    :param integer n: order of the conditional central moment
    :return: poly with ``keyfor`` = ('e^{kt}', 'k^{-}', 'v0-theta', 'theta',
      'sigma', 'lmbd', 'mu_v')
    :rtype: Poly
    """
    kf = ['e^{kt}', 'k^{-}', 'v0-theta', 'theta', 'sigma', 'lmbd', 'mu_v']
    poly = Poly()
    poly.set_keyfor(kf)
    for i in range(n+1):
        bino1 = math.comb(n, i)
        poln = moment_x(i)
        for j in range(n - i + 1):
            bino2 = math.comb(n - i, j)
            for k, v in poln.items():
                key = list(k)
                key[0] += j
                key[1] += n - i
                key[5] += n - i
                key[6] += n - i
                key = tuple(key)
                val = bino1 * bino2 * v * ((-1) ** j)
                poly.add_keyval(key, val)
    poly.remove_zero()
    return poly

def cmoment_v(n):
    """conditional central moment of the state :math:`v_t`

    :param integer n: order of the conditional central moment
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
            poln = moment_x(i)
            for k, v in poln.items():
                key = list(k)
                key[0] -= i + j
                key[1] += n - i
                key[5] += n - i
                key[6] += n - i
                key = tuple(key)
                val = bino1 * bino2 * ((-1) ** (n-i-j)) * v
                poly.add_keyval(key, val)
    poly.remove_zero()
    return poly

def cm_x(n, par):
    cmoment = cmoment_x(n)
    value = poly2num(cmoment, par)
    return value

def cm(n, par):
    cmoment = cmoment_v(n)
    value = poly2num(cmoment, par)
    return value


if __name__ == '__main__':
    from pprint import pprint
    n = 4
    # print(f"\ncmoment_x({n}) = "); pprint(cmoment_x(n))
    print(f"\ncmoment_v({n}) = "); pprint(cmoment_v(n))
    #
    v0, h = 0.5, 1  # failed
    k, theta, sigma = 0.1, 0.5, 0.05
    lmbd, mu_v = 1, 0.3
    par = {'v0': v0, 'h': h, 'k': k, 'theta': theta,
           'sigma': sigma, 'mu_v': mu_v, 'lmbd': lmbd}
    print(f"\ncm({n}) = {cm(n, par)}")

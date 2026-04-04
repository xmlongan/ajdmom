"""
Conditional Moments (I)

Conditional moments of the SRJD model, given the initial variance.
"""
import math
from ajdmom import Poly
from ajdmom.mdl_srjd.cond_ie_iez_mom import (moment_ie_iez,
                                             moment_ie_iez_to,
                                             poly2num)

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

def moment_x_to(m):
    IE_IEZ = moment_ie_iez_to(m)
    kf = ['e^{kt}', 'k^{-}', 'v0-theta', 'theta', 'sigma', 'lmbd', 'mu_v']
    X = []
    for n in range(m + 1):
        poly = Poly()
        poly.set_keyfor(kf)
        for i in range(n + 1):
            bino = math.comb(n, i)
            poln = IE_IEZ[(i, n - i)]
            for k, v in poln.items():
                key = list(k)
                key[4] += i
                key = tuple(key)
                val = bino * v
                poly.add_keyval(key, val)
        poly.remove_zero()
        X.append(poly)
    return X

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

def moment_v_to(m):
    r'''Moments of :math:`v_t` to order m

    :math:`v_t = e^{-kt}[(v_0-\theta) + x_t] + \theta`,
    where :math:`x_t = \sigma_vI\!E_t + I\!E\!Z_t`.

    :param m: the highest order
    :return: a list of moments
    '''
    X = moment_x_to(m)
    kf = ['e^{kt}', 'k^{-}', 'v0-theta', 'theta', 'sigma', 'lmbd', 'mu_v']
    V = []
    for n in range(m + 1):
        poly = Poly()
        poly.set_keyfor(kf)
        for i in range(n + 1):
            bino1 = math.comb(n, i)
            for j in range(n - i + 1):
                bino2 = math.comb(n - i, j)
                for k, v in X[n - i - j].items():
                    key = list(k)
                    key[0] -= n - j  # e^{-kt}
                    key[2] += i      # v0-theta
                    key[3] += j      # theta
                    key = tuple(key)
                    val = bino1 * bino2 * v
                    poly.add_keyval(key, val)
        poly.remove_zero()
        V.append(poly)
    return V

def m_x(n, par):
    moment = moment_x(n)
    value = poly2num(moment, par)
    return value

def m(n, par):
    moment = moment_v(n)
    value = poly2num(moment, par)
    return value

def m_to(m, par):
    moments = moment_v_to(m)
    values = [poly2num(moment, par) for moment in moments]
    return values


if __name__ == '__main__':
    from pprint import pprint
    import time
    n = 3
    # print(f"\nmoment_x({n}) = "); pprint(moment_x(n))
    # print(f"\nmoment_v({n}) = "); pprint(moment_v(n))
    #
    # v0, h = 0.5, 1  # failed
    # k, theta, sigma = 0.1, 0.5, 0.05
    # lmbd, mu_v = 1, 0.3

    v0, k, theta, sigma = 0.007569, 3.46, 0.008, 0.14
    lmbd, mu_v, h = 0.47, 0.05, 1.0

    par = {'v0': v0, 'h': h, 'k': k, 'theta': theta,
           'sigma': sigma, 'mu_v': mu_v, 'lmbd': lmbd}
    # print(f"\nm({n}) = {m(n, par)}")

    # moms = [m(n, par) for n in range(9)]
    # print(f'moms = {moms}')

    # V = moment_v_to(8)
    start_time = time.perf_counter()
    moms = m_to(8, par)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    print(f'moms = {moms}')

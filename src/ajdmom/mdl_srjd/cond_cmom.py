"""
Conditional Central Moments (I)

Conditional central moments of the SRJD model, given the initial variance.
"""
import math
from ajdmom import Poly
from ajdmom.mdl_srjd.cond_ie_iez_cmom import (cmoment_ie_iez,
                                              cmoment_ie_iez_to,
                                              poly2num)

def cmoment_x(n):
    r"""conditional central moment of :math:`\sigma IE_t + IEZ_t`

    Note x is defined as :math:`x_t \equiv e^{kt}(v_t - \theta) - (v_0 - \theta)`,
    which equals to :math:`\sigma IE_t + IEZ_t`.

    We have :math:`E[(x_t - E[x_t])^n|v_0] = E[(\sigma IE_t + \overline{IEZ}_t)^n|v_0]`

    :param integer n: order of the conditional central moment
    :return: poly with ``keyfor`` = ('e^{kt}', 'k^{-}', 'v0-E[v]', 'E[v]',
      'sigma', 'lmbd', 'mu_v')
    :rtype: Poly
    """
    kf = ['e^{kt}', 'k^{-}', 'v0-E[v]', 'E[v]', 'sigma', 'lmbd', 'mu_v']
    poly = Poly()
    poly.set_keyfor(kf)
    for i in range(n + 1):
        bino = math.comb(n, i)
        for K, V in cmoment_ie_iez(i, n - i).items():
            key = list(K)
            key[4] += i
            key = tuple(key)
            val = bino * V
            poly.add_keyval(key, val)
    poly.remove_zero()
    return poly

def cmoment_v(n):
    """conditional central moment of the state :math:`v_t`

    :param integer n: order of the conditional central moment
    :return: poly with ``keyfor`` = ('e^{kt}', 'k^{-}', 'v0-E[v]', 'E[v]',
      'sigma', 'lmbd', 'mu_v')
    :rtype: Poly
    """
    kf = ['e^{kt}', 'k^{-}', 'v0-E[v]', 'E[v]', 'sigma', 'lmbd', 'mu_v']
    poly = Poly()
    poly.set_keyfor(kf)
    for i in range(n + 1):
        for k, v in cmoment_x(i).items():
            key = list(k)
            key[0] -= n  # times e^{-nkt}
            key[2] += n - i
            key = tuple(key)
            v *= math.comb(n, i)
            poly.add_keyval(key, v)
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

def cmoment_x_to(m):
    IE_IEZ = cmoment_ie_iez_to(m)
    kf = ['e^{kt}', 'k^{-}', 'v0-E[v]', 'E[v]', 'sigma', 'lmbd', 'mu_v']
    cX = []
    for n in range(m + 1):
        poly = Poly()
        poly.set_keyfor(kf)
        for i in range(n+1):
            bino = math.comb(n, i)
            for K, V in IE_IEZ[(i, n - i)].items():
                key = list(K)
                key[4] += i
                key = tuple(key)
                val = bino * V
                poly.add_keyval(key, val)
        poly.remove_zero()
        cX.append(poly)
    return cX

def cmoment_v_to(m):
    cX = cmoment_x_to(m)
    kf = ['e^{kt}', 'k^{-}', 'v0-E[v]', 'E[v]', 'sigma', 'lmbd', 'mu_v']
    cV = []  # (v_t - E[v])
    for n in range(m + 1):
        poly = Poly()
        poly.set_keyfor(kf)
        for i in range(n + 1):
            for k, v in cX[i].items():
                key = list(k)
                key[0] -= n  # times e^{-nkt}
                key[2] += n - i
                key = tuple(key)
                v *= math.comb(n, i)
                poly.add_keyval(key, v)
        poly.remove_zero()
        cV.append(poly)
    return cV

def cm_to(m, par):
    cmoments = cmoment_v_to(m)
    values = [poly2num(cmom, par) for cmom in cmoments]
    return values

def moment_v_to(m):
    cV = cmoment_v_to(m)
    kf = ['e^{kt}', 'k^{-}', 'v0-E[v]', 'E[v]', 'sigma', 'lmbd', 'mu_v']
    V = []  # v_t = (v_t - E[v]) + E[v]
    for n in range(m + 1):
        poly = Poly()
        poly.set_keyfor(kf)
        for i in range(n + 1):
            poln = Poly()
            poln.set_keyfor(kf)
            for k, v in cV[i].items():
                key = list(k)
                key[3] += n - i
                key = tuple(key)
                v *= math.comb(n, i)
                poln.add_keyval(key, v)
            poly += poln
        poly.remove_zero()
        V.append(poly)
    return V

def m_to(n, par):
    moments = moment_v_to(n)
    values = [poly2num(moment, par) for moment in moments]
    return values


if __name__ == '__main__':
    from pprint import pprint
    import time
    n = 4
    # print(f"\ncmoment_x({n}) = "); pprint(cmoment_x(n))
    # print(f"\ncmoment_v({n}) = "); pprint(cmoment_v(n))
    #
    # v0, h = 0.5, 1  # failed
    # k, theta, sigma = 0.1, 0.5, 0.05
    # lmbd, mu_v = 1, 0.3

    v0, k, theta, sigma = 0.007569, 3.46, 0.008, 0.14
    lmbd, mu_v, h = 0.47, 0.05, 1.0

    par = {'v0': v0, 'h': h, 'k': k, 'theta': theta,
           'sigma': sigma, 'mu_v': mu_v, 'lmbd': lmbd}
    # print(f"\ncm({n}) = {cm(n, par)}")

    start_time = time.perf_counter()
    moms = m_to(8, par)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    print(f'moms = {moms}')

    # the following running speed is much slower
    # from ajdmom.mdl_srjd.cond_mom import m_to as m_to_2
    #
    # start_time = time.perf_counter()
    # moms = m_to_2(18, par)
    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # print(f"Elapsed time: {elapsed_time:.6f} seconds")
    # print(f'moms = {moms}')

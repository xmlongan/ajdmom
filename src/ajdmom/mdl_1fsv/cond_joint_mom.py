r"""
Conditional Joint Moment

Conditional joint moment of volatility and return,
i.e., :math:`\mathbb{E}[v_t^n y_t^m|v_0]`.
"""
import math
from fractions import Fraction

from ajdmom import Poly
from ajdmom.mdl_1fsv.cond_joint_cmom import joint_cmom


def cond_mean_y() -> Poly:
    """Conditional mean of return

    :return: poly with ``keyfor`` =
     ('mu', 'e^{kh}', 'h', 'v0', 'k^{-}', 'theta', 'sigma', 'rho')
    :rtype: Poly
    """
    kf = ['mu', 'e^{kh}', 'h', 'v0', 'k^{-}', 'theta', 'sigma', 'rho']
    pol1 = Poly()
    pol1.set_keyfor(kf)
    k1 = (1, 0, 1, 0, 0, 0, 0, 0)
    k2 = (0, 0, 1, 0, 0, 1, 0, 0)
    v1, v2 = Fraction(1, 1), Fraction(-1, 2)
    k3 = (0, 0, 0, 0, 1, 1, 0, 0)
    k4 = (0, -1, 0, 0, 1, 1, 0, 0)
    v3, v4 = Fraction(1, 2), Fraction(-1, 2)
    k5 = (0, 0, 0, 1, 1, 0, 0, 0)
    k6 = (0, -1, 0, 1, 1, 0, 0, 0)
    v5, v6 = Fraction(-1, 2), Fraction(1, 2)
    pol1.add_keyval(k1, v1)
    pol1.add_keyval(k2, v2)
    pol1.add_keyval(k3, v3)
    pol1.add_keyval(k4, v4)
    pol1.add_keyval(k5, v5)
    pol1.add_keyval(k6, v6)
    return pol1


def cond_mean_v() -> Poly:
    """Conditional mean of volatility

    :return: poly with ``keyfor`` =
     ('mu', 'e^{kh}', 'h', 'v0', 'k^{-}', 'theta', 'sigma', 'rho')
    :rtype: Poly
    """
    kf = ['mu', 'e^{kh}', 'h', 'v0', 'k^{-}', 'theta', 'sigma', 'rho']
    pol2 = Poly()
    pol2.set_keyfor(kf)
    k1 = (0, -1, 0, 1, 0, 0, 0, 0)
    v1 = Fraction(1, 1)
    k2 = (0, 0, 0, 0, 0, 1, 0, 0)
    k3 = (0, -1, 0, 0, 0, 1, 0, 0)
    pol2.add_keyval(k1, v1)
    pol2.add_keyval(k2, v1)
    pol2.add_keyval(k3, -v1)
    return pol2


def joint_mom(n: int, m: int) -> Poly:
    """Conditional joint moment of volatility and return

    :param int n: power of volatility.
    :param int m: power of return.
    :return: Poly object with ``keyfor`` =
     ('mu', 'e^{kh}', 'h', 'v0', 'k^{-}', 'theta', 'sigma', 'rho')
    :rtype: Poly
    """
    kf = ['mu', 'e^{kh}', 'h', 'v0', 'k^{-}', 'theta', 'sigma', 'rho']
    poly = Poly()
    poly.set_keyfor(kf)
    mean_y = cond_mean_y()
    mean_v = cond_mean_v()
    for i in range(n + 1):
        for j in range(m + 1):
            bino = math.comb(n, i) * math.comb(m, j)
            pol1 = mean_v ** (n - i)  # ('mu', 'e^{kh}', 'h', 'v0', 'k^{-}', 'theta', 'sigma', 'rho')
            pol2 = mean_y ** (m - j)  # ('mu', 'e^{kh}', 'h', 'v0', 'k^{-}', 'theta', 'sigma', 'rho')
            pol3 = joint_cmom(i, j)  # ('e^{kh}', 'h', 'v0', 'k^{-}', 'theta', 'sigma', 'rho')
            poln = Poly.multiply(pol1 * pol2, pol3)
            poly.merge(bino * poln)
    return poly


def poly2num(poly, par):
    """Evaluate poly to numerical value"""
    h, v0, k, theta, sigma, rho = par['h'], par['v0'], par['k'], par['theta'], par['sigma'], par['rho']
    mu = par['mu']
    f = 0
    for K, v in poly.items():
        val = mu ** K[0]
        val *= math.exp(K[1] * k * h) * (h ** K[2]) * (v0 ** K[3]) * (k ** (-K[4]))
        val *= (theta ** K[5]) * (sigma ** K[6]) * (rho ** K[7])
        f += val * v
    return f


def joint_mom_to(n, par):
    """Conditional joint moments to order (n, n)"""
    return [[poly2num(joint_mom(i, j), par) for j in range(n + 1)] for i in range(n + 1)]


if __name__ == '__main__':
    from pprint import pprint

    n, m = 2, 2
    jmom = joint_mom(n, m)
    pprint(jmom)

    par = {'h': 1, 'v0': 0.010201, 'k': 6.21, 'theta': 0.019, 'sigma': 0.61, 'rho': -0.7, 'mu': 0.0319}
    jmoms = joint_mom_to(3, par)
    for i in range(3 + 1):
        print(jmoms[i])

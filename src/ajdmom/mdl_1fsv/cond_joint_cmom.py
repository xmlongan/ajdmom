r"""
Conditional Joint Central Moment

Conditional joint central moment of volatility and return,
i.e., :math:`\mathbb{E}[\bar{v}_t^n \bar{y}_t^m|v_0]`.
The derivation reduces to
:math:`\mathbb{E}[I\!E_t^{m_1+n} I_t^{m_2} I_t^{*m_3}|v_0]`.
"""
import math
from fractions import Fraction

from ajdmom import Poly
from ajdmom.ito_mom import moment_IEII
from ajdmom.utils import comb, simplify_rho


def b_mn(m1: int, m2: int, m3: int, n: int) -> Poly:
    r"""Coefficient for this combination (m1, m2, m3, n)

    :param int m1: power of :math:`I\!E`
    :param int m2: power of :math:`I`
    :param int m3: power of :math:`I^{*}`
    :param int n: power of :math:`I\!E`, from :math:`v_t`
    :return: Poly object with ``keyfor`` =
      ('e^{kh}', 'h', 'v0', 'k^{-}', 'theta', 'sigma', 'rho', 'sqrt(1-rho^2)')
    :rtype: Poly
    """
    kf = ['e^{kh}', 'h', 'v0', 'k^{-}', 'theta', 'sigma', 'rho', 'sqrt(1-rho^2)']
    poly = Poly()
    poly.set_keyfor(kf)
    for i in range(m2 + 1):
        key = (-(m1 + n), 0, 0, m1 + i, 0, m1 + i + n, m2 - i, m3)
        val = math.comb(m2, i) * ((-1) ** i) * Fraction(1, 2 ** (m1 + i))
        poly.add_keyval(key, val)
    return poly


def align(poly: Poly, keyfor: list) -> Poly:
    """Align poly to keyfor

    :param Poly poly: Poly object with ``keyfor`` =
     ('e^{0h}','e^{kh}', 'h', 'v0', 'k^{-}', 'theta', 'sigma')
    :param list keyfor: the list
     ('e^{kh}', 'h', 'v0', 'k^{-}', 'theta', 'sigma', 'rho', 'sqrt(1-rho^2)')
    :return: Poly object with ``keyfor`` = keyfor
    :rtype: Poly
    """
    poln = Poly()
    poln.set_keyfor(keyfor)
    for k, v in poly.items():
        key = k[1:] + (0, 0)
        poln.add_keyval(key, v)
    return poln


def joint_cmom(n: int, m: int) -> Poly:
    """Conditional joint central moment of volatility and return

    :param int n: power of :math:`\bar{v}_t`
    :param int m: power of :math:`\bar{y}_t`
    :return: Poly object with ``keyfor`` =
     ('e^{kh}', 'h', 'v0', 'k^{-}', 'theta', 'sigma', 'rho')
    :rtype: Poly
    """
    kf = ['e^{kh}', 'h', 'v0', 'k^{-}', 'theta', 'sigma', 'rho', 'sqrt(1-rho^2)']
    poly = Poly()
    poly.set_keyfor(kf)
    for m1 in range(m + 1):
        for m2 in range(m - m1 + 1):
            m3 = m - m1 - m2
            bino = comb(m, [m1, m2, m3])
            b = b_mn(m1, m2, m3, n)
            # kf = ['e^{kh}', 'h', 'v0', 'k^{-}', 'theta', 'sigma', 'rho', 'sqrt(1-rho^2)']
            poln = moment_IEII(m1 + n, m2, m3)
            # kf = ['e^{0h}','e^{kh}', 'h', 'v0', 'k^{-}', 'theta', 'sigma']
            poln = align(poln, kf)
            poln = bino * (b * poln)
            poly.merge(poln)
    return simplify_rho(poly, len(kf) - 1)


def poly2num(poly, par):
    """Evaluate poly to numerical value"""
    h, v0, k, theta, sigma, rho = par['h'], par['v0'], par['k'], par['theta'], par['sigma'], par['rho']
    f = 0
    for K, v in poly.items():
        val = math.exp(K[0] * k * h) * (h ** K[1]) * (v0 ** K[2]) * (k ** (-K[3]))
        val *= (theta ** K[4]) * (sigma ** K[5]) * (rho ** K[6])
        f += val * v
    return f


def joint_cmom_to(n, par):
    """Conditional joint moments to order (n, n)"""
    return [[poly2num(joint_cmom(i, j), par) for j in range(n + 1)] for i in range(n + 1)]


if __name__ == '__main__':
    from pprint import pprint

    n, m = 2, 2
    jcmom = joint_cmom(n, m)
    pprint(jcmom)

    par = {'h': 1, 'v0': 0.010201, 'k': 6.21, 'theta': 0.019, 'sigma': 0.61, 'rho': -0.7, 'mu': 0.0319}
    jcmoms = joint_cmom_to(3, par)
    for i in range(3 + 1):
        print(jcmoms[i])

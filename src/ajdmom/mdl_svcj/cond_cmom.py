"""
Conditional Central Moments of the SVCJ model, given initial variance
"""
import math
from fractions import Fraction

from ajdmom import Poly
from ajdmom.mdl_svcj.cond_mom import moment_y, poly2num

def cmoment_y(n):
    r"""conditional central moment of :math:`\overline{y}_t|v_0`

    :param integer n: order of the conditional central moment
    :return: poly with ``keyfor`` = ('e^{kt}', 't', 'k^{-}',
      'mu', 'v0-theta', 'theta', 'sigma', 'rho',
      'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s')
    :rtype: Poly
    """
    mean_y = moment_y(1)
    kf = ['e^{kt}', 't', 'k^{-}',
          'mu', 'v0-theta', 'theta', 'sigma', 'rho',
          'lmbd', 'mu_v', 'rhoJ', 'mu_s', 'sigma_s']
    poly = Poly()
    poly.set_keyfor(kf)
    if n < 0:
        raise ValueError(f'cmoment_y({n}) is called')
    p1 = Poly({(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): Fraction(1, 1)})
    p1.set_keyfor(kf)
    # special case
    if n == 0:
        return p1
    # typical cases
    for i in range(n + 1):
        bino = math.comb(n, i)
        sign = (-1) ** i
        pol1 = p1 if i == 0 else mean_y ** i
        poln = (bino * sign) * (pol1 * moment_y(n - i))
        poly.merge(poln)
    return poly

def cm_y(n, par):
    cmoment = cmoment_y(n)
    value = poly2num(cmoment, par)
    return value

if __name__ == '__main__':
    from pprint import pprint

    # n = 1
    # poly = cmoment_y(n)
    # print(f"cmoment_y({n}) = "); pprint(poly); print(f"which is a poly with keyfor = {poly.keyfor}")

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

    for n in range(1, 4 + 1):
        print(f"cmoment({n}) = {cm_y(n, par)}")

"""
Conditional Moment of the SVVJ model, given initial variance
"""
import math
from fractions import Fraction

from ajdmom import Poly
from ajdmom.mdl_svvj.cond_cmom import cmoment_y, poly2num


def moment_y(n):
    """conditional moment of the SVVJ model :math:`y_t|v_0`

    :param integer n: order of the conditional moment
    :return: poly with ``keyfor`` =  ('e^{kt}', 't', 'k^{-}',
      'mu', 'v0-theta', 'theta', 'sigma', 'rho',
      'lmbd', 'mu_v')
    :rtype: Poly
    """
    kf = ['e^{kt}', 't', 'k^{-}',
          'mu', 'v0-theta', 'theta', 'sigma', 'rho',
          'lmbd', 'mu_v']
    poly = Poly()
    poly.set_keyfor(kf)
    if n < 0:
        raise ValueError('n must be non-negative')
    p1 = Poly({(0, 0, 0, 0, 0, 0, 0, 0, 0, 0): Fraction(1, 1)})
    p1.set_keyfor(kf)
    # special case
    if n == 0:
        return p1
    # E[y|v0] = (mu - theta/2)*t - (v0 - theta)*(1-e^{-kt})/(2k)
    #           + lmbd * mu_v (1-e^{-kt})/(2k^2) - lmbd t mu_v / (2k)
    mean_y = Poly({
        (0, 1, 0,   1, 0, 0, 0, 0,   0, 0): Fraction(1, 1),
        (0, 1, 0,   0, 0, 1, 0, 0,   0, 0): Fraction(-1, 2),
        (0, 0, 1,   0, 1, 0, 0, 0,   0, 0): Fraction(-1, 2),
        (-1,0, 1,   0, 1, 0, 0, 0,   0, 0): Fraction(1, 2),
        (0, 0, 2,   0, 0, 0, 0, 0,   1, 1): Fraction(1, 2),
        (-1,0, 2,   0, 0, 0, 0, 0,   1, 1): Fraction(-1, 2),
        (0, 1, 1,   0, 0, 0, 0, 0,   1, 1): Fraction(-1, 2)
    })
    mean_y.set_keyfor(kf)
    if n == 1:
        return mean_y
    # typical cases
    # y_t = mean_y + \overline{y}_t
    for i in range(n + 1):
        bino = math.comb(n, i)
        pol1 = p1 if i == 0 else mean_y ** i
        poln = bino * (pol1 * cmoment_y(n - i))
        poly.merge(poln)
    return poly

def m_y(n, par):
    moment = moment_y(n)
    value = poly2num(moment, par)
    return value


if __name__ == '__main__':
    from pprint import pprint

    n = 3
    poly = moment_y(n)
    print(f"moment_y({n}) = "); pprint(poly); print(f"which is a poly with keyfor = {poly.keyfor}")

    v0 = 0.007569
    k = 3.46
    theta = 0.008
    sigma = 0.14
    rho = -0.82
    r = 0.0319
    lmbd = 0.47
    mu_v = 0.05

    mu = r
    h = 1

    par = {'v0': v0, 'mu': mu, 'k': k, 'theta': theta, 'sigma': sigma, 'rho': rho,
           'lmbd': lmbd, 'mu_v': mu_v,
           'h': h}

    for n in range(1, 4+1):
        print(f"moment({n}) = {m_y(n, par)}")

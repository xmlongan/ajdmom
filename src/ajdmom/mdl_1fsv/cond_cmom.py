r"""
Conditional Central Moment
----------------------------

Conditional central moments of the Heston SV model, given the initial variance.

Note that the keyfor attribute is different from that in the module
:py:mod:`ajdmom.mdl_1fsv.cmom`, which now is
('e^{kt}', 't', 'k^{-}', 'v_0-theta', 'theta', 'sigma', 'sigma/2k',
'rho-sigma/2k', 'sqrt(1-rho^2)')

.. math::

   y_{n-1,t} = (\mu - \theta/2)[t-(n-1)h] - \beta_{n-1,t}(v_{n-1}-\theta)
    + \bar{y}_{n-1,t}

where :math:`\beta_{n-1,t} = (1-e^{-k[t-(n-1)h]})/(2k)` and the centralized term

.. math::

   \bar{y}_{n-1,t}
   = \frac{\sigma_v}{2k}e^{-kt}I\!E_{n-1,t} +
   \left( \rho - \frac{\sigma_v}{2k} \right)I_{n-1,t} +
   \sqrt{1-\rho^2}I_{n-1,t}^{*}.

The module implements the derivation of conditional moments of the centralized return

.. math::

   \mathbb{E}[\bar{y}_{n-1,t}^m|v_{n-1}]
   = \sum_{m_1+m_2+m_3=m} c(\boldsymbol{m}) b(\boldsymbol{m})
   (e^{-kt}I\!E_{n-1,t})^{m_1}(I_{n-1,t})^{m_2}(I_{n-1,t}^{*})^{m_3}

with :math:`c(\boldsymbol{m}) = C_m^{m_1}C_{m-m_1}^{m_2}`,
:math:`b(\boldsymbol{m}) =
\left(\frac{\sigma_v}{2k}\right)^{m_1}
\left(\rho - \frac{\sigma_v}{2k}\right)^{m_2}
\left(\sqrt{1-\rho^2}\right)^{m_3}`
and
:math:`\boldsymbol{m} = (m_1,m_2,m_3)`.

"""
import math
from fractions import Fraction as Frac

from ajdmom.utils import comb, simplify_rho
from ajdmom.poly import Poly
from ajdmom.ito_mom import int_et


def int_mIEII(c, tp, m, poly):
    r"""Integral of :math:`c \times tp \times \int_{(n-1)h}^t e^{mks} poly ds`

    :param int c: coefficient to multiply with
    :param int tp: type of the multiplication,

       +----+--------------------------------------+
       | tp | multiply with                        |
       +====+======================================+
       | 1  | :math:`e^{k(n-1)h}(v_{n-1}-\theta)`  |
       +----+--------------------------------------+
       | 2  | :math:`\theta`                       |
       +----+--------------------------------------+
       | 3  | :math:`\sigma_v`                     |
       +----+--------------------------------------+

    :param int m: power of :math:`e^{ks}` in the integrand
    :param Poly poly: a Poly object, such as
       :math:`\mathbb{E}[I\!E_t^{m_1} I_t^{m_2} I_t^{*m_3}]`, with attribute
       ``keyfor`` = ('e^{kt}', 't', 'k^{-}', 'v_0-theta', 'theta', 'sigma').
    :return: poly with the same ``keyfor`` as that in the input poly
    :rtype: Poly
    """
    poln = Poly()
    poln.set_keyfor(poly.keyfor) # poly = IEII[(m1, m2, m3)]
    for k1 in poly:  # ('e^{kt}', 't', 'k^{-}', 'v_0-theta', 'theta', 'sigma')
        poly_sub = int_et(m + k1[0], k1[1])
        for k2 in poly_sub:  # ('e^{kt}', 't', 'k^{-}')
            key = [k2[0], k2[1], k2[2] + k1[2], k1[3], k1[4], k1[5]]
            if tp == 1:
                key[3] += 1
            elif tp == 2:
                key[4] += 1
            else:
                key[5] += 1
            val = c * poly[k1] * poly_sub[k2]
            poln.add_keyval(tuple(key), val)
    return poln


def recursive_IEII(n3, n4, n5, IEII):
    r"""Recursive step in equation :eq:`ito-moment`

    :param int n3: :math:`n_3`
       in :math:`\mathbb{E}[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]`.
    :param int n4: :math:`n_4`
       in :math:`\mathbb{E}[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]`.
    :param int n5: :math:`n_5`
       in :math:`\mathbb{E}[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]`.
    :param dict IEII: a dict with key (n3,n4,n5) and value Poly object with attribute
       ``keyfor`` = ('e^{kt}', 't', 'k^{-}', 'v_0-theta', 'theta', 'sigma')
    :return: updated IEII.
    :rtype: dict
    """
    kf = ['e^{kt}', 't', 'k^{-}', 'v_0-theta', 'theta', 'sigma']
    poly = Poly()
    poly.set_keyfor(kf)
    #
    if n3 >= 2 and n4 >= 0 and n5 >= 0:
        c = Frac(n3 * (n3 - 1), 2)
        poly.merge(int_mIEII(c, 1, 1, IEII[(n3 - 2, n4, n5)]))
        poly.merge(int_mIEII(c, 2, 2, IEII[(n3 - 2, n4, n5)]))
        poly.merge(int_mIEII(c, 3, 1, IEII[(n3 - 1, n4, n5)]))
    if n3 >= 0 and n4 >= 2 and n5 >= 0:
        c = Frac(n4 * (n4 - 1), 2)
        poly.merge(int_mIEII(c, 1, -1, IEII[(n3, n4 - 2, n5)]))
        poly.merge(int_mIEII(c, 2, 0, IEII[(n3, n4 - 2, n5)]))
        poly.merge(int_mIEII(c, 3, -1, IEII[(n3 + 1, n4 - 2, n5)]))
    if n3 >= 1 and n4 >= 1 and n5 >= 0:
        c = Frac(n3 * n4, 1)
        poly.merge(int_mIEII(c, 1, 0, IEII[(n3 - 1, n4 - 1, n5)]))
        poly.merge(int_mIEII(c, 2, 1, IEII[(n3 - 1, n4 - 1, n5)]))
        poly.merge(int_mIEII(c, 3, 0, IEII[(n3, n4 - 1, n5)]))
    if n3 >= 0 and n4 >= 0 and n5 >= 2:
        c = Frac(n5 * (n5 - 1), 2)
        poly.merge(int_mIEII(c, 1, -1, IEII[(n3, n4, n5 - 2)]))
        poly.merge(int_mIEII(c, 2, 0, IEII[(n3, n4, n5 - 2)]))
        poly.merge(int_mIEII(c, 3, -1, IEII[(n3 + 1, n4, n5 - 2)]))
    return poly


def simplify(poly, tp=1):
    """Simplify polynomials differently

    :param Poly poly:  Poly with ``keyfor`` =
      ('e^{kt}', 't', 'k^{-}', 'v_0-theta', 'theta', 'sigma',
      'sigma/2k', 'rho-sigma/2k', 'sqrt(1-rho^2)')
    :param int tp: type of the simplification.
     If ``tp=1`` (default), change 'e^{kt}' to 'e^{-kt}'.
     If ``tp=2``, change to
     ('e^{-kt}', 't', 'k^{-}', 'v_0-theta', 'theta', 'sigma', 'rho').
     If ``tp=3``, change to
     ('e^{-kt}', 't', 'k^{-}', 'v_0', 'theta', 'sigma', 'rho').
    :return: simplified poly.
    :rtype: Poly
    """
    if tp not in [1, 2, 3]: raise Exception("tp must be 1, 2, or 3")
    #
    kf = ('e^{-kt}',) + poly.keyfor[1:]
    poln = Poly()
    poln.set_keyfor(kf)
    #
    if tp == 1:  # change 'e^{kt}' to 'e^{-kt}'
        for k, v in poly.items():
            poln.add_keyval((-k[0],) + k[1:], v)
        return poln
    #
    if tp >= 2:  # change 'sigma/2k', 'rho-sigma/2k', 'sqrt(1-rho^2)' to 'rho'
        kf = ['e^{-kt}', 't', 'k^{-}', 'v_0-theta', 'theta', 'sigma', 'rho',
        'sqrt(1-rho^2)']
        poln.set_keyfor(kf)
        for k, v in poly.items():
            for i in range(0, k[7] + 1):  # expand 'rho-sigma/2k'
                p = k[6] + i
                key = (-k[0], k[1], k[2] + p, k[3], k[4], k[5] + p, k[7] - i, k[8])
                val = math.comb(k[7], i) * ((-1) ** i) * Frac(1, 2 ** p) * v
                poln.add_keyval(key, val)
        poln.remove_zero()
        poln = simplify_rho(poln, 7)
    #
    if tp == 3:
        # kf = ['e^{-kt}', 't', 'k^{-}', 'v_0-theta', 'theta', 'sigma', 'rho']
        # to
        # kf = ['e^{-kt}', 't', 'k^{-}', 'v_0', 'theta', 'sigma', 'rho']
        kf = ['e^{-kt}', 't', 'k^{-}', 'v_0', 'theta', 'sigma', 'rho']
        poly = Poly()
        poly.set_keyfor(kf)
        #
        for k, v in poln.items():
            for i in range(0, k[3] + 1):
                key = (k[0], k[1], k[2], k[3] - i, k[4] + i, k[5], k[6])
                val = v * math.comb(k[3], i) * ((-1) ** i)
                poly.add_keyval(key, val)
        poln = poly
    return poln


def cmoments_y_to(l, show=False):
    """Conditional central moments from order 1 to :math:`l`.

    Derive the conditional central moments and write to csv files.

    :param int l: highest order of the conditional central moments to derive.
    :param bool show: show the in-process message or not, defaults to False
    :return: a list of the conditional central moments with ``keyfor`` =
     ('e^{kt}', 't', 'k^{-}', 'v_0-theta', 'theta', 'sigma',
     'sigma/2k', 'rho-sigma/2k', 'sqrt(1-rho^2)')
    :rtype: list
    """
    # IEII: a dict of moments of E[IE^{n3}I^{n4}I^{*n5}]
    IEII = {}
    kf = ['e^{kt}', 't', 'k^{-}', 'v_0-theta', 'theta', 'sigma']
    #
    P1 = Poly({(0, 0, 0, 0, 0, 0): Frac(1, 1)})
    P1.set_keyfor(kf)
    P0 = Poly({(0, 0, 0, 0, 0, 0): Frac(0, 1)})
    P0.set_keyfor(kf)
    # n3 + n4 + n5 = 0
    # support for special case
    IEII[(0, 0, 0)] = P1
    # n3 + n4 + n5 = 1
    IEII[(1, 0, 0)] = P0
    IEII[(0, 1, 0)] = P0
    IEII[(0, 0, 1)] = P0
    # n3 + n4 + n5 = 2
    poly = Poly({(1, 0, 1, 1, 0, 0): Frac(1, 1),
                 (0, 0, 1, 1, 0, 0): -Frac(1, 1),
                 (2, 0, 1, 0, 1, 0): Frac(1, 2),
                 (0, 0, 1, 0, 1, 0): -Frac(1, 2)})
    poly.set_keyfor(kf)
    IEII[(2, 0, 0)] = poly
    poly = Poly({(0, 1, 0, 1, 0, 0): Frac(1, 1),
                 (1, 0, 1, 0, 1, 0): Frac(1, 1),
                 (0, 0, 1, 0, 1, 0): -Frac(1, 1)})
    poly.set_keyfor(kf)
    IEII[(1, 1, 0)] = poly
    IEII[(1, 0, 1)] = P0
    poly = Poly({(-1, 0, 1, 1, 0, 0): -Frac(1, 1),
                 (0, 0, 1, 1, 0, 0): Frac(1, 1),
                 (0, 1, 0, 0, 1, 0): Frac(1, 1)})
    poly.set_keyfor(kf)
    IEII[(0, 2, 0)] = poly
    IEII[(0, 1, 1)] = P0
    IEII[(0, 0, 2)] = poly
    #
    kf = kf + ['sigma/2k', 'rho-sigma/2k', 'sqrt(1-rho^2)']
    #
    cmoms = []
    #
    # special case: 0-th central moment
    #
    P1 = Poly({(0, 0, 0, 0, 0, 0, 0, 0, 0): Frac(1, 1)})
    P1.set_keyfor(kf)
    cmoms.append(P1)  # equiv to constant 1
    #
    # special case: 1-th central moment
    #
    P0 = Poly({(0, 0, 0, 0, 0, 0, 0, 0, 0): Frac(0, 1)})
    P0.set_keyfor(kf)
    cmoms.append(P0)  # equiv to constant 0
    #
    # typical cases
    #
    for n in range(2, l + 1):
        # container for the n-th conditional central moment of y
        poly = Poly()
        poly.set_keyfor(kf)
        # iter over all combinations of i + j + (n-i-j) = n
        for i in range(n, -1, -1):
            for j in range(n - i, -1, -1):
                # derive for this exact combination
                c = comb(n, [i, j, n - i - j])
                if n < 3:
                    poln = IEII[(i, j, n - i - j)]  # already in IEII
                else:
                    poln = recursive_IEII(i, j, n - i - j, IEII)
                # add this exact combination into poly
                # c * b * poln
                for k in poln:
                    # key = list(k)
                    # key[0] -= i
                    # key = tuple(key)
                    # key = key + (i, j, n - i - j)
                    key = (k[0]-i,) + k[1:] + (i, j, n - i - j)  # enlarge with last 3
                    poly.add_keyval(key, c * poln[k])
                # reserve poln for further use or delete it otherwise
                if n >= 3 and n < l:
                    IEII[(i, j, n - i - j)] = poln
                else:
                    del poln
        # delete polys when no more needed
        if n == l:
            del IEII
        else:  # only reserve the most recent polys
            index = [key for key in IEII if key[0] + key[1] + key[2] == n - 2]
            for key in index: del IEII[key]
        poly.remove_zero()
        cmoms.append(poly)
        if show: print(f"complete the derivation for the {n}-th conditional central moment.")
    return cmoms


def poly2num(poly, par):
    """Decode poly back to scalar

    :param Poly poly: poly to be decoded with attribute ``keyfor`` =
     ('e^{-kt}', 't', 'k^{-}', 'v_0-theta', 'theta', 'sigma', 'rho').
    :param dict par: parameters in dict.
    :return: scalar of the poly.
    :rtype: float
    """
    v0 = par['v0']
    k = par['k']
    h = par['h']
    theta = par['theta']
    sigma = par['sigma_v']
    rho = par['rho']
    #
    value = 0
    for K in poly:
        val = poly[K] * math.exp(-K[0] * k * h) * (h ** K[1]) * (k ** (-K[2]))
        val *= ((v0 - theta) ** K[3]) * (theta ** K[4]) * (sigma ** K[5])
        val *= rho ** K[6]
        value += val
    return value

def cond_cm(l, par):
    """Conditional central moment in scalar

    :param int l: order of the conditional central moment.
    :param dict par: parameters in dict.
    :return: scalar of the conditional central moment.
    :rtype: float
    """
    cmoments = cmoments_y_to(l)
    cmoment = cmoments[-1]
    cmoment = simplify(cmoment, tp=2)
    # ('e^{-kt}', 't', 'k^{-}', 'v_0-theta', 'theta', 'sigma', 'rho')
    value = poly2num(cmoment, par)
    return value

if __name__ == "__main__":
    import sys
    from pprint import pprint

    print('\nExample usage of the module function\n')
    args = sys.argv[1:]
    l = 3 if len(args) == 0 else int(args[0])
    tp = 2 if len(args) <= 1 else int(args[1])
    cmoms = cmoments_y_to(l, show=True)  # 0-th to l-th conditional central moments
    kf = cmoms[0].keyfor
    print(f"\ncmoments_y_to(l) returns a list of polys with keyfor = \n{kf}")
    # for i in range(1, l+1):
    #   cmom = cmoms[i]
    #   cmom = simplify(cmom, tp)
    #   cmom.write_to_csv(f"tp{tp}-heston-cond-cmoment-{i}-formula.csv")
    print(f"\ncmoment_y({l}) = ")
    pprint(cmoms[l])
    print(f"which is a poly with keyfor = \n{kf}")

"""
Conditional moments (II)

Conditional moments of the SRJD model, given the initial variance
and the already realized jump time points and jump sizes of the CPP
"""
import math
from fractions import Fraction as Frac

from ajdmom.poly import Poly


def int_e_poly(i, poly):
    r"""integral of :math:`\int_0^t e^{iks} poly ds`

    :param int i: power of :math:`e^{ks}`
    :param Poly poly: poly attribute ``keyfor`` =
      ('v_0', 'k^{-}', 'theta', 'sigma', 'e^{kt}', 'l_{1:n}', 'o_{2:n}')
    :return: poly with the same attribute ``keyfor`` of the input poly
    :rtype: Poly
    """
    poln = Poly()
    kf = ['v_0', 'k^{-}', 'theta', 'sigma', 'e^{kt}', 'l_{1:n}', 'o_{2:n}']
    poln.set_keyfor(kf)
    #
    for k in poly:
        # same 1st part of the integration
        key = (k[0], k[1] + 1, k[2], k[3], k[4] + i, k[5], k[6])
        val = poly[k] * Frac(1, k[4] + i)
        poln.add_keyval(key, val)
        # diff 2nd part of the integration
        if len(k[5]) == 0:
            # no summation  inside: - e^{(k[4]+i)*k*0}
            key = list(key)
            key[4] = 0
            key = tuple(key)
        elif len(k[5]) == 1:
            # 1  summation  inside: - e^{(k[4]+i)*k*s_i}
            k5 = (k[5][0] + (k[4] + i),)
            key = list(key)
            key[4] = 0
            key[5] = k5
            key = tuple(key)
        else:
            # 2  summations inside: - e^{(k[4]+i)*k*(s_i1 v ... v s_in)}
            k6 = list(k[6])
            k6[-1] += k[4] + i
            k6 = tuple(k6)
            key = list(key)
            key[4] = 0
            key[6] = k6
            key = tuple(key)
        #
        poln.add_keyval(key, -val)
    return poln


def int_e_IE(i, m, IE):
    r"""Integral of :math:`\int_0^t e^{iks} I\!E_s^m ds`

    :param int i: power of :math:`e^{ks}`.
    :param int m: power of :math:`IE_s`.
    :param dict IE: dict with key m and value Poly object with attribute ``keyfor`` =
      ('v_0', 'k^{-}', 'theta', 'sigma', 'e^{kt}', 'l_{1:n}', 'o_{2:n}')
    :return: poly with the same attribute ``keyfor`` of the Poly object in IE
    :rtype: Poly
    """
    return int_e_poly(i, IE[m])


def int_e_IEZ_IE(m, IE):
    r"""Integral of :math:`\int_0^t e^{ks} I\!E\!Z_s I\!E_s^m ds`

    :param int m: power of :math:`I\!E_s`
    :param dict IE: dict with key m and value Poly object with attribute ``keyfor`` =
      ('v_0', 'k^{-}', 'theta', 'sigma', 'e^{kt}', 'l_{1:n}', 'o_{2:n}')
    :return: poly with the same attribute ``keyfor`` of the Poly object in IE
    :rtype: Poly
    """
    b = IE[m]
    poly = Poly()
    kf = ['v_0', 'k^{-}', 'theta', 'sigma', 'e^{kt}', 'l_{1:n}', 'o_{2:n}']
    poly.set_keyfor(kf)
    #
    # IEZ times IE_s^m
    #
    for k in b:
        k5 = k[5] + (1,)
        if len(k[5]) == 0:  # no summation inside
            k6 = ()
        else:  # 1  summation inside at least
            k6 = k[6] + (0,)
        key = (k[0], k[1], k[2], k[3], k[4], k5, k6)
        poly.add_keyval(key, b[k])
    #
    # \int_0^t e^{ks} IEZ_s IE_s^m ds
    #
    return int_e_poly(1, poly)


def coef_poly(coef, poly, tp):
    """Multiply poly with different type coefficients

    :param Fraction coef: coefficient.
    :param Poly poly: poly with attribute ``keyfor`` =
      ('v_0', 'k^{-}', 'theta', 'sigma', 'e^{kt}', 'l_{1:n}', 'o_{2:n}')
    :param int tp: type of the multiplication,

       +----+----------------------------+
       | tp | multiply with              |
       +====+============================+
       | 1  | :math:`v_0-\\theta`         |
       +----+----------------------------+
       | 2  | :math:`\\theta`             |
       +----+----------------------------+
       | 3  | :math:`1`                  |
       +----+----------------------------+
       | 4  | :math:`\\sigma_v`           |
       +----+----------------------------+

    :return: poly with the same attribute ``keyfor`` of input poly
    :rtype: Poly
    """
    poln = Poly()
    kf = ['v_0', 'k^{-}', 'theta', 'sigma', 'e^{kt}', 'l_{1:n}', 'o_{2:n}']
    poln.set_keyfor(kf)
    #
    if tp == 1:
        for k in poly:
            key = (k[0] + 1, k[1], k[2], k[3], k[4], k[5], k[6])
            val = coef * poly[k]
            poln.add_keyval(key, val)
            #
            key = (k[0], k[1], k[2] + 1, k[3], k[4], k[5], k[6])
            poln.add_keyval(key, -val)
    if tp == 2:
        for k in poly:
            key = (k[0], k[1], k[2] + 1, k[3], k[4], k[5], k[6])
            val = coef * poly[k]
            poln.add_keyval(key, val)
    if tp == 3:
        for k in poly:
            key = k
            val = coef * poly[k]
            poln.add_keyval(key, val)
    if tp == 4:
        for k in poly:
            key = (k[0], k[1], k[2], k[3] + 1, k[4], k[5], k[6])
            val = coef * poly[k]
            poln.add_keyval(key, val)
    return poln


def recursive_IE(m, IE):
    r"""Recursive step in equation :eq:`srjd-IE-moment`

    :param int m: power of :math:`I\!E_s` in the integrand.
    :param dict IE: a dict of conditional moments of :math:`I\!E_t`,
      with key ``m`` and value Poly object with attribute ``keyfor`` =
      ('v_0', 'k^{-}', 'theta', 'sigma', 'e^{kt}', 'l_{1:n}', 'o_{2:n}')
    :return: poly with the same attribute ``keyfor`` of the Poly object in ``IE``
    :rtype: Poly
    """
    poly = Poly()
    kf = ['v_0', 'k^{-}', 'theta', 'sigma', 'e^{kt}', 'l_{1:n}', 'o_{2:n}']
    poly.set_keyfor(kf)
    #
    c = Frac(m * (m - 1), 2)
    poly.merge(coef_poly(c, int_e_IE(1, m - 2, IE), 1))
    poly.merge(coef_poly(c, int_e_IE(2, m - 2, IE), 2))
    poly.merge(coef_poly(c, int_e_IEZ_IE(m - 2, IE), 3))
    poly.merge(coef_poly(c, int_e_IE(1, m - 1, IE), 4))
    return poly


def moment_IE(m):
    r""":math:`\mathbb{E}[I\!E_t^m|v_0, z(s), 0\le s\le t]`

    :param int m: order of the conditional moment
    :return: poly if return_all=False else IE, a dict of conditional moments
      of :math:`I\!E_t`, with key ``m`` and value Poly object with attribute
      ``keyfor`` =
      ('v_0', 'k^{-}', 'theta', 'sigma', 'e^{kt}', 'l_{1:n}', 'o_{2:n}')
    :rtype: Poly or dict of Poly
    """
    # IE： dict of E[IE^{m}]
    IE = {}
    #
    kf = ['v_0', 'k^{-}', 'theta', 'sigma', 'e^{kt}', 'l_{1:n}', 'o_{2:n}']
    #
    P1 = Poly({(0, 0, 0, 0, 0, (), ()): Frac(1, 1)})  # equiv to 1
    P1.set_keyfor(kf)
    P0 = Poly({(0, 0, 0, 0, 0, (), ()): Frac(0, 1)})  # equiv to 0
    P0.set_keyfor(kf)
    # m = 0
    # support for special case
    IE[0] = P1
    # m = 1
    IE[1] = P0
    # m = 2
    poly = Poly({
        (0, 1, 1, 0, 2, (), ()): Frac(1, 2),
        (1, 1, 0, 0, 1, (), ()): Frac(1, 1),
        (0, 1, 1, 0, 1, (), ()): -Frac(1, 1),
        (0, 1, 1, 0, 0, (), ()): Frac(1, 2),
        (1, 1, 0, 0, 0, (), ()): -Frac(1, 1),
        (0, 1, 0, 0, 1, (1,), ()): Frac(1, 1),
        (0, 1, 0, 0, 0, (2,), ()): -Frac(1, 1)
    })
    poly.set_keyfor(kf)
    IE[2] = poly
    #
    if m <= 2:
        return IE[m]
    #
    if m > 3:
        # compute all lower-order moments to get ready
        for n in range(3, m):
            poly = recursive_IE(n, IE)
            poly.remove_zero()
            IE[n] = poly
            # delete polys no more needed
            index = [key for key in IE if key == n - 2]
            for key in index: del IE[key]
    # the last one
    poly = recursive_IE(m, IE)
    poly.remove_zero()
    return poly


def moment_v(m):
    r"""conditional moment :math:`\mathbb{E}[v_t^m|v_0, z(s), 0\le s\le t]`

    :param int m: order of the conditional moments of :math:`v(t)`
    :return: poly with attribute ``keyfor`` =
      ('v_0', 'k^{-}', 'theta', 'sigma', 'e^{kt}', 'l_{1:n}', 'o_{2:n}')
    :rtype: Poly
    """
    kf = ['v_0', 'k^{-}', 'theta', 'sigma', 'e^{kt}', 'l_{1:n}', 'o_{2:n}']
    poly = Poly()
    poly.set_keyfor(kf)
    #
    if m == 0:
        poly.add_keyval((0, 0, 0, 0, 0, (), ()), Frac(1, 1))
        return poly
    if m == 1:
        poly.add_keyval((1, 0, 0, 0, -1, (), ()), Frac(1, 1))
        poly.add_keyval((0, 0, 1, 0, -1, (), ()), -Frac(1, 1))
        poly.add_keyval((0, 0, 1, 0, 0, (), ()), Frac(1, 1))
        poly.add_keyval((0, 0, 0, 0, -1, (1,), ()), Frac(1, 1))
        return poly
    # m >= 2
    for j in range(m + 1):  # be careful of 0:m <- range(m+1)
        C1 = math.comb(m, j)
        # prepare pol1
        pol1 = Poly()
        pol1.set_keyfor(kf)
        for j1 in range(j, -1, -1):
            for j2 in range(j - j1, -1, -1):
                for j3 in range(j - j1 - j2, -1, -1):
                    j4 = j - j1 - j2 - j3
                    # k5 = tuple(1 for i in range(j4))
                    if j4 == 0:
                        k5 = ()
                        k6 = ()
                    else:
                        k5 = tuple(1 for i in range(j4))
                        k6 = tuple(0 for i in range(j4 - 1))  # j4=1 -> k6 = ()
                    key = (j1, 0, j2 + j3, m - j, j3 - m, k5, k6)
                    #
                    C2 = math.comb(j, j1) * math.comb(j - j1, j2) * math.comb(j - j1 - j2, j3)
                    val = C1 * C2 * ((-1) ** j2)
                    pol1.add_keyval(key, val)
        # prepare pol2
        pol2 = moment_IE(m - j)
        # pol1 * pol2
        for k1 in pol1:
            for k2 in pol2:  # works either m-j = 0 or m-j = 1
                if len(k2[5]) == 0:
                    k6 = () + tuple(0 for i in range(len(k1[5]) - 1))
                else:
                    k6 = k2[6] + tuple(0 for i in range(len(k1[5])))
                key = (k2[0] + k1[0], k2[1] + k1[1], k2[2] + k1[2], k2[3] + k1[3],
                       k2[4] + k1[4],
                       k2[5] + k1[5],  # tuple + : k2[5] must be put in front of k1[5]
                       k6)  # k2[6] is encoded according to k2[5]
                val = pol1[k1] * pol2[k2]
                poly.add_keyval(key, val)
    poly.remove_zero()
    return poly


##########
# scalar
##########
def fZ(l, o, k, s, J):
    r"""function :math:`f_{Z_t}(l_{1:n},o_{1:n})` as defined in :eq:`fZ_IE`

    :param tuple l: vector :math:`\boldsymbol{l}`, should be a tuple
    :param tuple o: vector :math:`\boldsymbol{o}`, should be a tuple
    :param float k: parameter :math:`k`
    :param tuple s: jump time points
    :param tuple J: jump sizes
    :return: function value
    :rtype: float
    """
    # 1. Continuous part
    if len(l) == 0: return 1
    #
    # 2. Jump part, len(l) >= 1, may be single|mutiple summation
    # 2.1 no jump occurs
    if len(J) == 0: return 0
    # 2.2 >=1 jumps occur
    val = 0
    stack = []
    nextstep = "forward"  # downward|backward
    while len(stack) < len(l):
        if nextstep == "forward":
            if len(stack) < len(l) - 1:
                stack.append(0)
                nextstep = "forward"
            else:
                # come to the innermost-level summation
                for i in range(len(J)):  # index start with 0
                    stack.append(i)
                    index = stack.copy()
                    stack.pop()
                    # evaluate and add to sum
                    f = 1
                    s_max = s[index[0]]
                    for ii in range(len(index)):
                        j = index[ii]
                        s_j = s[j]
                        f *= math.exp(l[ii] * k * s_j) * J[j]
                        if ii > 0:
                            s_max = s_max if s_max > s_j else s_j
                            f *= math.exp(o[ii - 1] * k * s_max)
                    val += f
                    #
                # checking whether all indexes traversed
                all_N = True
                for i in range(len(l) - 1):
                    if stack[i] != len(J) - 1:
                        all_N = False;
                        break
                if all_N: break
                #
                nextstep = "backward"
        else:  # nextstep = "backward"
            i = stack.pop()
            if i == len(J) - 1:
                nextstep = "backward"
            else:
                stack.append(i + 1)
                nextstep = "forward"
    return val


def poly2num(poly, par):
    """Decode poly back to scalar

    :param Poly poly: poly to be decoded with attribute ``keyfor`` =
      ('v_0', 'k^{-}', 'theta', 'sigma', 'e^{kt}', 'l_{1:n}', 'o_{2:n}')
    :param dict par: parameters in dict, ``jumptime`` and ``jumpsize`` must also be
      included
    :return: scalar of the poly
    :rtype: float
    """
    v0 = par['v0']
    k = par['k']
    theta = par['theta']
    sigma = par['sigma']
    h = par['h']  # t = h
    J = par['jumpsize']  # vector of jump sizes
    s = par['jumptime']  # vector of jump time points
    #
    f = 0
    for K in poly:
        val = (v0 ** K[0]) * (k ** (-K[1])) * (theta ** K[2]) * (sigma ** K[3])
        val *= math.exp(K[4] * k * h)
        #
        l, o = K[5], K[6]
        val *= fZ(l, o, k, s, J)
        #
        f += val * poly[K]
    return f


def m(l, par):
    """conditional moment in scalar

    :param int l: order of the conditional moment
    :param dict par: parameters in dict, ``jumptime`` and ``jumpsize`` must also be
      included
    :return: scalar of the central moment
    :rtype: float
    """
    moment = moment_v(l)
    value = poly2num(moment, par)
    return value


if __name__ == "__main__":
    from pprint import pprint
    import sys

    print('\nExample usage of the module function\n')
    args = sys.argv[1:]
    n = 2 if len(args) == 0 else int(args[0])
    mom = moment_v(n)
    print(f"moment_v({n}) = ")
    pprint(mom)
    print(f"which is a Poly with attribute keyfor = \n{mom.keyfor}")
    #
    m = 1 if len(args) <= 1 else int(args[1])
    moment = moment_IE(m)
    print(f"\nmoment_IE({m}) = ")
    pprint(moment)
    print(f"which is a Poly with attribute keyfor = \n{moment.keyfor}")

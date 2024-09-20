"""
Utilities for combination and normal distribution
"""
import math
from fractions import Fraction as Frac
from ajdmom.poly import Poly


# ----------------------------combination----------------------------------------
def comb(n, ns):
    """number of occurrences for an exact combination

    :param int n: total power, = ``sum(ns)``
    :param list ns: list of powers, :math:`n_1 + \\cdots + n_N = n`
    :return: number of occurrences for this exact combination
    :return: float
    """
    f = 1
    for k in ns:
        f *= math.comb(n, k)
        n -= k
    return f


def multilevel_index(n, m, show=False):
    """get the indexes of multiple summation

    :param int n: levels of the multiple summation, >= 1
    :param int m: number of possibilities in each summation, >= 1
    :param bool show: whether show the process or not, default to ``False``
    :return: list of the multi-level indexes.
    :rtype: list
    """
    index = []
    stack = []
    nextstep = "forward"  # downward|backward
    while len(stack) < n:
        if nextstep == "forward":
            if len(stack) < n - 1:
                stack.append(0)
                nextstep = "forward"
            else:
                for i in range(m):  # adapt to python 0-indexing
                    stack.append(i)
                    index.append(stack.copy())
                    if show: print(stack)
                    stack.pop()
                # checking whether all indexes traversed
                all_m = True
                for i in range(n - 1):  # no need to test the innermost level
                    if stack[i] != m - 1:
                        all_m = False
                        break
                if all_m: break
                nextstep = "backward"
        else:
            i = stack.pop()
            if i == m - 1:
                nextstep = "backward"
            else:
                stack.append(i + 1)
                nextstep = "forward"
    return index


def multilevel_comb(n, m, show=False):
    """get the indexes of multiple level combination

    :param int n: parts to split the number
    :param int m: total number resource
    :param bool show: whether show the process or not, default to ``False``
    :return: list of the multiple level combination indexes
    :return: list
    """
    if n == 1:
        return m
    elif n == 2:
        index = [[i, m - i] for i in range(m)]
        return index
    # N > 2
    index = []
    stack = []
    nextstep = "forward"
    while len(stack) < n:
        if nextstep == "forward":
            if len(stack) < n - 2:
                stack.append(0)
                nextstep = "forward"
            else:
                left = m - sum(stack)
                for i in range(left + 1):
                    stack.append(i)
                    stack.append(left - i)
                    index.append(stack.copy())
                    if show: print(stack)
                    stack.pop()
                    stack.pop()
                # checking whether all indexes traversed
                if stack[0] == m: break
                #
                nextstep = "backward"
        else:
            left = m - sum(stack)
            if left == 0:
                stack.pop()
                nextstep = "backward"
            else:
                stack[-1] += 1
                nextstep = "forward"
    return index


# -----------------------------norm distribution---------------------------------
def dbfactorial(n):
    """double factorial :math:`n!!`

    :param int n: number :math:`n`
    :return: value of :math:`n!! = n(n-2)(n-4)\\cdots 4 \\cdot 2` for
      even :math:`n`, :math:`n!! = n(n-2)(n-4)\\cdots 3 \\cdot 1` for odd :math:`n`
    :rtype: float
    """
    if n == 0 or n == 1:
        return (1)
    elif n == 2:
        return (2)
    #
    f = n
    while n > 2:
        n -= 2
        f *= n
    return f


def cmnorm(n):
    """central moment of Normal distribution

    :param int n: order of the central moment, n >= 0
    :return: poly with attribute ``keyfor`` = ('sigma'), noting that its key
      is still a tuple, though it contains only one element
    :rtype: Poly
    """
    # special case
    if n == 0:
        P1 = Poly({(0,): Frac(1, 1)})
        P1.set_keyfor(['sigma'])
        return P1
    # n >= 1
    poly = Poly()
    poly.set_keyfor(['sigma'])
    if n % 2 != 0:
        poly.add_keyval((0,), Frac(0, 1))
    else:
        poly.add_keyval((n,), Frac(dbfactorial(n - 1), 1))
    return poly


# --------------------------------misc-------------------------------------------
def fZ_each_index(l, o, p, q, k, s, J, index):
    """function value :math:`f_{Z_t}(\\boldsymbol{l},\\boldsymbol{o},
    \\boldsymbol{p}, \\boldsymbol{q})` as defined in :eq:`fZ` for an index

    :param tuple l: vector :math:`\\boldsymbol{l}`, should be a tuple
    :param tuple o: vector :math:`\\boldsymbol{o}`, should be a tuple
    :param tuple p: vector :math:`\\boldsymbol{p}`, should be a tuple
    :param tuple q: vector :math:`\\boldsymbol{q}`, should be a tuple
    :param float k: parameter :math:`k`
    :param tuple s: jump time points
    :param tuple J: jump sizes
    :param list index: an exact index combination
    :return: function value for an exact index combination
    :rtype: float
    """
    f = 1
    s_max = s[index[0]]
    for i in range(len(index)):
        j = index[i]
        s_j = s[j]
        f *= math.exp(l[i] * k * s_j) * J[j] * (s_j ** o[i])
        if i > 0:
            s_max = s_max if s_max > s_j else s_j
            f *= math.exp(p[i - 1] * k * s_max) * (s_max ** q[i - 1])
    return f


def fZ(l, o, p, q, k, s, J):
    """function :math:`f_{Z_t}(\\boldsymbol{l},\\boldsymbol{o},\\boldsymbol{p},
    \\boldsymbol{q})` as defined in :eq:`fZ`

    :param tuple l: vector :math:`\\boldsymbol{l}`, should be a tuple
    :param tuple o: vector :math:`\\boldsymbol{o}`, should be a tuple
    :param tuple p: vector :math:`\\boldsymbol{p}`, should be a tuple
    :param tuple q: vector :math:`\\boldsymbol{q}`, should be a tuple
    :param float k: parameter :math:`k`
    :param tuple s: jump time points
    :param tuple J: jump sizes
    :return: function value
    :rtype: float
    """
    # 1. Continuous part
    if len(l) == 0: return 1
    #
    # 2. Jump part, len(l) >= 1, may be single|multiple summation
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
                        f *= math.exp(l[ii] * k * s_j) * J[j] * (s_j ** o[ii])
                        if ii > 0:
                            s_max = s_max if s_max > s_j else s_j
                            f *= math.exp(p[ii - 1] * k * s_max) * (s_max ** q[ii - 1])
                    val += f
                    #
                # checking whether all indexes traversed
                all_N = True
                for i in range(len(l) - 1):
                    if stack[i] != len(J) - 1:
                        all_N = False
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


def simplify_rho(poly, n):
    """expand the term 'sqrt(1-rho^2)'

    :param Poly poly: the poly contains 'sqrt(1-rho^2)' which must follow 'rho' immediately
    :param int n: index of 'sqrt(1-rho^2)'
    :return: poly without 'sqrt(1-rho^2)'
    :rtype: Poly
    """
    # exclude the 'sqrt(1-rho^2)' in the keyfor attribute
    kf = poly.keyfor[0:n] + poly.keyfor[(n + 1):]
    poln = Poly()
    poln.set_keyfor(kf)
    #
    for k in poly:
        if k[n] == 0:
            key = k[0:n] + k[(n + 1):]
            poln.add_keyval(key, poly[k])
        elif k[n] % 2 == 0:
            # (1-rho^2)^p = \sum_{i=0}^p C_p^i * (-1)^i * 1^{p-i} * rho^{2*i}
            p = k[n] // 2
            for i in range(0, p + 1):
                key = k[0:n] + k[(n + 1):]
                key = list(key)
                key[n - 1] += 2 * i
                key = tuple(key)
                val = poly[k] * math.comb(p, i) * ((-1) ** i)
                poln.add_keyval(key, val)
        else:
            raise Exception("can not reduce the 'sqrt(1-rho^2)' dimension!")
    poln.remove_zero()
    return poln


if __name__ == "__main__":
    from pprint import pprint

    print(f"comb(5, [1, 2, 2) = {comb(5, [1, 2, 2])}")
    print("\nThe index combinations of 3 levels of summation, each 2 possibilities: ")
    multilevel_index(3, 2, show=True)
    print("\nThe combinations of split 4 identical items into 3 parts: ")
    multilevel_comb(3, 4, show=True)
    print("\ncmnorm() returns a poly with keyfor = ('sigma'): ")
    for i in range(7):
        print(f"cmnorm({i}) = ")
        pprint(cmnorm(i))

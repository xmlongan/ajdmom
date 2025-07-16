"""
Conditional central moments (II)

Conditional central moments of the SRJD model, given the initial state
and the realized jumps over the concerned interval.
"""
from ajdmom.poly import Poly
from ajdmom.mdl_srjd.cond2_mom import moment_IE, poly2num


def cmoment_v(m):
    r"""conditional central moment :math:`\mathbb{E}[\overline{v}^m(t)
    |v_0, z(s), 0\le s\le t]`

    :param int m: order of the conditional central moments of :math:`v(t)`
    :return: poly with attribute ``keyfor`` =
      ('v_0', 'k^{-}', 'theta', 'sigma', 'e^{kt}', 'l_{1:n}')
    :rtype: Poly
    """
    poly = Poly()
    kf = ('v_0', 'k^{-}', 'theta', 'sigma', 'e^{kt}', 'l_{1:n}')
    poly.set_keyfor(kf)
    #
    poln = moment_IE(m)
    # ('v_0', 'k^{-}', 'theta', 'sigma', 'e^{kt}', 'l_{1:n}')
    for k in poln:
        key = list(k)
        key[3] += m
        key[4] -= m
        key = tuple(key)
        val = poln[k]
        poly.add_keyval(key, val)
    return poly


##########
# scalar
##########
def cm(l, par):
    """conditional central moment in scalar

    :param int l: order of the conditional central moment
    :param dict par: parameters in dict, ``jumptime`` and ``jumpsize`` must also be
      included
    :return: scalar of the central moment
    :rtype: float
    """
    cmoment = cmoment_v(l)
    value = poly2num(cmoment, par)
    return value


if __name__ == "__main__":
    from pprint import pprint
    import sys

    print('\nExample usage of the module function\n')
    args = sys.argv[1:]
    n = 2 if len(args) == 0 else int(args[0])
    cmom = cmoment_v(n)
    print(f"cmoment_v({n}) = \n")
    pprint(cmom)
    print(f"\nwhich is a Poly with attribute keyfor = \n{cmom.keyfor}")

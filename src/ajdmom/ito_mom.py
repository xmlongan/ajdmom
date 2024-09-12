"""
Itô process moments under Single Square-Root Diffusion Process

The content has also been explained in :doc:`../design` page.

Insights
=========

All :math:`\mathbb{E}[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]`
can be represented as a "Polynomial" of the following form

.. _polynomial-representation:

.. math::

   &\mathbb{E}[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]\\\\
   &= \sum_{n_3,i,j,l,o,p,q} b_{n_3ijlopq} e^{n_3k(n-1)h} e^{ik[t-(n-1)h]}
   [t-(n-1)h]^jv_{n-1}^l k^{-o}\\theta^p\sigma_v^q

where :math:`b_{ijlopq}` is the coefficient.

To facilitate the representation and corresponding operations, I designed
a new *class* :py:class:`~ajdmom.poly.Poly` which is derived from
:class:`~collections.UserDict` in the Python Standard Library
`collections <https://docs.python.org/3/library/collections.html>`_.

Integrals
==========

The essential computation in recursive equation :eq:`ito-moment`
of :doc:`../theory` is that of the following integral

.. math::

   \int_{(n-1)h}^t e^{ik[s-(n-1)h]} [s-(n-1)h]^j ds.


For the indefinite integral, we have

.. math::

   \int e^{nkt} t^m dt =
   \\begin{cases}
   \sum_{i=0}^m c_{nmi} \\frac{1}{k^{i+1}}e^{nkt} t^{m-i}
    & \\text{if } n\\neq 0, m \\neq 0,\\\\
   \\frac{1}{nk}e^{nkt}t^0 & \\text{if } n\\neq 0, m = 0,\\\\
   \\frac{1}{m+1}e^{0kt}t^{m+1} & \\text{if } n = 0, m \\neq 0,\\\\
   e^{0kt}t^1 & \\text{if } n =0 , m=0,
   \end{cases}

where :math:`c_{nm0} \\triangleq \\frac{1}{n}` and

.. math::

   c_{nmi} \\triangleq \\frac{(-1)^{i}}{n^{i+1}} \prod_{j=m-i+1}^{m} j,
   \quad 1\le i \le m.

The coefficient :math:`c_{nmi}` is implemented in function
:py:func:`~ajdmom.ito_mom.c_nmi` which returns a :class:`fractions.Fraction`
instead of a decimal (float number).

For the definite integral,

.. math::

   \int_{(n-1)h}^t e^{ik[s-(n-1)h]}[s-(n-1)h]^jds = F(t-(n-1)h) - F(0)

where :math:`F(t) = \int e^{nkt} t^m dt`. The definite integral is implemented
in :py:func:`~ajdmom.ito_mom.int_et`.


Polynomial Representation
--------------------------

The result of the integral, returned by :py:func:`~ajdmom.ito_mom.int_et`,
is represented as a "polynomial" of the following form

.. math::

   \int_{(n-1)h}^t e^{ik[s-(n-1)h]} [s-(n-1)h]^j ds
    = \sum_{i,j^{'},l}c_{ij^{'}l}e^{ik[t-(n-1)h]}[t-(n-1)h]^{j^{'}}k^{-l}

which is encoded in a :py:class:`~ajdmom.poly.Poly`, derived from
:class:`collections.UserDict`, with
:code:`keyfor = ('e^{k[t-(n-1)h]}','[t-(n-1)h]','k^{-}')`,
``key`` = :math:`(i,j^{'},l)` and ``value`` = :math:`c_{ij^{'}l}`.


Code Design
============

Itô process moment - I
-----------------------

With :math:`\mathbb{E}[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}|v_{n-1}]` represented as
a "polynomial" of the following form

.. math::

   &\mathbb{E}[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]\\\\
   &= \sum_{n_3,i,j,l,o,p,q} b_{n_3ijlopq} e^{n_3k(n-1)h} e^{ik[t-(n-1)h]}
   [t-(n-1)h]^jv_{n-1}^l k^{-o}\\theta^p\sigma_v^q,

consequently, we have

.. math::

   &e^{-kt}\mathbb{E}[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]\\\\
   &= \sum_{n_3,i,j,l,o,p,q} b_{n_3ijlopq} e^{(n_3-1)k(n-1)h}
   e^{(i-1)k[t-(n-1)h]}[t-(n-1)h]^jv_{n-1}^l k^{-o}\\theta^p\sigma_v^q,\\\\
   &e^{kt}\mathbb{E}[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]\\\\
   &= \sum_{n_3,i,j,l,o,p,q} b_{n_3ijlopq} e^{(n_3+1)k(n-1)h}
   e^{(i+1)k[t-(n-1)h]}[t-(n-1)h]^jv_{n-1}^l k^{-o}\\theta^p\sigma_v^q,\\\\
   &e^{2kt}\mathbb{E}[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]\\\\
   &= \sum_{n_3,i,j,l,o,p,q} b_{n_3ijlopq} e^{(n_3+2)k(n-1)h}
   e^{(i+2)k[t-(n-1)h]}[t-(n-1)h]^jv_{n-1}^l k^{-o}\\theta^p\sigma_v^q.

Therefore, it's profitable to consider the following generic integral

.. math::

   &\int_{(n-1)h}^t e^{mks}\mathbb{E}[I\!E_{n-1,s}^{n_3}I_{n-1,s}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]ds\\\\
   &= \sum_{n_3,i,j,l,o,p,q} b_{n_3ijlopq} e^{(n_3+m)k(n-1)h}
   \cdot int\_et(i+m,j)\cdot v_{n-1}^l k^{-o}\\theta^p\sigma_v^q\\\\
   &= \sum_{n_3+m,i+m,j^{'},l,o^{'},p,q} b_{(n_3+m)(i+m)j^{'}l o^{'}pq}
   e^{(n_3+m)k(n-1)h} e^{(i+m)k[t-(n-1)h]} [t-(n-1)h]^{j^{'}}\\\\
   &\qquad \cdot v_{n-1}^{l} k^{-o^{'}}\\theta^{p}\sigma_v^{q}

where

.. math::

   int\_et(i+m,j)
   =\sum_{i+m,j^{'},l^{'}} c_{(i+m)j^{'}l^{'}}e^{(i+m)k[t-(n-1)h]}
   [t-(n-1)h]^{j^{'}} k^{-l^{'}}.

Implementation:

1. Function :py:func:`~ajdmom.ito_mom.int_e_poly` in module
   :py:mod:`~ajdmom.ito_mom` is defined to accomplish the computation in
   equation :eq:`int-e-poly` of :doc:`../design`.

2. Function :py:func:`~ajdmom.ito_mom.recursive_IEII` in module
   :py:mod:`~ajdmom.ito_mom` is defined to realize
   the recursive step in equation :eq:`ito-moment` of :doc:`../theory`.

3. Function :py:func:`~ajdmom.ito_mom.moment_IEII` in module
   :py:mod:`~ajdmom.ito_mom` is implemented to calculate
   :math:`\mathbb{E}[I\!E_n^{n_3}I_n^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]`.

For demonstration, I re-write the following initial three moments in
:ref:`ito-recursive-i` in :doc:`../theory` according to the "polynomial"
representation

.. math::

   \mathbb{E}[I\!E_{n-1,t}^2|v_{n-1}]
   &=& \\frac{1}{2}&e^{2k(n-1)h} e^{2k[t-(n-1)h]}[t-(n-1)h]^0v_{n-1}^0
   k^{-1}\\theta^1\sigma_v^0\\\\
   && + &e^{2k(n-1)h}e^{k[t-(n-1)h]}[t-(n-1)h]^0v_{n-1}^1
   k^{-1}\\theta^0\sigma_v^0\\\\
   && - &e^{2k(n-1)h}e^{k[t-(n-1)h]}[t-(n-1)h]^0v_{n-1}^0
   k^{-1}\\theta^1\sigma_v^0\\\\
   && - &e^{2k(n-1)h}e^{0k[t-(n-1)h]}[t-(n-1)h]^0v_{n-1}^1
   k^{-1}\\theta^0\sigma_v^0\\\\
   && + \\frac{1}{2} &e^{2k(n-1)h}e^{0k[t-(n-1)h]}[t-(n-1)h]^0v_{n-1}^0
   k^{-1}\\theta^1\sigma_v^0,\\\\
   %
   \mathbb{E}[I\!E_{n-1,t}I_{n-1,t}|v_{n-1}]
   &=& &e^{k(n-1)h} e^{k[t-(n-1)h]}[t-(n-1)h]^0v_{n-1}^0
   k^{-1}\\theta^1\sigma_v^0\\\\
   && +&e^{k(n-1)h} e^{0k[t-(n-1)h]}[t-(n-1)h]^1v_{n-1}^1
   k^{-0}\\theta^0\sigma_v^0\\\\
   && -&e^{k(n-1)h} e^{0k[t-(n-1)h]}[t-(n-1)h]^1v_{n-1}^0
   k^{-0}\\theta^1\sigma_v^0\\\\
   && -&e^{k(n-1)h} e^{0k[t-(n-1)h]}[t-(n-1)h]^0v_{n-1}^0
   k^{-1}\\theta^1\sigma_v^0,\\\\
   %
   \mathbb{E}[I_{n-1,t}^2|v_{n-1}]
   &=&-&e^{0k(n-1)h} e^{-k[t-(n-1)h]}[t-(n-1)h]^0v_{n-1}^1
   k^{-1}\\theta^0\sigma_v^0\\\\
   && +&e^{0k(n-1)h} e^{-k[t-(n-1)h]}[t-(n-1)h]^0v_{n-1}^0
   k^{-1}\\theta^1\sigma_v^0\\\\
   && +&e^{0k(n-1)h} e^{0k[t-(n-1)h]}[t-(n-1)h]^1v_{n-1}^0
   k^{-0}\\theta^1\sigma_v^0\\\\
   && +&e^{0k(n-1)h} e^{0k[t-(n-1)h]}[t-(n-1)h]^0v_{n-1}^1
   k^{-1}\\theta^0\sigma_v^0\\\\
   && -&e^{0k(n-1)h} e^{0k[t-(n-1)h]}[t-(n-1)h]^0v_{n-1}^0
   k^{-1}\\theta^1\sigma_v^0.

"""
from fractions import Fraction as Frac

from ajdmom.poly import Poly


def c_nmi(n, m, i):
    """Coefficent :math:`c_{nmi}` as in :eq:`c-nmi`.

    :param int n: n in :math:`e^{nkt}`.
    :param int m: m in :math:`t^{m-i}`.
    :param int i: i in :math:`t^{m-i}`.
    :return: the coefficient :math:`c_{nmi}`.
    :rtype: Fraction
    """
    prod = 1
    for j in range(m - i + 1, m + 1):
        prod = prod * j
    den = n ** (i + 1)  # denumerator
    c = ((-1) ** i) * Frac(prod, den)
    return c


def int_et(n, m):
    """:math:`\int_{(n-1)h}^{t} e^{ik[s-(n-1)h]}[s-(n-1)h]^jds`

    :param int n: i in :math:`e^{ik[s-(n-1)h]}[s-(n-1)h]^j`.
    :param int m: j in :math:`e^{ik[s-(n-1)h]}[s-(n-1)h]^j`.
    :return: a poly with attribute ``keyfor`` =
       ('e^{k[t-(n-1)h]}', '[t-(n-1)h]', 'k^{-}').
    :rtype: Poly
    """
    if m < 0:
        msg = f"m in int_et(n,m) equals {m}, however it shouldn't be negative!"
        raise ValueError(msg)
    #
    poly = Poly()
    poly.set_keyfor(['e^{k[t-(n-1)h]}', '[t-(n-1)h]', 'k^{-}'])
    #
    if n == 0:
        poly[(0, m + 1, 0)] = Frac(1, m + 1)
    elif n != 0 and m == 0:
        poly[(n, 0, 1)] = Frac(1, n)
        poly[(0, 0, 1)] = -Frac(1, n)  # - F(0)
    else:
        poly[(n, m, 1)] = Frac(1, n)
        for i in range(1, m + 1):
            c = c_nmi(n, m, i)
            poly[(n, m - i, i + 1)] = c
            if i == m:  # - F(0): - c_nmi
                poly[(0, 0, i + 1)] = -c  # - c * 1/k^{i+1}
    return poly


def int_e_poly(c, tp, m, poly):
    """Integral of :math:`c \\times tp \\times \int_{(n-1)h}^t e^{mks} poly ds`

    :param int c: coefficient to multiply with
    :param int tp: type of the multiplication,

       +----+----------------------------+
       | tp | multiply with              |
       +====+============================+
       | 1  | :math:`e^{k(n-1)h}v_{n-1}` |
       +----+----------------------------+
       | 2  | :math:`-e^{k(n-1)h}\\theta` |
       +----+----------------------------+
       | 3  | :math:`\\theta`             |
       +----+----------------------------+
       | 4  | :math:`\\sigma_v`           |
       +----+----------------------------+

    :param int m: power of :math:`e^{ks}` in the integrand
    :param Poly poly: a Poly object, such as
       :math:`\mathbb{E}[I\!E_t^{n_3} I_t^{n_4} I_t^{*n_5}]`, with attribute
       ``keyfor`` = ('e^{k(n-1)h}', 'e^{k[t-(n-1)h]}', '[t-(n-1)h]',
       'v_{n-1}','k^{-}', 'theta', 'sigma_v').
    :return: poly with the same ``keyfor`` as that in the input poly
    :rtype: Poly
    """
    poln = Poly()
    poln.set_keyfor(poly.keyfor)  # poly = IEII[(n3, n4, n5)]
    for k1 in poly:
        # \int e^{ik[s-(n-1)h]} [s-(n-1)h]^j ds
        #
        # e^{mks} -> e^{mk[s-(n-1)h]} (× e^{mk(n-1)h} compensated back later)
        poly_sub = int_et(m + k1[1], k1[2])
        # ('e^{k[t-(n-1)h]}', '[t-(n-1)h]', 'k^{-}')
        #
        for k2 in poly_sub:
            val = c * poly[k1] * poly_sub[k2]
            #
            # k1[0]+m: × e^{mk(n-1)h} to compensate e^{mk[s-(n-1)h]} in int_et(m + k1[1], k1[2])
            key = (k1[0] + m, k2[0], k2[1], k1[3], k1[4] + k2[2], k1[5], k1[6])
            #
            key = list(key)
            if tp == 1:
                key[0] += 1
                key[3] += 1
            elif tp == 2:
                key[0] += 1
                key[5] += 1
                val = -val
            elif tp == 3:
                key[5] += 1
            else:
                key[6] += 1
            key = tuple(key)
            poln.add_keyval(key, val)
    return poln


def recursive_IEII(n3, n4, n5, IEII):
    """Recursive step in equation :eq:`ito-moment`

    :param int n3: :math:`n_3`
       in :math:`\mathbb{E}[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]`.
    :param int n4: :math:`n_4`
       in :math:`\mathbb{E}[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]`.
    :param int n5: :math:`n_5`
       in :math:`\mathbb{E}[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]`.
    :param dict IEII: a dict with key (n3,n4,n5) and value Poly object with attribute
       ``keyfor`` =
       ('e^{k(n-1)h}', 'e^{k[t-(n-1)h]}', '[t-(n-1)h]', 'v_{n-1}',
       'k^{-}', 'theta', 'sigma_v').
    :return: updated IEII.
    :rtype: dict
    """
    kf = ['e^{k(n-1)h}', 'e^{k[t-(n-1)h]}', '[t-(n-1)h]',
          'v_{n-1}', 'k^{-}', 'theta', 'sigma_v']
    poly = Poly()
    poly.set_keyfor(kf)
    if n3 >= 2 and n4 >= 0 and n5 >= 0:
        c = Frac(n3 * (n3 - 1), 2)
        poly.merge(int_e_poly(c, 1, 1, IEII[(n3 - 2, n4, n5)]))
        poly.merge(int_e_poly(c, 2, 1, IEII[(n3 - 2, n4, n5)]))
        poly.merge(int_e_poly(c, 3, 2, IEII[(n3 - 2, n4, n5)]))
        poly.merge(int_e_poly(c, 4, 1, IEII[(n3 - 1, n4, n5)]))
    if n3 >= 0 and n4 >= 2 and n5 >= 0:
        c = Frac(n4 * (n4 - 1), 2)
        poly.merge(int_e_poly(c, 1, -1, IEII[(n3, n4 - 2, n5)]))
        poly.merge(int_e_poly(c, 2, -1, IEII[(n3, n4 - 2, n5)]))
        poly.merge(int_e_poly(c, 3, 0, IEII[(n3, n4 - 2, n5)]))
        poly.merge(int_e_poly(c, 4, -1, IEII[(n3 + 1, n4 - 2, n5)]))
    if n3 >= 1 and n4 >= 1 and n5 >= 0:
        c = Frac(n3 * n4, 1)
        poly.merge(int_e_poly(c, 1, 0, IEII[(n3 - 1, n4 - 1, n5)]))
        poly.merge(int_e_poly(c, 2, 0, IEII[(n3 - 1, n4 - 1, n5)]))
        poly.merge(int_e_poly(c, 3, 1, IEII[(n3 - 1, n4 - 1, n5)]))
        poly.merge(int_e_poly(c, 4, 0, IEII[(n3, n4 - 1, n5)]))
    if n3 >= 0 and n4 >= 0 and n5 >= 2:
        c = Frac(n5 * (n5 - 1), 2)
        poly.merge(int_e_poly(c, 1, -1, IEII[(n3, n4, n5 - 2,)]))
        poly.merge(int_e_poly(c, 2, -1, IEII[(n3, n4, n5 - 2)]))
        poly.merge(int_e_poly(c, 3, 0, IEII[(n3, n4, n5 - 2)]))
        poly.merge(int_e_poly(c, 4, -1, IEII[(n3 + 1, n4, n5 - 2)]))
    return poly


def moment_IEII(n3, n4, n5):
    """:math:`\mathbb{E}[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]`

    :param int n3: :math:`n_3`
       in :math:`\mathbb{E}[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]`.
    :param int n4: :math:`n_4`
       in :math:`\mathbb{E}[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]`.
    :param int n5: :math:`n_5`
       in :math:`\mathbb{E}[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]`.
    :return: poly with attribute ``keyfor`` =
       ('e^{k(n-1)h}','e^{k[t-(n-1)h]}','[t-(n-1)h]','v_{n-1}','k^{-}','theta',
       'sigma_v').
    :rtype: dict
    """
    if n3 + n4 + n5 < 0:
        raise ValueError(f"moment_IEII({n3},{n4},{n5}) is called!")
    #
    # IEII: a dict of moments of E[IE^{n3}I^{n4}I^{*n5}]
    #
    IEII = {}
    #
    kf = ['e^{k(n-1)h}', 'e^{k[t-(n-1)h]}', '[t-(n-1)h]', 'v_{n-1}', 'k^{-}',
          'theta', 'sigma_v']
    #
    # special poly constants, analog to 0 and 1
    #
    P0 = Poly({(0, 0, 0, 0, 0, 0, 0, 0): Frac(0, 1)})
    P0.set_keyfor(kf)
    P1 = Poly({(0, 0, 0, 0, 0, 0, 0, 0): Frac(1, 1)})
    P1.set_keyfor(kf)
    #
    # n3 + n4 + n5 = 0: special case
    #
    IEII[(0, 0, 0)] = P1  # equiv to constant 1
    #
    # n3 + n4 + n5 = 1
    #
    IEII[(1, 0, 0)] = P0  # equiv to constant 0
    IEII[(0, 1, 0)] = P0
    IEII[(0, 0, 1)] = P0
    #
    # n3 + n4 + n5 = 2
    #
    P200 = Poly({
        (2, 2, 0, 0, 1, 1, 0): Frac(1, 2),
        (2, 1, 0, 1, 1, 0, 0): Frac(1, 1),
        (2, 1, 0, 0, 1, 1, 0): Frac(-1, 1),
        (2, 0, 0, 1, 1, 0, 0): Frac(-1, 1),
        (2, 0, 0, 0, 1, 1, 0): Frac(1, 2)
    })
    P200.set_keyfor(kf)
    P110 = Poly({
        (1, 1, 0, 0, 1, 1, 0): Frac(1, 1),
        (1, 0, 1, 1, 0, 0, 0): Frac(1, 1),
        (1, 0, 1, 0, 0, 1, 0): Frac(-1, 1),
        (1, 0, 0, 0, 1, 1, 0): Frac(-1, 1)
    })
    P110.set_keyfor(kf)
    P020 = Poly({
        (0, -1, 0, 1, 1, 0, 0): Frac(-1, 1),
        (0, -1, 0, 0, 1, 1, 0): Frac(1, 1),
        (0, 0, 1, 0, 0, 1, 0): Frac(1, 1),
        (0, 0, 0, 1, 1, 0, 0): Frac(1, 1),
        (0, 0, 0, 0, 1, 1, 0): Frac(-1, 1)
    })
    P020.set_keyfor(kf)
    #
    IEII[(2, 0, 0)] = P200
    IEII[(1, 1, 0)] = P110
    IEII[(1, 0, 1)] = P0
    IEII[(0, 2, 0)] = P020
    IEII[(0, 1, 1)] = P0
    IEII[(0, 0, 2)] = P020
    #
    if n3 + n4 + n5 <= 2:
        return (IEII[(n3, n4, n5)])
    #
    # n1 + n2 + n3 >= 3: typical cases
    #
    if n3 + n4 + n5 > 3:
        # compute all lower-order moments to get ready
        for n in range(3, n3 + n4 + n5):
            for i in range(n, -1, -1):
                for j in range(n - i, -1, -1):
                    poly = recursive_IEII(i, j, n - i - j, IEII)
                    poly.remove_zero()
                    IEII[(i, j, n - i - j)] = poly
            # delete polys no more needed
            index = [key for key in IEII if key[0] + key[1] + key[2] == n - 2]
            for key in index: del IEII[key]
    # the last one
    poly = recursive_IEII(n3, n4, n5, IEII)
    poly.remove_zero()
    return poly


def moment_v(n):
    """Moment of :math:`v_{n-1}` as in equation :eq:`moment-v`

    :param int n: order of the moment.
    :return: a poly with attribute ``keyfor`` = ('theta','sigma_v^2/k').
    :rtype: Poly
    """
    v = []
    kf = ['theta', 'sigma_v^2/k']
    # n = 0
    poly = Poly({(0, 0): Frac(1, 1)})
    poly.set_keyfor(kf)
    v.append(poly)
    if n == 0: return (poly)
    # n = 1
    poly = Poly({(1, 0): Frac(1, 1)})
    poly.set_keyfor(kf)
    v.append(poly)
    if n == 1: return poly
    # n >= 2
    for i in range(2, n + 1):
        poly = v[-1]  # recursively computing
        poln = Poly()
        poln.set_keyfor(kf)
        for k in poly:
            # times theta and ((i-1)/2) sigma_v^2/k
            poln.add_keyval((k[0] + 1, k[1]), poly[k])
            poln.add_keyval((k[0], k[1] + 1), poly[k] * Frac(i - 1, 2))
        v.append(poln)
    return poln


if __name__ == "__main__":
    # Example usage of the module, see 'tests/test_ito_mom.py' for more test
    from pprint import pprint
    import sys

    print('\nExample usage of the module functions\n')
    #
    kf = ('theta', 'sigma_v^2/k')
    print(f'\nmoment_v(n) returns a poly with keyfor = \n{kf}')
    print(f'moment_v(4) = ')
    pprint(moment_v(4))
    #
    kf = ('e^{k(n-1)h}', 'e^{k[t-(n-1)h]}', '[t-(n-1)h]', 'v_{n-1}', 'k^{-}',
          'theta', 'sigma_v')
    print(f"\nmoment_IEII(n3,n4,n5) returns a poly with keyfor = \n{kf}")
    args = sys.argv[1:]
    n = 3 if len(args) == 0 else int(args[0])
    #
    for n3 in range(n, -1, -1):
        for n4 in range(n - n3, -1, -1):
            n5 = n - n3 - n4
            print(f'\nmoment_IEII({n3},{n4},{n5}) = ')
            pprint(moment_IEII(n3, n4, n5))

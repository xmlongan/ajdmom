r"""
Itô process conditional moments under Square-Root Jump-Diffusion Process

Condition on the starting variance and jumps occurred over the interval.

Note that

- The unconditional moment derivation is supported within
  :py:mod:`ajdmom.mdl_srjd.mom` and :py:mod:`ajdmom.mdl_srjd.cmom`.

- The conditional (given :math:`v_0`) moment derivation is supported within
  :py:mod:`ajdmom.mdl_srjd.cond_mom` and :py:mod:`ajdmom.mdl_srjd.cond_cmom`.


Highlights
===========

- Offer supports for deriving the conditional moments for models including
  jumps in the variance (
  :abbr:`SRJD(Square-Root Jump Diffusion)`,
  :abbr:`SVVJ(Stochastic Volatility with Jumps in the Variance)`,
  :abbr:`SVIJ(Stochastic Volatility with Independent Jumps in the price and
  variance)` and
  :abbr:`SVCJ(Stochastic Volatility with Contemporaneous Jumps in the price
  and variance)`).

- The conditions are that current variance and jumps over the interval are
  given beforehand.


Square-Root Jump Diffusion
============================

The Square-Root Jump Diffusion (SRJD) is described by the following
:abbr:`SDE(Stochastic Differential Equation)`,

.. math::

   dv(t) = k(\theta - v(t))dt + \sigma_v\sqrt{v(t)}dw^v(t) + dz(t),

which adds a jump component :math:`z(t)`
(a :abbr:`CPP(Compound Poisson Process)`) into the CIR diffusion.
We introduce the following notations for simplification,

.. math::

   \begin{align*}
   I\!Z_t &\triangleq \int_0^tdz(s)
   ~~\left(\equiv \sum_{i=1}^{N(t)} J_i\right),\\
   I\!E\!Z_t &\triangleq \int_0^te^{ks}dz(s)
   ~~\left(\equiv \sum_{i=1}^{N(t)} e^{ks_i}J_i\right).
   \end{align*}

The solution of the SDE can then be expressed as

.. math::

  e^{kt}v(t)  = (v_0-\theta) + e^{kt}\theta + \sigma_v I\!E_t + I\!E\!Z_t,

noting that :math:`v_0 \equiv v(0)` and
:math:`I\!E_t \equiv \int_0^t e^{ks}\sqrt{v(s)} dw^v(s)`.
Further,

.. math::

   e^{kt}(v(t) - \theta) - (v_0-\theta) = \sigma_v I\!E_t + I\!E\!Z_t.

In order to derive moment formulas for models including jumps in
the variance, SVVJ, SVIJ and SVCJ, we first compute the condtional
moments

.. math::

   \mathbb{E}[I\!E_t^{n_1} I_t^{n_2} (I_t^{*})^{n_3}|v_0, z(s), 0\le s\le t],

noting that :math:`I_t \equiv \int_0^t \sqrt{v(s)} dw^v(s)`,
:math:`I_t^{*} \equiv \int_0^t \sqrt{v(s)} dw(s)` and the Brownian motion in
the price process is decomposed as
:math:`w^s(t) = \rho w^v(t) + \sqrt{1-\rho^2}w(t)`, refer to the
:doc:`../theory` page
for the definitions of these quantities.

Recursive Equations
====================

Itô process moment
------------------------

.. math::
  :label: ito-jmp-moment-ii

  \begin{align*}
  &\mathbb{E}[ I\!E_t^{n_1} I_t^{n_2} (I_t^{*})^{n_3}|v_0, z(s), 0\le s\le t ] \\
  &= \frac{1}{2} n_1(n_1-1)(v_0-\theta)\times&\int_0^t e^{ks} \mathbb{E}[ I\!E_s^{n_1-2}I_s^{n_2}(I_t^{*})^{n_3}]ds\\
  &\quad + \frac{1}{2} n_1(n_1-1)\theta    \times&\int_0^t e^{2ks} \mathbb{E}[ I\!E_s^{n_1-2}I_s^{n_2}(I_t^{*})^{n_3}]ds\\
  &\quad + \frac{1}{2} n_1(n_1-1)\sigma_v   \times&\int_0^t e^{ks} \mathbb{E}[ I\!E_s^{n_1-1}I_s^{n_2}(I_t^{*})^{n_3}]ds\\
  &\quad + \frac{1}{2} n_1(n_1-1)       \times&\int_0^t e^{ks}I\!E\!Z_s \mathbb{E}[ I\!E_s^{n_1-2}I_s^{n_2}(I_t^{*})^{n_3}]ds\\
  &\color{blue}\quad + \frac{1}{2} n_2(n_2-1)(v_0-\theta)  \times&\color{blue}\int_0^t e^{-ks} \mathbb{E}[ I\!E_s^{n_1}I_s^{n_2-2}(I_t^{*})^{n_3}]ds\\
  &\color{blue}\quad + \frac{1}{2} n_2(n_2-1)\theta  \times&\color{blue}\int_0^t \mathbb{E}[ I\!E_s^{n_1}I_s^{n_2-2}(I_t^{*})^{n_3}]ds\\
  &\color{blue}\quad + \frac{1}{2} n_2(n_2-1)\sigma_v  \times&\color{blue}\int_0^t e^{-ks} \mathbb{E}[ I\!E_s^{n_1+1}I_s^{n_2-2}(I_t^{*})^{n_3}]ds\\
  &\color{blue}\quad + \frac{1}{2} n_2(n_2-1)  \times&\color{blue}\int_0^t e^{-ks}I\!E\!Z_s \mathbb{E}[ I\!E_s^{n_1}I_s^{n_2-2}(I_t^{*})^{n_3}]ds\\
  &\quad + n_1n_2(v_0-\theta) \times&\int_0^t \mathbb{E}[ I\!E_s^{n_1-1}I_s^{n_2-1}(I_t^{*})^{n_3}]ds\\
  &\quad + n_1n_2\theta         \times&\int_0^t e^{ks} \mathbb{E}[ I\!E_s^{n_1-1}I_s^{n_2-1}(I_t^{*})^{n_3}]ds\\
  &\quad + n_1n_2\sigma_v       \times&\int_0^t \mathbb{E}[ I\!E_s^{n_1}I_s^{n_2-1}(I_t^{*})^{n_3}]ds\\
  &\quad + n_1n_2                 \times&\int_0^t I\!E\!Z_s \mathbb{E}[ I\!E_s^{n_1-1}I_s^{n_2-1}(I_t^{*})^{n_3}]ds\\
  &\color{blue}\quad + \frac{1}{2}n_3(n_3-1)(v_0-\theta)  \times&\color{blue} \int_0^t e^{-ks} \mathbb{E}[ I\!E_s^{n_1} I_s^{n_2} (I_t^{*})^{n_3-2}]ds\\
  &\color{blue}\quad + \frac{1}{2}n_3(n_3-1)\theta  \times&\color{blue} \int_0^t  \mathbb{E}[ I\!E_s^{n_1} I_s^{n_2} (I_t^{*})^{n_3-2}]ds\\
  &\color{blue}\quad + \frac{1}{2}n_3(n_3-1)\sigma_v  \times&\color{blue} \int_0^t e^{-ks} \mathbb{E}[ I\!E_s^{n_1+1} I_s^{n_2} (I_t^{*})^{n_3-2}]ds\\
  &\color{blue}\quad + \frac{1}{2}n_3(n_3-1)  \times&\color{blue} \int_0^t e^{-ks} I\!E\!Z_s \mathbb{E}[ I\!E_s^{n_1} I_s^{n_2} (I_t^{*})^{n_3-2}]ds,
  \end{align*}

in which, for notation simplification, the condition notation
:math:`|v_0, z(s), 0\le s\le t` is also removed from
all conditional expectations on the right-hand side of above equation.

We decode
:math:`\mathbb{E}[I\!E_t^{n_1} I_t^{n_2} (I_t^{*})^{n_3}|v_0, z(s), 0\le s\le t]`
as the following :py:class:`~ajdmom.poly.Poly`,

.. math::

  \begin{align*}
  &\mathbb{E}[I\!E_t^{n_1} I_t^{n_2}|v_0, z(s), 0\le s\le t]\\
  &= \sum_{\boldsymbol{j}, \boldsymbol{l}, \boldsymbol{o}, \boldsymbol{p},
  \boldsymbol{q}}
  c_{\boldsymbol{j}, \boldsymbol{l}, \boldsymbol{o}, \boldsymbol{p},
  \boldsymbol{q}}
  e^{j_1kt} t^{j_2} k^{-j_3} (v_0-\theta)^{j_4} \theta^{j_5} \sigma_v^{j_6}
  f_{Z_t}(\boldsymbol{l}, \boldsymbol{o}, \boldsymbol{p},\boldsymbol{q}),
  \end{align*}

where vectors :math:`\boldsymbol{j}, \boldsymbol{l}, \boldsymbol{o},
\boldsymbol{p}, \boldsymbol{q}` denote

.. math::

   \begin{align*}
   \boldsymbol{j} \equiv (j_1, \cdots, j_6),\\
   \boldsymbol{l} \equiv (l_1, \cdots, l_n),\quad
   \boldsymbol{o} \equiv (o_1, \cdots, o_n),\\
   \boldsymbol{p} \equiv (p_2, \cdots, p_n),\quad
   \boldsymbol{q} \equiv (q_2, \cdots, q_n),
   \end{align*}

respectively,
:math:`c_{\boldsymbol{j}, \boldsymbol{l}, \boldsymbol{o}, \boldsymbol{p},\boldsymbol{q}}`
denotes the corresponding constant coefficient
and function :math:`f_{Z_t}(\boldsymbol{l}, \boldsymbol{o},
\boldsymbol{p}, \boldsymbol{q})` is defined as

.. math::
  :label: fZ

   \begin{align*}
   &f_{Z_t}(\boldsymbol{l},\boldsymbol{o},\boldsymbol{p},\boldsymbol{q})\\
   &\triangleq \sum_{i_1=1}^{N(t)}\cdots\sum_{i_n=1}^{N(t)}
     e^{l_1ks_{i_1} + \cdots + l_nks_{i_n}} J_{i_1} \cdots J_{i_n}
     s_{i_1}^{o_1}\cdots s_{i_n}^{o_n} \\
   &\qquad\qquad\qquad \cdot
    e^{p_2k(s_{i_1}\vee s_{i_2}) + \cdots + p_nk(s_{i_1}\vee \cdots
       \vee s_{i_n})}\\
   &\qquad\qquad\qquad \cdot
    (s_{i_1}\vee s_{i_2})^{q_2}\cdots
    (s_{i_1}\vee \cdots \vee s_{i_n})^{q_n}.
    \end{align*}

It should be noted that

- :math:`s_{i1}\vee s_{i2} \equiv \max\{s_{i1}, s_{i2}\}`,
  :math:`s_{i1}\vee s_{i2} \vee s_{i3} \equiv \max\{s_{i1}, s_{i2}, s_{i3}\}`,
  so on and so forth.

- when :math:`n=1`, :math:`\boldsymbol{p} = \boldsymbol{q} = ()`,
  :math:`f_{Z_t}(\boldsymbol{l}, \boldsymbol{o}, \boldsymbol{p},
  \boldsymbol{q}) = \sum_{i_1=1}^{N(t)} e^{l_1ks_{i_1}}J_{i_1} s_{i_1}^{o_1}`.

- when :math:`n=0`, :math:`\boldsymbol{l} = \boldsymbol{o} =
  \boldsymbol{p} = \boldsymbol{q} = ()`,
  :math:`f_{Z_t}(\boldsymbol{l}, \boldsymbol{o}, \boldsymbol{p},
  \boldsymbol{q}) = 1`.

Example formulas for :math:`n_1+n_2=2` combinations,

.. math::

  \begin{align*}
  &\mathbb{E}[I\!E_t^2 |v_0, z(s), 0\le s\le t] \\
  &= (v_0-\theta)k^{-1}(e^{kt}-1) + \frac{1}{2}\theta k^{-1} (e^{2kt} - 1)
  + k^{-1}\sum_{i=1}^{N(t)} e^{ks_i}J_i (e^{kt} - e^{ks_i}),\\
  &\mathbb{E}[I\!E_t I_t |v_0, z(s), 0\le s\le t] \\
  &= (v_0-\theta) t + \theta k^{-1} (e^{kt} - 1)
  + \sum_{i=1}^{N(t)} e^{ks_i}J_i (t-s_i),\\
  &\mathbb{E}[I_t^2 |v_0, z(s), 0\le s\le t] \\
  &= -(v_0-\theta)k^{-1}(e^{-kt} - 1) + \theta t
  - k^{-1}\sum_{i=1}^{N(t)} e^{ks_i}J_i (e^{-kt} - e^{-ks_i}).
  \end{align*}


Integrals
===========

The essential computation now becomes

.. math::
   :label: int_et_fZ

   \int_{0}^t e^{iks} s^j f_{Z_s}(\boldsymbol{l}, \boldsymbol{o},
    \boldsymbol{p}, \boldsymbol{q}) ds.

We present the result and implementation first.

.. math::

  \int_{0}^t e^{iks} s^j f_{Z_s}(\boldsymbol{l}, \boldsymbol{o},
    \boldsymbol{p}, \boldsymbol{q}) ds
  = F_{Z_t}(\boldsymbol{l}, \boldsymbol{o}, \boldsymbol{p}, \boldsymbol{q},
  i, j),

where the function on the right-hand side is defined as

.. math::

  \begin{align*}
   &F_{Z_t}(\boldsymbol{l},\boldsymbol{o},\boldsymbol{p},\boldsymbol{q},
    i, j)\\
   &\triangleq \sum_{i_1=1}^{N(t)}\cdots\sum_{i_n=1}^{N(t)}
     e^{l_1ks_{i_1} + \cdots + l_nks_{i_n}} J_{i_1} \cdots J_{i_n}
     s_{i_1}^{o_1}\cdots s_{i_n}^{o_n} \\
   &\qquad\qquad\qquad \cdot
    e^{p_2k(s_{i_1}\vee s_{i_2}) + \cdots + p_nk(s_{i_1}\vee \cdots
       \vee s_{i_n})}\\
   &\qquad\qquad\qquad \cdot
    (s_{i_1}\vee s_{i_2})^{q_2}\cdots
    (s_{i_1}\vee \cdots \vee s_{i_n})^{q_n}\\
   &\qquad\qquad\qquad \cdot
   \int_{s_{i_1}\vee \cdots \vee s_{i_n}}^t e^{iks} s^j ds.
   \end{align*}

The integral in :eq:`int_et_fZ` is implemented in
:py:func:`~ajdmom.ito_cond_mom.int_et_fZ` in module
:py:mod:`~ajdmom.ito_cond_mom`. The integral
:math:`\int_{s_{i_1}\vee \cdots \vee s_{i_n}}^t e^{iks} s^j ds` can be
calculated as we did for :math:`\int_0^t e^{iks} s^j ds` in
:doc:`../generated/ajdmom.ito_mom`.

Then we explain the calculations.
Let's take a look at a simple example,
:math:`\int_0^te^{ks}I\!E\!Z_s ds`.

.. math::

  e^{ks}I\!E\!Z_s =
   \begin{cases}
     0,                                     &0 ~~~~     \le s<s_1,\\
     e^{ks}\sum_{i=1}^1 e^{ks_i}J_i,        &s_1~~~     \le s<s_2,\\
                                                  &   \vdots     \\
     e^{ks}\sum_{i=1}^{N(t)-1} e^{ks_i}J_i, &s_{N(t)-1}  \le s<s_{N(t)},\\
     e^{ks}\sum_{i=1}^{N(t)} e^{ks_i}J_i,   &s_{N(t)}~~~     \le s< t.
   \end{cases}

.. math::

  \begin{align*}
  &\int_0^t e^{ks}I\!E\!Z_s ds\\
  &= \int_0^{s_1} e^{ks}I\!E\!Z_s ds + \int_{s_1}^{s_2} e^{ks}I\!E\!Z_s ds
    + \cdots + \int_{s_n}^{t} e^{ks}I\!E\!Z_s ds\\
  &= \frac{1}{k}(e^{ks_2} - e^{ks_1}) \sum_{i=1}^{1}e^{ks_i}J_i + \cdots
    + \frac{1}{k}(e^{ks_n} - e^{ks_{n-1}})\sum_{i=1}^{N(t)-1}e^{ks_i}J_i\\
  &  \quad + \frac{1}{k}(e^{kt} - e^{ks_n})\sum_{i=1}^{N(t)}e^{ks_i}J_i\\
  &= \sum_{i=1}^{N(t)} e^{ks_i}J_i \frac{1}{k}(e^{kt} - e^{ks_i}).
  \end{align*}

In short,

.. math::

  \begin{align*}
  I\!E\!Z_t &= \sum_{i=1}^{N(t)} e^{ks_i}J_i,\\
  \int_0^t e^{ks}I\!E\!Z_s ds &= \sum_{i=1}^{N(t)} e^{ks_i}J_i
  \frac{1}{k} (e^{kt} - e^{ks_i}).
  \end{align*}

Another example

.. math::

  \int_0^t e^{ks} I\!E\!Z_s I\!E\!Z_s ds = ?

.. math::

  \begin{align*}
  &\int_0^t e^{ks} \sum_{i=1}^{N(s)}\sum_{j=1}^{N(s)} e^{ks_i+ks_j}J_iJ_jds\\
  &= \sum_{i=1}^{N(t)}\sum_{j=1}^{N(t)}e^{ks_i+ks_j}J_iJ_j \frac{1}{k}
  (e^{kt} -e^{k(s_i\vee s_j)}).
  \end{align*}

The two examples should have explained the derivation well.

Implementation summary
-----------------------

1. Define :py:func:`~ajdmom.ito_cond_mom.recursive_IEII` to realize the
   recursive step in equation :eq:`ito-jmp-moment-ii`.

2. Define :py:func:`~ajdmom.ito_cond_mom.moment_IEII` to finish the computation
   of :math:`\mathbb{E}[I\!E_t^{n_1}I_t^{n_2}(I_t^{*})^{n_3}|v_0, z_s,
   0\le s\le t]`.

"""
from fractions import Fraction as Frac

from ajdmom.poly import Poly
from ajdmom.ito_mom import c_nmi


def int_et_fZ(n, m, N_sum):
    r"""integral of :math:`\int_0^t e^{iks} s^j f_{Z_s}(l,o,p,q)ds`

    For each element with index :math:`(s_{i1},\dots,s_{in})`,
    the integral becomes
    :math:`\int_{s_{i1}\vee\cdots\vee s_{in}}^t e^{iks} s^j ds`.
    When there is no summation at all in :math:`f_{Z_s}(l,o,p,q)`,
    the whole integral simplifies to :math:`\int_0^t e^{iks} s^jds`.

    :param int n: power of :math:`e^{ks}`, i.e., :math:`i`.
    :param int m: power of :math:`s`, i.e., :math:`j`.
    :param int N_sum: level of summations in :math:`f_{Z_s}(l,o,p,q)`.
    :return: a poly with attribute ``keyfor`` =
      ('e^{kt}', 't', 'k^{-}', 'e^{k(s_i1 v...v s_in)}', '(s_i1 v...v s_in)').
    :rtype: Poly
    """
    if m < 0:
        msg = f"m in int_et_fZ(n,m) equals {m}, however it must be non-negative!"
        raise ValueError(msg)
    #
    poly = Poly()
    kf = ['e^{kt}', 't', 'k^{-}', 'e^{k(s_i1 v...v s_in)}', '(s_i1 v...v s_in)']
    poly.set_keyfor(kf)
    #
    if N_sum == 0:  # lower bound -> 0
        if n == 0:
            poly[(0, m + 1, 0, 0, 0)] = Frac(1, m + 1)
        elif n != 0 and m == 0:
            poly[(n, 0, 1, 0, 0)] = Frac(1, n)
            poly[(0, 0, 1, 0, 0)] = -Frac(1, n)  # - F(0), F(0) != 0
        else:
            poly[(n, m, 1, 0, 0)] = Frac(1, n)
            for i in range(1, m + 1):
                c = c_nmi(n, m, i)
                poly[(n, m - i, i + 1, 0, 0)] = c
                if i == m:  # - F(0): - c_nmi, F(0) != 0
                    poly[(0, 0, i + 1, 0, 0)] = -c
    else:  # lower bound -> (s_i1 v...v s_in)
        if n == 0:
            poly[(0, m + 1, 0, 0, 0)] = Frac(1, m + 1)
            poly[(0, 0, 0, 0, m + 1)] = -Frac(1, m + 1)
        elif n != 0 and m == 0:
            poly[(n, 0, 1, 0, 0)] = Frac(1, n)
            poly[(0, 0, 1, n, 0)] = -Frac(1, n)
        else:
            poly[(n, m, 1, 0, 0)] = Frac(1, n)
            poly[(0, 0, 1, n, m)] = -Frac(1, n)
            for i in range(1, m + 1):
                c = c_nmi(n, m, i)
                poly[(n, m - i, i + 1, 0, 0)] = c
                poly[(0, 0, i + 1, n, m - i)] = -c
    return poly


def int_e_poly(coef, tp, m, poly):
    r"""integral of :math:`coef \times tp \times \int_0^t e^{mks} poly ds`

    :param float coef: coefficient to multiply with
    :param int tp: type of the multiplication,

       +----+----------------------------+
       | tp | multiply with              |
       +====+============================+
       | 1  | :math:`v_0-\theta`         |
       +----+----------------------------+
       | 2  | :math:`\theta`             |
       +----+----------------------------+
       | 3  | :math:`\sigma_v`           |
       +----+----------------------------+

    :param int m: power of :math:`e^{ks}` in the integrand
    :param Poly poly: a Poly object, such as
      :math:`\mathbb{E}[I\!E_t^{n_1} I_t^{n_2} I_t^{*n_3}
      |v_0, z(s), 0\le s\le t]`, with attribute ``keyfor`` =
      ('e^{kt}', 't', 'k^{-}', 'v0-theta', 'theta', 'sigma',
      'l_{1:n}', 'o_{1:n}', 'p_{2:n}', 'q_{2:n}')
    :return: a Poly with the same ``keyfor`` of the input poly
    :rtype: Poly
    """
    poln = Poly()
    poln.set_keyfor(poly.keyfor)  # poly = IEII[(n1, n2, n3)]
    for k in poly:
        poly_sub = int_et_fZ(m + k[0], k[1], len(k[6]))  # \int e^{iks} s^{j} ds
        # ['e^{kt}', 't', 'k^{-}', 'e^{k(s_i1 v...v s_in)}', '(s_i1 v...v s_in)']
        for kk in poly_sub:
            # key1 + key2
            if tp == 1:
                key1 = (kk[0], kk[1], kk[2] + k[2], k[3] + 1, k[4], k[5])
            elif tp == 2:
                key1 = (kk[0], kk[1], kk[2] + k[2], k[3], k[4] + 1, k[5])
            else:
                key1 = (kk[0], kk[1], kk[2] + k[2], k[3], k[4], k[5] + 1)
            #
            if len(k[6]) == 0:  # 0 summation  inside
                key2 = ((), (), (), ())
            elif len(k[6]) == 1:  # 1 summation  inside
                k6 = (k[6][0] + kk[3],)
                k7 = (k[7][0] + kk[4],)
                key2 = (k6, k7, (), ())
            else:  # 2 summations inside or more
                k8 = list(k[8])
                k8[-1] += kk[3]
                k8 = tuple(k8)
                k9 = list(k[9])
                k9[-1] += kk[4]
                k9 = tuple(k9)
                key2 = (k[6], k[7], k8, k9)
            poln.add_keyval(key1 + key2, coef * poly[k] * poly_sub[kk])
    return poln


def int_e_IEZ_poly(coef, m, poly):
    r"""integral of :math:`coef \times \int_0^t e^{mks} I\!E\!Z_s poly ds`

    :param int coef: coefficient to multiply with
    :param int m: power of :math:`e^{ks}` in the integrand
    :param Poly poly: a Poly object, such as
      :math:`\mathbb{E}[I\!E_t^{n_1} I_t^{n_2} I_t^{*n_3}
      |v_0, z(s), 0\le s\le t]`, with attribute ``keyfor`` =
      ('e^{kt}', 't', 'k^{-}', 'v0-theta', 'theta', 'sigma',
      'l_{1:n}', 'o_{1:n}', 'p_{2:n}', 'q_{2:n}')
    :return: a Poly with the same ``keyfor`` of the input poly
    :rtype: Poly
    """
    poln = Poly()
    poln.set_keyfor(poly.keyfor)  # poly = IEII[(n1, n2, n3)]
    for k in poly:
        # IEZ times poly
        k6 = k[6] + (1,)
        k7 = k[7] + (0,)
        if len(k6) >= 2:
            k8 = k[8] + (0,)
            k9 = k[9] + (0,)
        #
        # \int e^{iks} s^{j} poly ds
        poly_sub = int_et_fZ(m + k[0], k[1], len(k6))
        # ['e^{kt}', 't', 'k^{-}', 'e^{k(s_i1 v...v s_in)}', '(s_i1 v...v s_in)']
        for kk in poly_sub:
            # key1
            key1 = (kk[0], kk[1], kk[2] + k[2], k[3], k[4], k[5])
            # key2
            if len(k6) == 1:  # 1 summation  inside
                k6 = (k6[0] + kk[3],)
                k7 = (k7[0] + kk[4],)
                key2 = (k6, k7, (), ())
            else:  # 2 summations inside or more
                k8 = list(k8)
                k8[-1] += kk[3]
                k8 = tuple(k8)
                k9 = list(k9)
                k9[-1] += kk[4]
                k9 = tuple(k9)
                key2 = (k6, k7, k8, k9)
            poln.add_keyval(key1 + key2, coef * poly[k] * poly_sub[kk])
    return poln


def recursive_IEII(n1, n2, n3, IEII):
    r"""Recursive step in equation :eq:`ito-jmp-moment-ii`

    :param int n1: power of :math:`I\!E_s` in the integrand.
    :param int n2: power of :math:`I_s` in the integrand.
    :param int n3: power of :math:`I_s^{*}` in the integrand.
    :param dict IEII: a dict of joint conditional moments of
      :math:`I\!E_sI_sI_s^{*}`, with key (n1, n2, n3) and value Poly object
      with attribute ``keyfor`` =
      ('e^{kt}','t','k^{-}','v0-theta','theta','sigma',
      'l_{1:n}', 'o_{1:n}', 'p_{2:n}', 'q_{2:n}').
    :return: poly with the same ``keyfor`` of that of in IEII
    :rtype: Poly
    """
    kf = ['e^{kt}', 't', 'k^{-}', 'v0-theta', 'theta', 'sigma']
    kf += ['l_{1:n}', 'o_{1:n}', 'p_{2:n}', 'q_{2:n}']
    poly = Poly()
    poly.set_keyfor(kf)
    #
    if n1 >= 2 and n2 >= 0 and n3 >= 0:
        c = Frac(n1 * (n1 - 1), 2)
        poly.merge(int_e_poly(c, 1, 1, IEII[(n1 - 2, n2, n3)]))
        poly.merge(int_e_poly(c, 2, 2, IEII[(n1 - 2, n2, n3)]))
        poly.merge(int_e_poly(c, 3, 1, IEII[(n1 - 1, n2, n3)]))
        poly.merge(int_e_IEZ_poly(c, 1, IEII[(n1 - 2, n2, n3)]))
    if n1 >= 0 and n2 >= 2 and n3 >= 0:
        c = Frac(n2 * (n2 - 1), 2)
        poly.merge(int_e_poly(c, 1, -1, IEII[(n1, n2 - 2, n3)]))
        poly.merge(int_e_poly(c, 2, 0, IEII[(n1, n2 - 2, n3)]))
        poly.merge(int_e_poly(c, 3, -1, IEII[(n1 + 1, n2 - 2, n3)]))
        poly.merge(int_e_IEZ_poly(c, -1, IEII[(n1, n2 - 2, n3)]))
    if n1 >= 1 and n2 >= 1 and n3 >= 0:
        c = Frac(n1 * n2, 1)
        poly.merge(int_e_poly(c, 1, 0, IEII[(n1 - 1, n2 - 1, n3)]))
        poly.merge(int_e_poly(c, 2, 1, IEII[(n1 - 1, n2 - 1, n3)]))
        poly.merge(int_e_poly(c, 3, 0, IEII[(n1, n2 - 1, n3)]))
        poly.merge(int_e_IEZ_poly(c, 0, IEII[(n1 - 1, n2 - 1, n3)]))
    if n1 >= 0 and n2 >= 0 and n3 >= 2:
        c = Frac(n3 * (n3 - 1), 2)
        poly.merge(int_e_poly(c, 1, -1, IEII[(n1, n2, n3 - 2)]))
        poly.merge(int_e_poly(c, 2, 0, IEII[(n1, n2, n3 - 2)]))
        poly.merge(int_e_poly(c, 3, -1, IEII[(n1 + 1, n2, n3 - 2)]))
        poly.merge(int_e_IEZ_poly(c, -1, IEII[(n1, n2, n3 - 2)]))
    return poly


def moment_IEII(n1, n2, n3):
    r""":math:`\mathbb{E}[I\!E_t^{n_1}I_t^{n_2}(I_t^{*})^{n_3}
    |v_0, z(s), 0\le s\le t]`

    :param int n1: power of :math:`I\!E_t`.
    :param int n2: power of :math:`I_t`.
    :param int n3: power of :math:`I_t^{*}`.
    :return: poly with ``keyfor`` =
       ('e^{kt}','t','k^{-}','v0-theta','theta','sigma',
       'l_{1:n}', 'o_{1:n}', 'p_{2:n}', 'q_{2:n}').
    :rtype: Poly
    """
    if n1 + n2 + n3 < 0:
        raise ValueError(f"moment_IEII({n1},{n2},{n3}) is called!")
    #
    # IEII: a dict of conditional moments of E[IE_t^{n1}I_t^{n2}I_t^{*n3}]
    #
    IEII = {}
    #
    kf = ['e^{kt}', 't', 'k^{-}', 'v0-theta', 'theta', 'sigma']
    kf += ['l_{1:n}', 'o_{1:n}', 'p_{2:n}', 'q_{2:n}']
    #
    # special poly constants, analog to 0 and 1
    #
    P0 = Poly({(0, 0, 0, 0, 0, 0, (), (), (), ()): Frac(0, 1)})
    P0.set_keyfor(kf)
    P1 = Poly({(0, 0, 0, 0, 0, 0, (), (), (), ()): Frac(1, 1)})
    P1.set_keyfor(kf)
    #
    # n1 + n2 + n3 = 0: special case
    #
    IEII[(0, 0, 0)] = P1  # equiv to constant 1
    #
    # n1 + n2 + n3 = 1
    #
    IEII[(1, 0, 0)] = P0  # equiv to constant 0
    IEII[(0, 1, 0)] = P0
    IEII[(0, 0, 1)] = P0
    #
    # n1 + n2 +n3 = 2
    #
    P200 = Poly({
        (1, 0, 1, 1, 0, 0, (), (), (), ()): Frac(1, 1),
        (0, 0, 1, 1, 0, 0, (), (), (), ()): -Frac(1, 1),
        (2, 0, 1, 0, 1, 0, (), (), (), ()): Frac(1, 2),
        (0, 0, 1, 0, 1, 0, (), (), (), ()): -Frac(1, 2),
        (1, 0, 1, 0, 0, 0, (1,), (0,), (), ()): Frac(1, 1),
        (0, 0, 1, 0, 0, 0, (2,), (0,), (), ()): -Frac(1, 1)
    })
    P200.set_keyfor(kf)
    P110 = Poly({
        (0, 1, 0, 1, 0, 0, (), (), (), ()): Frac(1, 1),
        (1, 0, 1, 0, 1, 0, (), (), (), ()): Frac(1, 1),
        (0, 0, 1, 0, 1, 0, (), (), (), ()): -Frac(1, 1),
        (0, 1, 0, 0, 0, 0, (1,), (0,), (), ()): Frac(1, 1),
        (0, 0, 0, 0, 0, 0, (1,), (1,), (), ()): -Frac(1, 1)
    })
    P110.set_keyfor(kf)
    P020 = Poly({
        (-1, 0, 1, 1, 0, 0, (), (), (), ()): -Frac(1, 1),
        (0, 0, 1, 1, 0, 0, (), (), (), ()): Frac(1, 1),
        (0, 1, 0, 0, 1, 0, (), (), (), ()): Frac(1, 1),
        (-1, 0, 1, 0, 0, 0, (1,), (0,), (), ()): -Frac(1, 1),
        (0, 0, 1, 0, 0, 0, (0,), (0,), (), ()): Frac(1, 1)
    })
    P020.set_keyfor(kf)
    IEII[(2, 0, 0)] = P200
    IEII[(1, 1, 0)] = P110
    IEII[(1, 0, 1)] = P0
    IEII[(0, 2, 0)] = P020
    IEII[(0, 1, 1)] = P0
    IEII[(0, 0, 2)] = P020
    #
    if n1 + n2 + n3 <= 2: return IEII[(n1, n2, n3)]
    #
    # n1 + n2 + n3 >= 3: typical cases
    #
    if n1 + n2 + n3 > 3:
        # compute all lower-order moments to get ready
        for n in range(3, n1 + n2 + n3):
            for i in range(n, -1, -1):
                for j in range(n - i, -1, -1):
                    poly = recursive_IEII(i, j, n - i - j, IEII)
                    poly.remove_zero()
                    IEII[(i, j, n - i - j)] = poly
            # delete polys no more needed
            index = [key for key in IEII if key[0] + key[1] + key[2] == n - 2]
            for key in index: del IEII[key]
    # the last one
    poly = recursive_IEII(n1, n2, n3, IEII)
    poly.remove_zero()
    return poly


if __name__ == "__main__":
    import sys
    from pprint import pprint

    print('\nExample usage of the module function\n')
    kf = ['e^{kt}', 't', 'k^{-}', 'v0-theta', 'theta', 'sigma']
    kf += ['l_{1:n}', 'o_{1:n}', 'p_{2:n}', 'q_{2:n}']
    print(f"moment_IEII(n1, n2, n3) returns a poly with keyfor = \n{kf}")
    #
    args = sys.argv[1:]
    n = 3 if len(args) == 0 else int(args[0])
    for n1 in range(n, -1, -1):
        for n2 in range(n - n1, -1, -1):
            n3 = n - n1 - n2
            print(f'\nmoment_IEII({n1},{n2},{n3}) = ')
            pprint(moment_IEII(n1, n2, n3))

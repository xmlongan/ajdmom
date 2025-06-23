r"""
Joint Conditional Moment of :math:`I\!E_t,I_t,I_t^{*},I\!E\!Z_t,I\!Z_t,I\!Z_t^{*}|v_0`

In this module, we focus on computing the following conditional joint moment:

.. math::

   \begin{equation}\label{eqn:joint-ieii-ieziziz}
    \mathbb{E}[I\!E_t^{m_1}I_t^{m_2}I_t^{*m_3}I\!E\!Z_t^{m_4}I\!Z_t^{m_5}I\!Z_t^{*m_6}|v_0].
   \end{equation}

While the recursive computation of :math:`\mathbb{E}[I\!E_t^{m_1} I_t^{m_2} I_t^{*m_3}|v_0]` is 
well-established for models such as the Heston model, we encounter a new challenge: 
the last three quantities :math:`I\!E\!Z_t^{m_4} I\!Z_t^{m_5} I\!Z_t^{*m_6}` are not independent
of the first three quantities :math:`I\!E_t^{m_1} I_t^{m_2} I_t^{*m_3}` in above equation.
For :math:`m_4 + m_5 + m_6 \ge 1`, we must evaluate integrals of the form

.. math::

    \begin{equation}
        \int_0^t e^{lks} \mathbb{E}[I\!E_s^{m_1}I_s^{m_2}I_s^{*m_3} I\!E\!Z_t^{m_4} I\!Z_t^{m_5} I\!Z_t^{*m_6}|v_0]\mathrm{d} s.
    \end{equation}

The dependence between :math:`I\!E_s^{m_1}I_s^{m_2}I_s^{*m_3}` and
:math:`I\!E\!Z_t^{m_4} I\!Z_t^{m_5} I\!Z_t^{*m_6}` motivates us to decompose the latter
as follows:

.. math::

    \begin{align*}
        &I\!E\!Z_t^{m_4}I\!Z_t^{m_5}I\!Z_t^{*m_6}\\
        &= \sum_{i_1=0}^{m_4}\sum_{i_2=0}^{m_5}\sum_{i_3=0}^{m_6} \binom{m_4}{i_1}\binom{m_5}{i_2}
        \binom{m_6}{i_3} I\!E\!Z_s^{i_1}I\!Z_s^{i_2}I\!Z_s^{*i_3} I\!E\!Z_{s,t}^{m_4-i_1}
        I\!Z_{s,t}^{m_5-i_2} I\!Z_{s,t}^{*m_6-i_3}, \quad \forall s \le t,
    \end{align*}

where :math:`I\!E\!Z_t` is split into two independent parts :math:`I\!E\!Z_s, I\!E\!Z_{s,t}`, i.e.,
:math:`I\!E\!Z_t = I\!E\!Z_s + I\!E\!Z_{s,t}`. Similarly, :math:`I\!Z_t` and :math:`I\!Z_t^*`
are decomposed as :math:`I\!Z_t = I\!Z_s + I\!Z_{s,t}` and :math:`I\!Z_t^* = I\!Z_s^* + I\!Z_{s,t}^*`,
respectively. Here, the new terms :math:`I\!E\!Z_{s,t}`, :math:`I\!Z_{s,t}` and :math:`I\!Z_{s,t}^{*}`
are defined as

.. math::

    \begin{equation*}
        I\!E\!Z_{s,t} \mathrel{:=} \int_s^te^{ku}\mathrm{d} z^v(u),
        \quad I\!Z_{s,t} \mathrel{:=} \int_s^t\mathrm{d} z^v(u),
        \quad I\!Z^{*}_{s,t} \mathrel{:=} \int_s^t\mathrm{d} z^{s}(u).
    \end{equation*}

Consequently, we have

.. math::

    \begin{align*}
        &\int_0^t e^{lks} \mathbb{E}[I\!E_s^{m_1}I_s^{m_2}I_s^{*m_3} I\!E\!Z_t^{m_4} I\!Z_t^{m_5} I\!Z_t^{*m_6}|v_0]\mathrm{d} s\\
        &= \sum_{i_1=0}^{m_4}\sum_{i_2=0}^{m_5}\sum_{i_3=0}^{m_6}\binom{m_4}{i_1}\binom{m_5}{i_2}\binom{m_6}{i_3} \int_0^t e^{lks} \mathbb{E}[I\!E_s^{m_1}I_s^{m_2}I_s^{*m_3} I\!E\!Z_s^{i_1}I\!Z_s^{i_2}I\!Z_s^{*i_3}|v_0] \cdot M(i_1,i_2,i_3)\mathrm{d} s.
    \end{align*}

where
:math:`M(i_1,i_2,i_3)\mathrel{:=} \mathbb{E}[I\!E\!Z_{s,t}^{m_4-i_1} I\!Z_{s,t}^{m_5-i_2} I\!Z_{s,t}^{*m_6-i_3}|v_0]`.
We have established that the conditional joint moment
:math:`\mathbb{E}[I\!E\!Z_{s,t}^{m_4}I\!Z_{s,t}^{m_5}I\!Z_{s,t}^{*m_6}|v_0]` can be computed as
the following "polynomial":

.. math::

    \begin{equation}
        \mathbb{E}[I\!E\!Z_{s,t}^{m_4}I\!Z_{s,t}^{m_5}I\!Z_{s,t}^{*m_6}|v_0]
        = \sum_{\boldsymbol{j}}c_{\boldsymbol{j}}e^{j_1kt}t^{j_2}e^{j_3js}s^{j_4}k^{-j_5}\lambda^{j_6}\mu_v^{j_7}\mu_s^{j_8}\sigma_s^{j_9},
    \end{equation}

where :math:`\boldsymbol{j}\mathrel{:=} (j_1,\dots, j_9), j_1,\dots,j_9` are integers,
:math:`c_{\boldsymbol{j}}` represents the corresponding monomial coefficient. For detailed derivations,
please refer to :doc:`../generated/ajdmom.mdl_svcj.ieziziz_mom`.

With above equation, the conditional joint moment can be computed recursively as follows:

.. math::

    \begin{align}
        &\mathbb{E}[I\!E_t^{m_1} I_t^{m_2} I_t^{*m_3} I\!E\!Z_t^{m_4} I\!Z_t^{m_5} I\!Z_t^{*m_6}|v_0]\nonumber\\
        &=\sum_{i_1=0}^{m_4}\sum_{i_2=0}^{m_5}\sum_{i_3=0}^{m_6}\binom{m_4}{i_1}\binom{m_5}{i_2}\binom{m_6}{i_3}
        \sum_{\boldsymbol{j}}c_{\boldsymbol{j}}e^{j_1kt}t^{j_2}k^{-j_5}\lambda^{j_6}\mu_v^{j_7}\mu_s^{j_8}
        \sigma_s^{j_9}F(m_1, m_2, m_3),%\label{eqn:recursive-ieii-ieziziz}
    \end{align}

where

.. math::

    \begin{align*}
        F(m_1,m_2,m_3)
        &\mathrel{:=} \sum_{i=1}^4\left[\frac{m_1(m_1-1)}{2}f_{6i}
          + \frac{m_2(m_2-1)}{2}g_{6i}
          + m_1m_2h_{6i}
          + \frac{m_3(m_3-1)}{2}q_{6i}\right],
    \end{align*}

and the terms :math:`f_{6i}, g_{6i}, h_{6i}, q_{6i}, i=1,2,3,4` are defined as:

.. math::

    \begin{align*}
        f_{61} &\mathrel{:=} \int_0^te^{(j_3+1)ks}s^{j_4}\mathbb{E}[I\!E_s^{m_1-2}I_s^{m_2}I_s^{*m_3}I\!E\!Z_s^{i_1}I\!Z_s^{i_2}I\!Z_s^{*i_3}|v_0]\mathrm{d} s \times (v_0-\theta),\\
        f_{62} &\mathrel{:=} \int_0^te^{(j_3+2)ks}s^{j_4}\mathbb{E}[I\!E_s^{m_1-2}I_s^{m_2}I_s^{*m_3}I\!E\!Z_s^{i_1}I\!Z_s^{i_2}I\!Z_s^{*i_3}|v_0]\mathrm{d} s \times \theta,\\
        f_{63} &\mathrel{:=} \int_0^te^{(j_3+1)ks}s^{j_4}\mathbb{E}[I\!E_s^{m_1-1}I_s^{m_2}I_s^{*m_3}I\!E\!Z_s^{i_1}I\!Z_s^{i_2}I\!Z_s^{*i_3}|v_0]\mathrm{d} s \times \sigma_v,\\
        f_{64} &\mathrel{:=} \int_0^te^{(j_3+1)ks}s^{j_4}\mathbb{E}[I\!E_s^{m_1-2}I_s^{m_2}I_s^{*m_3}I\!E\!Z_s^{i_1+1}I\!Z_s^{i_2}I\!Z_s^{*i_3}|v_0]\mathrm{d} s,
    \end{align*}

.. math::

    \begin{align*}
        g_{61} &\mathrel{:=} \int_0^te^{(j_3-1)ks}s^{j_4}\mathbb{E}[I\!E_s^{m_1}I_s^{m_2-2}I_s^{*m_3}I\!E\!Z_s^{i_1}I\!Z_s^{i_2}I\!Z_s^{*i_3}|v_0]\mathrm{d} s \times (v_0-\theta),\\
        g_{62} &\mathrel{:=} \int_0^te^{(j_3-0)ks}s^{j_4}\mathbb{E}[I\!E_s^{m_1}I_s^{m_2-2}I_s^{*m_3}I\!E\!Z_s^{i_1}I\!Z_s^{i_2}I\!Z_s^{*i_3}|v_0]\mathrm{d} s \times \theta,\\
        g_{63} &\mathrel{:=} \int_0^te^{(j_3-1)ks}s^{j_4}\mathbb{E}[I\!E_s^{m_1+1}I_s^{m_2-2}I_s^{*m_3}I\!E\!Z_s^{i_1}I\!Z_s^{i_2}I\!Z_s^{*i_3}|v_0]\mathrm{d} s \times \sigma_v,\\
        g_{64} &\mathrel{:=} \int_0^te^{(j_3-1)ks}s^{j_4}\mathbb{E}[I\!E_s^{m_1}I_s^{m_2-2}I_s^{*m_3}I\!E\!Z_s^{i_1+1}I\!Z_s^{i_2}I\!Z_s^{*i_3}|v_0]\mathrm{d} s,
    \end{align*}

.. math::

    \begin{align*}
        h_{61} &\mathrel{:=} \int_0^te^{(j_3+0)ks}s^{j_4}\mathbb{E}[I\!E_s^{m_1-1}I_s^{m_2-1}I_s^{*m_3}I\!E\!Z_s^{i_1}I\!Z_s^{i_2}I\!Z_s^{*i_3}|v_0]\mathrm{d} s \times (v_0-\theta),\\
        h_{62} &\mathrel{:=} \int_0^te^{(j_3+1)ks}s^{j_4}\mathbb{E}[I\!E_s^{m_1-1}I_s^{m_2-1}I_s^{*m_3}I\!E\!Z_s^{i_1}I\!Z_s^{i_2}I\!Z_s^{*i_3}|v_0]\mathrm{d} s \times \theta,\\
        h_{63} &\mathrel{:=} \int_0^te^{(j_3+0)ks}s^{j_4}\mathbb{E}[I\!E_s^{m_1-0}I_s^{m_2-1}I_s^{*m_3}I\!E\!Z_s^{i_1}I\!Z_s^{i_2}I\!Z_s^{*i_3}|v_0]\mathrm{d} s \times \sigma_v,\\
        h_{64} &\mathrel{:=} \int_0^te^{(j_3+0)ks}s^{j_4}\mathbb{E}[I\!E_s^{m_1-1}I_s^{m_2-1}I_s^{*m_3}I\!E\!Z_s^{i_1+1}I\!Z_s^{i_2}I\!Z_s^{*i_3}|v_0]\mathrm{d} s,
    \end{align*}

and

.. math::

    \begin{align*}
        q_{61} &\mathrel{:=} \int_0^te^{(j_3-1)ks}s^{j_4}\mathbb{E}[I\!E_s^{m_1}I_s^{m_2}I_s^{*m_3-2}I\!E\!Z_s^{i_1}I\!Z_s^{i_2}I\!Z_s^{*i_3}|v_0]\mathrm{d} s \times (v_0-\theta),\\
        q_{62} &\mathrel{:=} \int_0^te^{(j_3-0)ks}s^{j_4}\mathbb{E}[I\!E_s^{m_1}I_s^{m_2}I_s^{*m_3-2}I\!E\!Z_s^{i_1}I\!Z_s^{i_2}I\!Z_s^{*i_3}|v_0]\mathrm{d} s \times \theta,\\
        q_{63} &\mathrel{:=} \int_0^te^{(j_3-1)ks}s^{j_4}\mathbb{E}[I\!E_s^{m_1+1}I_s^{m_2}I_s^{*m_3-2}I\!E\!Z_s^{i_1}I\!Z_s^{i_2}I\!Z_s^{*i_3}|v_0]\mathrm{d} s \times \sigma_v,\\
        q_{64} &\mathrel{:=} \int_0^te^{(j_3-1)ks}s^{j_4}\mathbb{E}[I\!E_s^{m_1}I_s^{m_2}I_s^{*m_3-2}I\!E\!Z_s^{i_1+1}I\!Z_s^{i_2}I\!Z_s^{*i_3}|v_0]\mathrm{d} s.
    \end{align*}

Before closing this subsection, we highlight that the final expression for the conditional joint moment
takes the form of a polynomial in :math:`v_0-\theta`. Specifically, it can be expressed as:

.. math::

    \begin{equation}%\label{eqn:ieii-ieziziz-polynomial}
        \mathbb{E}[I\!E_t^{m_1}I_t^{m_2}I_t^{*m_3}I\!E\!Z_t^{m_4}I\!Z_t^{m_5}I\!Z_t^{*m_6}|v_0]
        = \sum_{i=0}^{\lfloor (m_1+m_2)/2 \rfloor + \lfloor m_3/2 \rfloor} c_i (v_0-\theta)^i,
    \end{equation}

where :math:`\lfloor x \rfloor` denotes the floor function, i.e., the greatest integer less than
or equal to :math:`x`. Here, with a slight abuse of notation, :math:`c_i` represents the coefficient,
which can be computed via the recursive equation.
"""
import math
from fractions import Fraction

from ajdmom.poly import Poly
from ajdmom.ito_mom import int_et
from ajdmom.mdl_svcj.ieziziz_mom import moment_ieziziz


def ieziziz_to_ieii_ieziziz(poly):
    # ['e^{kt}', 'e^{ks}', 't-s', 'k^{-}', 'lmbd', 'mu_v', 'mu_s', 'sigma_s']
    # s = 0 to
    kf = ['e^{kt}', 't', 'k^{-}', 'v0-theta', 'theta', 'sigma',
          'lmbd', 'mu_v', 'mu_s', 'sigma_s']
    poln = Poly()
    poln.set_keyfor(kf)
    for k, v in poly.items():
        key = (k[0], k[2], k[3], 0, 0, 0, k[4], k[5], k[6], k[7])
        poln.add_keyval(key, v)
    return poln


def expand_ieziziz(poly):
    # from ['e^{kt}', 'e^{ks}', 't-s', 'k^{-}', 'lmbd', 'mu_v', 'mu_s', 'sigma_s']
    # to ['e^{kt}', 't', 'e^{ks}', 's', 'k^{-}', 'lmbd', 'mu_v', 'mu_s', 'sigma_s']
    poln = Poly()
    poln.set_keyfor(['e^{kt}', 't', 'e^{ks}', 's', 'k^{-}', 'lmbd', 'mu_v', 'mu_s', 'sigma_s'])
    for k, v in poly.items():
        for i in range(k[2] + 1):
            bino = math.comb(k[2], i)
            j = k[2] - i
            key = (k[0], i, k[1], j) + k[3:]
            val = bino * ((-1) ** j) * v
            poln.add_keyval(key, val)
    return poln


def int_e_s_poly(c, tp, m1, m2, poly):
    kf = ['e^{kt}', 't', 'k^{-}', 'v0-theta', 'theta', 'sigma',
          'lmbd', 'mu_v', 'mu_s', 'sigma_s']
    poln = Poly()
    poln.set_keyfor(kf)
    for k, v in poly.items():
        n1 = k[0] + m1
        n2 = k[1] + m2
        # \int_0^t e^{n1 ks} s^{n2} ds
        pol1 = int_et(n1, n2)  # with kf = ('e^{kt}', 't', 'k^{-}')
        for K, V in pol1.items():
            # key = [K[0], K[1], K[2] + k[2], k[3], k[4], k[5], k[6], k[7], k[8], k[9]]
            key = list(K)
            key[2] += k[2]
            key.extend(k[3:])
            if tp in [1, 2, 3]: key[tp + 2] += 1
            val = c * v * V
            poln.add_keyval(tuple(key), val)
    return poln


def key_times_poly(k, poly):
    # k: ['e^{kt}', 't', 'e^{ks}', 's', 'k^{-}',
    #     'lmbd', 'mu_v', 'mu_s', 'sigma_s']
    # poly: ['e^{kt}', 't', 'k^{-}', 'v0-theta', 'theta', 'sigma',
    #     'lmbd', 'mu_v', 'mu_s', 'sigma_s']
    kf = ['e^{kt}', 't', 'k^{-}', 'v0-theta', 'theta', 'sigma',
          'lmbd', 'mu_v', 'mu_s', 'sigma_s']
    poln = Poly()
    poln.set_keyfor(kf)
    for key, val in poly.items():
        knw = (key[0] + k[0],
               key[1] + k[1],
               key[2] + k[4],
               key[3],
               key[4],
               key[5],
               key[6] + k[5],
               key[7] + k[6],
               key[8] + k[7],
               key[9] + k[8])
        poln.add_keyval(knw, val)
    return poln


def recursive_ieii_ieziziz(n1, n2, n3, n4, n5, n6, ieii_ieziziz):
    kf = ['e^{kt}', 't', 'k^{-}', 'v0-theta', 'theta', 'sigma',
          'lmbd', 'mu_v', 'mu_s', 'sigma_s']
    poly = Poly()
    poly.set_keyfor(kf)
    # special case
    if n1 == 0 and n2 == 0 and n3 == 0:
        return ieziziz_to_ieii_ieziziz(moment_ieziziz(n4, n5, n6))  # n4 + n5 + n6 > 0
    # typical cases
    for i1 in range(n4 + 1):
        for i2 in range(n5 + 1):
            for i3 in range(n6 + 1):
                bino1 = math.comb(n4, i1)
                bino2 = math.comb(n5, i2)
                bino3 = math.comb(n6, i3)
                bino = bino1 * bino2 * bino3
                pol1 = expand_ieziziz(moment_ieziziz(n4 - i1, n5 - i2, n6 - i3))  # pol1 can not be empty, i.e., {}
                if len(pol1.keys()) == 0:
                    raise Exception("poly is empty")
                # ['e^{kt}', 't', 'e^{ks}', 's', 'k^{-}', 'lmbd', 'mu_v', 'mu_s', 'sigma_s']
                for k, v in pol1.items():
                    if n1 >= 2 and n2 >= 0 and n3 >= 0:
                        c = bino * Fraction(n1 * (n1 - 1), 2) * v
                        pol2 = int_e_s_poly(c, 1, k[2] + 1, k[3], ieii_ieziziz[(n1 - 2, n2, n3, i1, i2, i3)])
                        pol2 += int_e_s_poly(c, 2, k[2] + 2, k[3], ieii_ieziziz[(n1 - 2, n2, n3, i1, i2, i3)])
                        pol2 += int_e_s_poly(c, 3, k[2] + 1, k[3], ieii_ieziziz[(n1 - 1, n2, n3, i1, i2, i3)])
                        pol2 += int_e_s_poly(c, 4, k[2] + 1, k[3], ieii_ieziziz[(n1 - 2, n2, n3, i1 + 1, i2, i3)])
                        poly.merge(key_times_poly(k, pol2))
                    if n1 >= 0 and n2 >= 2 and n3 >= 0:
                        c = bino * Fraction(n2 * (n2 - 1), 2) * v
                        pol2 = int_e_s_poly(c, 1, k[2] - 1, k[3], ieii_ieziziz[(n1, n2 - 2, n3, i1, i2, i3)])
                        pol2 += int_e_s_poly(c, 2, k[2] - 0, k[3], ieii_ieziziz[(n1, n2 - 2, n3, i1, i2, i3)])
                        pol2 += int_e_s_poly(c, 3, k[2] - 1, k[3], ieii_ieziziz[(n1 + 1, n2 - 2, n3, i1, i2, i3)])
                        pol2 += int_e_s_poly(c, 4, k[2] - 1, k[3], ieii_ieziziz[(n1, n2 - 2, n3, i1 + 1, i2, i3)])
                        poly.merge(key_times_poly(k, pol2))
                    if n1 >= 1 and n2 >= 1 and n3 >= 0:
                        c = bino * n1 * n2 * v
                        pol2 = int_e_s_poly(c, 1, k[2] + 0, k[3], ieii_ieziziz[(n1 - 1, n2 - 1, n3, i1, i2, i3)])
                        pol2 += int_e_s_poly(c, 2, k[2] + 1, k[3], ieii_ieziziz[(n1 - 1, n2 - 1, n3, i1, i2, i3)])
                        pol2 += int_e_s_poly(c, 3, k[2] + 0, k[3], ieii_ieziziz[(n1 + 0, n2 - 1, n3, i1, i2, i3)])
                        pol2 += int_e_s_poly(c, 4, k[2] + 0, k[3], ieii_ieziziz[(n1 - 1, n2 - 1, n3, i1 + 1, i2, i3)])
                        poly.merge(key_times_poly(k, pol2))
                    if n1 >= 0 and n2 >= 0 and n3 >= 2:
                        c = bino * Fraction(n3 * (n3 - 1), 2) * v
                        pol2 = int_e_s_poly(c, 1, k[2] - 1, k[3], ieii_ieziziz[(n1, n2, n3 - 2, i1, i2, i3)])
                        pol2 += int_e_s_poly(c, 2, k[2] - 0, k[3], ieii_ieziziz[(n1, n2, n3 - 2, i1, i2, i3)])
                        pol2 += int_e_s_poly(c, 3, k[2] - 1, k[3], ieii_ieziziz[(n1 + 1, n2, n3 - 2, i1, i2, i3)])
                        pol2 += int_e_s_poly(c, 4, k[2] - 1, k[3], ieii_ieziziz[(n1, n2, n3 - 2, i1 + 1, i2, i3)])
                        poly.merge(key_times_poly(k, pol2))
    return poly


def moment_ieii_ieziziz(n1, n2, n3, n4, n5, n6):
    """joint conditional moment of :math:`IE_t,I_t,I_t^{*},IEZ_t,IZ_t,IZ_t^{*}|v_0`

    :param integer n1: order of :math:`IE_t`
    :param integer n2: order of :math:`I_t`
    :param integer n3: order of :math:`I_t^{*}`
    :param integer n4: order of :math:`IEZ_t`
    :param integer n5: order of :math:`IZ_t`
    :param integer n6: order of :math:`IZ_t^{*}`
    :return: poly with ``keyfor`` = ('e^{kt}','t','k^{-}','v0-theta','theta','sigma',
      'lmbd','mu_v','mu_s','sigma_s')
    :rtype: Poly
    """
    if n1 < 0 or n2 < 0 or n3 < 0 or n4 < 0 or n5 < 0 or n6 < 0:
        raise ValueError(f'moment_ieii_ieziziz({n1},{n2},{n3},{n4},{n5},{n6}) is called!')
    #
    # ieii_ieziziz: a dict of moments of E[IE_t^n1 I_t^n2 I_t^{*n3} IEZ_t^n4 IZ_t^n5 IZ_t^{*n6}]
    #
    ieii_ieziziz = {}
    #
    kf = ['e^{kt}', 't', 'k^{-}', 'v0-theta', 'theta', 'sigma',
          'lmbd', 'mu_v', 'mu_s', 'sigma_s']
    #
    # special poly constants, analog to 0 and 1
    #
    P0 = Poly({(0, 0, 0, 0, 0, 0, 0, 0, 0, 0): Fraction(0, 1)})
    P0.set_keyfor(kf)
    P1 = Poly({(0, 0, 0, 0, 0, 0, 0, 0, 0, 0): Fraction(1, 1)})
    P1.set_keyfor(kf)
    #
    # n1 + n2 + n3 + n4 + n5 + n6 = 0: special case
    #
    ieii_ieziziz[(0, 0, 0, 0, 0, 0)] = P1  # equiv to constant 1
    #
    # n1 + n2 + n3 + n4 + n5 + n6 = 1
    #
    ieii_ieziziz[(1, 0, 0, 0, 0, 0)] = P0  # equiv to constant 0
    ieii_ieziziz[(0, 1, 0, 0, 0, 0)] = P0
    ieii_ieziziz[(0, 0, 1, 0, 0, 0)] = P0
    #
    ieii_ieziziz[(0, 0, 0, 1, 0, 0)] = ieziziz_to_ieii_ieziziz(moment_ieziziz(1, 0, 0))
    ieii_ieziziz[(0, 0, 0, 0, 1, 0)] = ieziziz_to_ieii_ieziziz(moment_ieziziz(0, 1, 0))
    ieii_ieziziz[(0, 0, 0, 0, 0, 1)] = ieziziz_to_ieii_ieziziz(moment_ieziziz(0, 0, 1))
    #
    # n1 + n2 + n3 + n4 + n5 + n6 >= 2
    #
    for n in range(2, n1 + n2 + n3 + n4 + n5 + n6):
        for i1 in range(n + 1):
            for i2 in range(n - i1 + 1):
                for i3 in range(n - i1 - i2 + 1):
                    for i4 in range(n - i1 - i2 - i3 + 1):
                        for i5 in range(n - i1 - i2 - i3 - i4 + 1):
                            i6 = n - i1 - i2 - i3 - i4 - i5
                            poly = recursive_ieii_ieziziz(i1, i2, i3, i4, i5, i6, ieii_ieziziz)
                            # poly.remove_zero()
                            ieii_ieziziz[(i1, i2, i3, i4, i5, i6)] = poly
    # the last one
    poly = recursive_ieii_ieziziz(n1, n2, n3, n4, n5, n6, ieii_ieziziz)
    # poly.remove_zero()
    return poly


def poly2num(poly, par):
    # kf = ['e^{kt}', 't', 'k^{-}', 'v0-theta', 'theta', 'sigma']
    # kf += ['lmbd', 'mu_v', 'mu_s', 'sigma_s']
    v0, k, theta, sigma = par['v0'], par['k'], par['theta'], par['sigma']
    lmbd, mu_v, h = par['lmbd'], par['mu_v'], par['h']
    mu_s, sigma_s = par['mu_s'], par['sigma_s']
    f = 0
    for K, V in poly.items():
        val = math.exp(K[0] * k * h) * (h ** K[1]) / (k ** K[2])
        val *= ((v0 - theta) ** K[3]) * (theta ** K[4]) * (sigma ** K[5])
        val *= (lmbd ** K[6]) * (mu_v ** K[7])
        val *= (mu_s ** K[8]) * (sigma_s ** K[9])
        f += val * V
    return f


def m_ieii_ieziziz(order, par):
    n1, n2, n3, n4, n5, n6 = order
    moment = moment_ieii_ieziziz(n1, n2, n3, n4, n5, n6)
    value = poly2num(moment, par)
    return value


if __name__ == '__main__':
    from pprint import pprint

    poly = moment_ieii_ieziziz(1, 1, 1, 2, 1, 1)
    print("poly = ")
    pprint(poly)
    print(f"which is a poly with keyfor = {poly.keyfor}")

r"""
Joint Moments of CPPs :math:`I\!E\!Z_{s,t}, I\!Z_{s,t}, I\!Z_{s,t}^{*}`

In this module, we present the computation of the joint moments of
(:math:`I\!E\!Z_{s,t}, I\!Z_{s,t}, I\!Z_{s,t}^*`). This is accomplished by deriving the
joint moment-generating function (MGF) of these processes.

The joint MGF of (:math:`I\!E\!Z_{s,t}, I\!Z_{s,t}, I\!Z_{s,t}^*`) can be derived as follows:

.. math:

    \begin{align*}
        M_{I\!E\!Z_{s,t},I\!Z_{s,t},I\!Z_{s,t}^*}(\boldsymbol{a})
        &\mathrel{:=} \mathbb{E}[e^{a_1I\!E\!Z_{s,t} + a_2I\!Z_{s,t} + a_3I\!Z_{s,t}^*}]\\
        %&= \mathbb{E}[\mathbb{E}[e^{a_1 \sum_{i=1}^{N(t-s)}e^{ks_i}J_i + a_2\sum_{i=1}^{N(t-s)}J_i + a_3\sum_{i=1}^{N(t-s)}J_i^*}|N(t-s) = n]]\\
        &= \mathbb{E}[\mathbb{E}[e^{\sum_{i=1}^n [(a_1 e^{ks_i} + a_2)J_i + a_3J_i^*]} |N(t-s) = n]]\\
        &= \mathbb{E}[\mathbb{E}[\left(\mathbb{E}[M_{J_i}(a_1e^{ks_i} + a_2)]\cdot M_{J_i^*}(a_3)\right)^n |N(t-s) = n]]\\
        &= \sum_{n=0}^{\infty} \left(\mathbb{E}[M_{J_i}(a_1e^{ks_i} + a_2)] \cdot M_{J_i^*}(a_3)\right)^n \frac{[\lambda (t-s)]^n e^{-\lambda (t-s)}}{n!}\\
        &= e^{-\lambda (t-s)}  \sum_{n=0}^{\infty}  \left[\lambda (t-s)\mathbb{E}[M_{J_i}(a_1e^{ks_i} + a_2)]\cdot M_{J_i^*}(a_3)\right]^n/n!\\
        %&= e^{-\lambda (t-s)} e^{\lambda (t-s)\mathbb{E}[M_{J_i}(a_1e^{ks_i} + a_2)]\cdot M_{J_i^*}(a_3)}\\
        &= e^{\lambda (t-s)\left(\mathbb{E}[M_{J_i}(a_1e^{ks_i} + a_2)]\cdot M_{J_i^*}(a_3) - 1\right)},
    \end{align*}

where :math:`\boldsymbol{a}` is a vector of real numbers, i.e.,
:math:`\boldsymbol{a} \mathrel{:=} (a_1, a_2, a_3)`, :math:`N(\cdot)` is the shared
counting process for (:math:`I\!E\!Z_{s,t}, I\!Z_{s,t}, I\!Z_{s,t}^*`), and
:math:`M_{J_i}(\cdot)` and :math:`M_{J_i^*}(\cdot)`
are the MGFs of :math:`J_i^*` and :math:`J_i`, respectively. Note that, conditional
on :math:`N(t-s) = n`, the unsorted arrival times are uniformly distributed over the
interval :math:`(s,t]`. For notational simplicity, we use :math:`\{s_1,\dots, s_n\}`
to denote these unsorted arrival times and omit the conditional notation :math:`|N(t-s) = n`
in the expectation. In what follows, we will demonstrate that
:math:`\mathbb{E}[M_{J_i}(a_1e^{ks_i} + a_2)]` admits a closed-form expression.

First, we substitute the MGF of :math:`J_i` (conditioned on :math:`s_i`) with its known
formula:

.. math::

    \begin{align*}
        \mathbb{E}[M_{J_i}(a_1e^{ks_i} + a_2)]
        = \mathbb{E}[\mathbb{E}[e^{(a_1e^{ks_i} + a_2)J_i}|s_i]]
        = \mathbb{E}\left[\frac{1}{1-(a_1e^{ks_i} + a_2 )\mu_v} \right].
    \end{align*}

By introducing two new variables :math:`a \mathrel{:=} - a_1\mu_v, b \mathrel{:=} 1 - a_2\mu_v`,
the above expectation simplifies to:

.. math:

    \begin{align*}
        \mathbb{E}[M_{J_i}(a_1e^{ks_i} + a_2)]
        = \mathbb{E}\left[\frac{1}{ae^{ks_i} + b}\right]
        = \frac{1}{t-s}\int_{s}^t\frac{1}{ae^{ks_i}+b}\mathrm{d} s_i.
    \end{align*}

The integral can be computed explicitly using the variable substitution method. Let us
introduce :math:`x \mathrel{:=} e^{ks_i}`. Then, :math:`s_i = \frac{1}{k}\log x`,
:math:`\mathrm{d} s_i = \frac{1}{k}\frac{1}{x}\mathrm{d} x`.
The integral thus becomes:

.. math:

    \begin{align*}
        \int_{s}^t\frac{1}{ae^{ks_i}+b}\mathrm{d} s_i
        = \int_{e^{ks}}^{e^{kt}} \frac{1}{ax + b} \frac{1}{k} \frac{1}{x} \mathrm{d} x
        = \frac{1}{k}\int_{e^{ks}}^{e^{kt}} \frac{1}{ax + b} \frac{1}{x} \mathrm{d} x.
    \end{align*}

If :math:`a = 0`, i.e., :math:`a_1 = 0`, the integral simplifies to:

.. math:

    \begin{equation*}
        \int_{s}^t\frac{1}{ae^{ks_i}+b}\mathrm{d} s_i = \frac{1}{b}(t-s).
    \end{equation*}

Otherwise, for :math:`a_1\neq 0`, in any neighborhood of the origin of the vector
:math:`(a_1,a_2)` such that  :math:`ax + b \approx 1`, we have:

.. math:

    \begin{align*}
        \int_{e^{ks}}^{e^{kt}} \frac{1}{ax + b} \frac{1}{x} \mathrm{d} x
        %&= \int_{e^{ks}}^{e^{kt}} \left(\frac{-a/b}{ax + b} + \frac{1/b}{x}\right) dx\\
        %&= -\frac{1}{b}\int_{e^{ks}}^{e^{kt}} \frac{1}{ax+b}\mathrm{d}(ax+b) + \frac{1}{b}\int_{e^{ks}}^{e^{kt}}\frac{1}{x}\mathrm{d} x\\
        &= -\frac{1}{b}\left[\log(ae^{kt} + b) - \log(ae^{ks} + b)\right] + \frac{1}{b}k(t-s).
    \end{align*}

Therefore, the integral evaluates to:

.. math:

    \begin{align*}
        \int_{s}^t\frac{1}{ae^{ks_i}+b}\mathrm{d} s_i
        &= -\frac{1}{kb}\left[\log(ae^{kt} + b) - \log(ae^{ks} + b)\right] + \frac{1}{b}(t-s),
    \end{align*}

provided that :math:`(a_1,a_2)` lies within a sufficiently small neighborhood of
the origin :math:`(0,0)`. Finally, we obtain the closed-form expression for
:math:`\mathbb{E}[M_{J_i}(a_1e^{ks_i} + a_2)]` as:

.. math:

    \begin{equation}
        \mathbb{E}[M_{J_i}(a_1e^{ks_i} + a_2)]
        = \frac{1}{b}\left[1 -\frac{1}{k(t-s)}\left(\log(ae^{kt} + b) - \log(ae^{ks} + b)\right)\right],
    \end{equation}

where :math:`a = -a_1\mu_v`, :math:`b = 1 - a_2\mu_v`, and :math:`(a_1,a_2)` is restricted
to a small neighborhood around the origin :math:`(0,0)`.

To simplify the notation, we define a new function :math:`M_{E\!J,J,J^*}(\cdot)` as the
following product:

.. math:

    \begin{equation*}
        M_{E\!J,J,J^*}(\boldsymbol{a}) \mathrel{:=} \mathbb{E}[M_{J_i}(a_1e^{ks_i} + a_2)]\cdot M_{J_i^*}(a_3).
    \end{equation*}

We know that the MGF of :math:`J_i^*` has the following expression:

.. math:

    \begin{equation}
        M_{J_i^*}(a_3) = e^{\mu_s a_3 + \sigma_s^2a_3^2/2}
    \end{equation}

since :math:`J_I^*` is normally distributed with mean :math:`\mu_s` and variance
:math:`\sigma_s^2`. Combining these results, we obtain the following closed-form
expression for the joint MGF of :math:`(I\!E\!Z_{s,t}, I\!Z_{s,t}, I\!Z_{s,t}^*)`:

.. math:

    \begin{equation}
        M_{I\!E\!Z_{s,t},I\!Z_{s,t},I\!Z_{s,t}^*}(\boldsymbol{a})
        = e^{\lambda (t-s) (M_{E\!J,J,J^*}(\boldsymbol{a})-1)},
    \end{equation}

where :math:`\boldsymbol{a} = (a_1,a_2,a_3)`, :math:`a = -a_1\mu_v`, :math:`b = 1-a_2\mu_v`,
and

.. math:

    \begin{equation*}
        M_{E\!J,J,J^*}(\boldsymbol{a})
        = \frac{1}{b} \left[ 1 - \frac{1}{k(t-s)}\left(\log(ae^{kt} + b) - \log(ae^{ks} + b)\right) \right] e^{\mu_s a_3 + \sigma_s^2 a_3^2/2}.
    \end{equation*}

Given the joint MGF of :math:`(I\!E\!Z_{s,t}, I\!Z_{s,t}, I\!Z_{s,t}^*)`, we can compute
the joint moment of :math:`(I\!E\!Z_{s,t}, I\!Z_{s,t},I\!Z_{s,t}^*)` of any order.

The :math:`n`-th (:math:`n\ge1`) partial derivative of :math:`M_{E\!J,J,J^*}(\boldsymbol{a})`
with respect to :math:`a_1` is given by

.. math:

    \begin{equation*}
        \frac{\partial^nM_{E\!J,J,J^*}}{\partial a_1^{n}}
        =  \frac{1}{b}\left[\frac{e^{nkt}}{(ae^{kt}+b)^{n}}  - \frac{e^{nks}}{(ae^{ks}+b)^{n}} \right] \frac{(n-1)! \mu_v^n}{k(t-s)} M_{J_i^*}(a_3).
    \end{equation*}

Consequently, the :math:`n`-th moment of :math:`E\!J` can be expressed as

.. math:

    \begin{equation}
        \mathbb{E}[(e^{ks_i}J_i)^n] = (e^{nkt} - e^{nks})\frac{1}{k(t-s)}(n-1)!\mu_v^n.
    \end{equation}


For :math:`n_1\ge 1` and :math:`n_2\ge 1`, the following formula holds:

.. math:

    \begin{align}
        \frac{\partial^{n_1+n_2}M_{E\!J,J,J^*}}{\partial a_1^{n_1}\partial a_2^{n_2}}
        &= \sum_{i=0}^{n_2} \frac{c(n_1,n_2,i)}{b^{n_2-i+1}}\left[\frac{e^{n_1kt}}{(ae^{kt}+b)^{n_1+i}} - \frac{e^{n_1ks}}{(ae^{ks}+b)^{n_1+i}} \right] \frac{\mu_v^{n_1+n_2}}{k(t-s)} M_{J_i^*}(a_3),
    \end{align}

where :math:`c(n_1,n_2,i) \mathrel{:=} n_2!(n_1 - 1 + i)!/(i!)`. An alternative
approach involves directly computing the joint moment directly, yielding:

.. math:

    \begin{equation}
        \mathbb{E}[(e^{ks_i}J_i)^{n_1}J_i^{n_2}]
        = \mathbb{E}[e^{n_1ks_i}]\mathbb{E}[J_i^{n_1+n_2}]
        = \frac{1}{n_1k(t-s)}(e^{n_1kt} - e^{n_1ks})(n_1+n_2)!\mu_v^{n_1+n_2}.
    \end{equation}

For :math:`n\ge 1`, we have:

.. math:

    \begin{align*}
        \frac{\partial^{n}M_{E\!J,J,J^*}}{\partial a_2^{n}}
        &= \frac{n!}{b^{n+1}}  \left[ 1 - \frac{1}{k(t-s)}\left(\log(ae^{kt} + b) - \log(ae^{ks} + b)\right) \right]\mu_v^n M_{J_i^*}(a_3)\\
        &\quad + \sum_{i=1}^n\binom{n}{i}\frac{(n-i)!(i-1)!}{b^{n-i+1}} \left[\frac{1}{(ae^{kt}+b)^i} - \frac{1}{(ae^{ks} + b)^i}\right]\frac{1}{k(t-s)} \mu_v^n M_{J_I^*}(a_3)\\
        &= \frac{n!}{b^{n+1}}  \left[ 1 - \frac{1}{k(t-s)}\left(\log(ae^{kt} + b) - \log(ae^{ks} + b)\right) \right]\mu_v^n M_{J_i^*}(a_3)\\
        &\quad + \sum_{i=1}^n\frac{n!/i}{b^{n-i+1}} \left[\frac{1}{(ae^{kt}+b)^i} - \frac{1}{(ae^{ks} + b)^i}\right]\frac{1}{k(t-s)} \mu_v^n M_{J_I^*}(a_3).
    \end{align*}

Thus, the :math:`n`-th moment of :math:`J` is given by

.. math:

    \begin{equation}
        \mathbb{E}[J_i^n] = n!\mu_v^n.
    \end{equation}

"""
import math
from fractions import Fraction

from ajdmom import Poly
from ajdmom.cpp_mom import mnorm


def moment_ejjj(n1, n2, n3):
    if n1 < 0 or n2 < 0 or n3 < 0:
        raise ValueError('n1 and n2 and n3 must be non-negative integers!')
    poly = Poly()
    kf = ['e^{kt}', 'e^{ks}', 't-s', 'k^{-}', 'lmbd', 'mu_v']
    poly.set_keyfor(kf)
    if n1 >= 1 and n2 == 0:
        key = (n1, 0, -1, 1, 0, n1)
        val = Fraction(math.factorial(n1 - 1), 1)
        poly.add_keyval(key, val)
        key = (0, n1, -1, 1, 0, n1)
        poly.add_keyval(key, -val)
    if n1 >= 1 and n2 >= 1:
        key = (n1, 0, -1, 1, 0, n1 + n2)
        val = Fraction(math.factorial(n1 + n2), n1)
        poly.add_keyval(key, val)
        key = (0, n1, -1, 1, 0, n1 + n2)
        poly.add_keyval(key, -val)
    if n1 == 0 and n2 >= 1:
        key = (0, 0, 0, 0, 0, n2)
        val = Fraction(math.factorial(n2), 1)
        poly.add_keyval(key, val)
    if n1 == 0 and n2 == 0:
        poly.add_keyval((0, 0, 0, 0, 0, 0), Fraction(1, 1))
    # multiply with M_J^*(n3)
    kf.extend(['mu_s', 'sigma_s'])
    poln = Poly()
    poln.set_keyfor(kf)
    pol1 = mnorm(n3)  # kf = ['mu_s', 'sigma_s^2']
    for k, v in poly.items():
        for K, V in pol1.items():
            poln.add_keyval(k + (K[0], 2 * K[1]), v * V)
    return poln

def d1_times_key(key, wrt):
    # key = (i, (n11, n21, n31, m1), ..., (n1l, n2l, n3l, ml))
    # multiply with [lambda (t-s)]
    knw = list(key)
    knw[0] += 1
    # multiply with \partial M_{E\!J,J,J^*} / \partial a_o (o = 1, 2, 3)
    if wrt == 1:
        n1n2n3 = (1, 0, 0)
    elif wrt == 2:
        n1n2n3 = (0, 1, 0)
    else:
        n1n2n3 = (0, 0, 1)
    #
    found = False
    for i in range(1, len(knw)):
        if knw[i][0:3] == n1n2n3:
            knw[i] = n1n2n3 + (knw[i][3] + 1,)
            found = True
            break
    if not found:
        knw.insert(1, n1n2n3 + (1,))
    return tuple(knw)

def dterm(key, coef, wrt):
    poly = Poly()
    # key = (i, (n11, n21, n31, m1), ..., (n1l, n2l, n3l, ml))
    for j in range(1, len(key)):
        knw = list(key)
        n1, n2, n3, m = key[j]
        coef_new = coef * m  # update coef
        # update exponent M^{(n1,n2,n3)m} -> M^{(n1,n2,n3)(m-1)}
        knw[j] = (n1, n2, n3, m - 1)
        # M^{(n1+1, n2, n3)1} or M^{(n1, n2+1, n3)1} or M^{(n1, n2, n3+1)1}
        if wrt == 1:
            n1 += 1
        elif wrt == 2:
            n2 += 1
        else:
            n3 += 1
        found = False
        for i in range(1, len(knw)):
            if knw[i][0:3] == (n1, n2, n3):
                knw[i] = knw[i][0:3] + (knw[i][3] + 1,)
                found = True
                break
        if not found:
            knw.append((n1, n2, n3, 1))
        if knw[j][3] == 0: del knw[j]  # delete M^{(n1,n2,n3)0}
        poly.add_keyval(tuple(knw), coef_new)
    return poly

def dmgf_ieziziz(poly, wrt):
    # step 1
    poly_sum = Poly()
    for key in poly:
        knw = d1_times_key(key, wrt)
        poly_sum.add_keyval(knw, poly[key])
    # step 2
    for key in poly:
        poly_sum.merge(dterm(key, poly[key], wrt))
    return poly_sum

def decode(poly):
    # poly with kf = ['(i, (n11, n21, n31, m1), ..., (n1l, n2l, n3l, ml))']
    kf = ['e^{kt}', 'e^{ks}', 't-s', 'k^{-}', 'lmbd', 'mu_v', 'mu_s', 'sigma_s']
    poln = Poly()
    poln.set_keyfor(kf)
    for k in poly:
        i = k[0]
        pol1 = Poly({(0, 0, 0, 0, 0, 0, 0, 0): Fraction(1, 1)})
        pol1.set_keyfor(kf)
        for j in range(1, len(k)):
            n1, n2, n3, m = k[j]
            pol1 = pol1 * (moment_ejjj(n1, n2, n3) ** m)
        for key in pol1:
            knw = list(key)
            knw[2] += i
            knw[4] += i
            val = poly[k] * pol1[key]
            poln.add_keyval(tuple(knw), val)
    poln.remove_zero()
    return poln

def moment_ieziziz(n1, n2, n3):
    """joint moment of :math:`E[IEZ_{s,t}^{n_1} IZ_{s,t}^{n_2} IZ_{s,t}^{*n_3}]`

    :param integer n1: order of :math:`IEZ_{s,t}`
    :param integer n2: order of :math:`IZ_{s,t}`
    :param integer n3: order of :math:`IZ_{*s,t}`
    :return: poly with ``keyfor`` = ('e^{kt}', 'e^{ks}', 't-s', 'k^{-}', 'lmbd',
      'mu_v', 'mu_s', 'sigma_s')
    :rtype: Poly
    """
    if n1 < 0 or n2 < 0 or n3 < 0:
        raise ValueError('n1 and n2 and n3 must be non-negative integers!')
    kf0 = ['e^{kt}', 'e^{ks}', 't-s', 'k^{-}', 'lmbd', 'mu_v', 'mu_s', 'sigma_s']
    M = {}
    kf = ['(i, (n11, n21, n31, m1), ..., (n1l, n2l, n3l, ml))']
    # n1 = n2 = n3 = 0
    poly = Poly({(0,): Fraction(1, 1)})
    poly.set_keyfor(kf)
    M[(0, 0, 0)] =poly
    #
    if n1 == 0 and n2 == 0 and n3 == 0:
        poln = Poly({(0, 0, 0, 0, 0, 0, 0, 0): Fraction(1, 1)})
        poln.set_keyfor(kf0)
        return poln
    # n1 + n2 + n3 = 1
    poly = Poly({(1, (1, 0, 0, 1)): Fraction(1, 1)})
    poly.set_keyfor(kf)
    M[(1, 0, 0)] = poly
    #
    poly = Poly({(1, (0, 1, 0, 1)): Fraction(1, 1)})
    poly.set_keyfor(kf)
    M[(0, 1, 0)] = poly
    #
    poly = Poly({(1, (0, 0, 1, 1)): Fraction(1, 1)})
    poly.set_keyfor(kf)
    M[(0, 0, 1)] = poly
    #
    if n1 == 1 and n2 == 0 and n3 == 0:
        poln = Poly({
            (1, 0, 0, 1, 1, 1, 0, 0): Fraction(1, 1),
            (0, 1, 0, 1, 1, 1, 0, 0): Fraction(-1, 1)
        })
        poln.set_keyfor(kf0)
        return poln
    if n1 == 0 and n2 == 1 and n3 == 0:
        poln = Poly({(0, 0, 1, 0, 1, 1, 0, 0): Fraction(1, 1)})
        poln.set_keyfor(kf0)
        return poln
    if n1 == 0 and n2 == 0 and n3 == 1:
        poln = Poly({(0, 0, 1, 0, 1, 0, 1, 0): Fraction(1, 1)})
        poln.set_keyfor(kf0)
        return poln
    # n1 + n2 + n3 > 1
    for n in range(2, n1 + n2 + n3):
        for i in range(n + 1):
            for j in range(n - i + 1):
                # (i, j, n-i-j)
                if i > 0 and j == 0:
                    M[(i, j, n-i-j)] = dmgf_ieziziz(M[(i-1, j, n-i-j)], 1)
                elif i == 0 and j > 0:
                    M[(i, j, n-i-j)] = dmgf_ieziziz(M[(i, j-1, n-i-j)], 2)
                elif i == 0 and j == 0:
                    M[(i, j, n-i-j)] = dmgf_ieziziz(M[(i, j, n-i-j-1)], 3)
                else:  # n1 > 0 and n2 > 0
                    M[(i, j, n-i-j)] = dmgf_ieziziz(M[(i, j-1, n-i-j)], 2)
    if n1 > 0 and n2 == 0:
        poln = dmgf_ieziziz(M[(n1-1, n2, n3)], 1)
    elif n1 == 0 and n2 > 0:
        poln = dmgf_ieziziz(M[(n1, n2-1, n3)], 2)
    elif n1 == 0 and n2 == 0:
        poln = dmgf_ieziziz(M[(n1, n2, n3-1)], 3)
    else:  # n1 > 0 and n2 > 0
        poln = dmgf_ieziziz(M[(n1, n2-1, n3)], 2)
    return decode(poln)

def poly2num(poly, par):
    # kf = ['e^{kt}', 'e^{ks}', 't-s', 'k^{-}', 'lmbd', 'mu_v', 'mu_s', 'sigma_s']
    # s = 0
    k, h, lmbd = par['k'], par['h'], par['lmbd']
    mu_v, mu_s, sigma_s = par['mu_v'], par['mu_s'], par['sigma_s']
    #
    f = 0
    for K, V in poly.items():
        val = math.exp(K[0] * k * h) * (h ** K[2]) / (k ** K[3])
        val *= (lmbd ** K[4]) * (mu_v ** K[5]) * (mu_s ** K[6])
        val *= sigma_s ** K[7]
        f += val * V
    return f

def m_ieziziz(order, par):
    n1, n2, n3 = order
    moment = moment_ieziziz(n1, n2, n3)
    value = poly2num(moment, par)
    return value

if __name__ == "__main__":
    from pprint import pprint

    n = 4
    for n1 in range(n):
        for n2 in range(n):
            for n3 in range(n):
                poly = moment_ieziziz(n1, n2, n3)
                print(f"moment_ieziziz({n1}, {n2}, {n3}) = ")
                pprint(poly)
                print(f"which is a poly with keyfor = {poly.keyfor}\n")

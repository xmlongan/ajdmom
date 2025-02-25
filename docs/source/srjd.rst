=====================================================================
:abbr:`SRJD(Square-Root Jump Diffusion)` Model
=====================================================================

In this subpackage (``ajdmom.mdl_srjd``), we consider the following 
Square-Root Jump-Diffusion process: 

.. math::
   
   dv(t) = k(\theta - v(t))dt + \sigma_v\sqrt{v(t)}dw^v(t) + dz(t),
   
which is the same as that in :doc:`../generated/ajdmom.ito_cond_mom`, details
can be referred to therein.

Please note that
we write :math:`v_t, z_t, v_0` and :math:`v(t), z(t), v(0)` interchangeably.

Conditional Moments - I
========================

The solution to the SRJD process can be rewritten as:

.. math::

   e^{kt}(v_t-\theta) = (v_0-\theta) + \sigma_v I\!E_t + I\!E\!Z_t.

The first unconditional moment is calculated as :math:`\mathbb{E}[v] = \theta + \lambda\mu_v/k,`
since :math:`\mathbb{E}[I\!E\!Z_t] = \lambda\mu_v (e^{kt}-1)/k`, :math:`\mathbb{E}[I\!E_t] = 0`
and :math:`\mathbb{E}[v_t] = \mathbb{E}[v_0] = \mathbb{E}[v]`. This result allows us to rewrite
solution to the SRJD in the following form:

.. math::

   e^{kt}(v_t-\mathbb{E}[v]) = \sigma_vI\!E_t + \overline{I\!E\!Z}_t + (v_0-\mathbb{E}[v]),

where :math:`\overline{I\!E\!Z}_t \mathrel{:=} I\!E\!Z_t - \mathbb{E}[I\!E\!Z_t]` represents
the centralized term. This centralized term can be decomposed similarly:
:math:`\overline{I\!E\!Z}_t = \overline{I\!E\!Z}_{s} + \overline{I\!E\!Z}_{s,t}` where
:math:`\overline{I\!E\!Z}_{s,t} \mathrel{:=} I\!E\!Z_{s,t} - \mathbb{E}[I\!E\!Z_{s,t}]`. It is
straightforward to verify that :math:`\mathbb{E}[\overline{I\!E\!Z}_{s,t}^m]` can be expressed
as a "polynomial":

.. math::

   \mathbb{E}[\overline{I\!E\!Z}_{s,t}^m]
   = \sum_{\boldsymbol{j}} c_{\boldsymbol{j}} e^{j_1kt}e^{j_2ks}k^{-j_3}\lambda^{j_4}\mu_v^{j_5},

where, with a slight abuse of notation, :math:`\boldsymbol{j}\mathrel{:=} (j_1,\dots,j_5)`,
and :math:`c_{\boldsymbol{j}}` denotes the associated coefficient for the corresponding monomial.
Therefore, the conditional joint moment :math:`\mathbb{E}[I\!E_t^{m_1}\overline{I\!E\!Z}_t^{m_2}|v_0]`
can be computed using the following recursive equation:

.. math::

   \mathbb{E}[I\!E_t^{m_1}\overline{I\!E\!Z}_t^{m_2}|v_0]
    = \sum_{i=0}^{m_2}\binom{m_2}{i}\sum_{\boldsymbol{j}}c_{\boldsymbol{j}}
    e^{j_1kt}k^{-j_3}\lambda^{j_4}\mu_v^{j_5}P(m_1,m_2),

where :math:`P(m_1,m_2) \mathrel{:=} [m_1(m_1-1)/2]\cdot(p_1 + p_2 + p_3 + p_4)`, and

.. math::

   \begin{align*}
     p_1 &\mathrel{:=} \int_0^t e^{(j_2+1)ks}\mathbb{E}[I\!E_s^{m_1-2}\overline{I\!E\!Z}_s^i|v_0] \mathrm{d} s \times (v_0-\mathbb{E}[v]),\\
     p_2 &\mathrel{:=} \int_0^t e^{(j_2+2)ks}\mathbb{E}[I\!E_s^{m_1-2}\overline{I\!E\!Z}_s^i|v_0] \mathrm{d} s \times \mathbb{E}[v],\\
     p_3 &\mathrel{:=} \int_0^t e^{(j_2+1)ks}\mathbb{E}[I\!E_s^{m_1-1}\overline{I\!E\!Z}_s^i|v_0] \mathrm{d} s \times \sigma_v,\\
     p_4 &\mathrel{:=} \int_0^t e^{(j_2+1)ks}\mathbb{E}[I\!E_s^{m_1-2}\overline{I\!E\!Z}_s^{i+1}|v_0] \mathrm{d} s.
   \end{align*}

For the special case :math:`m_1=1`, it is easy to find that
:math:`\mathbb{E}[I\!E_t\overline{I\!E\!Z}_t^{m_2}|v_0] = 0`.

With the preparations outlined above, the :math:`m`-th conditional central moment of :math:`v_t`
can be calculated as:

.. math::

   e^{mkt}\mathbb{E}[(v_t-\mathbb{E}[v])^m|v_0]
   = \sum_{m_1+m_2+m_3=m}\binom{m}{m_1,m_2,m_3}\mathbb{E}[I\!E_t^{m_1}
   \overline{I\!E\!Z}_t^{m_2}|v_0]\sigma_v^{m_1}(v_0-\mathbb{E}[v])^{m_3}.

We note that the conditional central moment :math:`\mathbb{E}[(v_t-\mathbb{E}[v])^m|v_0]` will be
computed as a polynomial in :math:`(v_0-\mathbb{E}[v])`:

.. math::

   \mathbb{E}[(v_t-\mathbb{E}[v])^m|v_0]
    = c_m(v_0-\mathbb{E}[v])^m + c_{m-1}(v_0-\mathbb{E}[v])^{m-1} + \cdots + c_1(v_0-\mathbb{E}[v]) + c_0,

where :math:`c_{m},\dots, c_0` are coefficients, some of which may be zero and :math:`c_m = e^{-mkt}`.
The reason is that the conditional joint moment :math:`\mathbb{E}[I\!E_t^{m_1}\overline{I\!E\!Z}_t^{m_2}|v_0]`
produces a polynomial in :math:`(v_0 - \mathbb{E}[v])` of order at most :math:`\lfloor m_1 / 2 \rfloor`.


Moments
======================

We further note that the unconditional central moment of :math:`v_t` can be computed via:

.. math::

   \mathbb{E}[(v_t-\mathbb{E}[v])^m] = \mathbb{E}[\mathbb{E}[(v_t-\mathbb{E}[v])^m|v_0]].

Meanwhile, due to the assumption that :math:`v_t` is strictly stationary, we have

.. math::

   \mathbb{E}[(v_t-\mathbb{E}[v])^m] = \mathbb{E}[(v_0-\mathbb{E}[v])^m].

Thus, the :math:`m`-th unconditional central moment of :math:`v_t` can be computed using the following
recursive equation:

.. math::

   (1-e^{-mkt})\mathbb{E}[(v_0-\mathbb{E}[v])^m]
   = c_{m-1}\mathbb{E}[(v_0-\mathbb{E}[v])^{m-1}] + \cdots + c_1\mathbb{E}[(v_0-\mathbb{E}[v])] + c_0.

For example, the second central moment is computed as:

.. math::

   \mathbb{E}[(v_0-\mathbb{E}[v])^2] = \frac{\lambda\mu_v^2}{k} + \frac{\mathbb{E}[v]\sigma_v^2}{2k},

and the third central moment:

.. math::

   \mathbb{E}[(v_0-\mathbb{E}[v])^3]
   = \frac{2\lambda\mu_v^3}{k} + \frac{\sigma_v^2\lambda\mu_v^2}{k^2} + \frac{\mathbb{E}[v]\sigma_v^4}{2k^2}.

Using the above recursive equation, we can compute the fourth and any higher central moments recursively.
Given the central moments, the corresponding non-central moments can be easily computed.


Conditional Moments - II
=========================

Given :math:`v_0` and :math:`z_{s}, 0\le s \le t`,

.. math::
  
  e^{kt}v_t = \mu_{ev} + \sigma_v I\!E_t,

where :math:`\mu_{ev} \triangleq (v_0-\theta) + \theta e^{kt} + I\!E\!Z_t`.
Thus, we have

.. math::
   
   \begin{align*}
   &\mathbb{E}[(e^{kt}v_t)^m|v_0, z_s, 0\le s \le t] \\
   &\quad = \sum_{j=0}^mC_m^j \mu_{ev}^j \sigma_v^{m-j} 
   \mathbb{E}[I\!E_t^{m-j}|v_0, z_s, 0\le s \le t],
   \end{align*}

further,

.. math::
   
   \begin{align*}
   &\mathbb{E}[v_t^m|v_0, z_s, 0\le s \le t] \\
   &= e^{-mkt} \sum_{j=0}^mC_m^j \mu_{ev}^j \sigma_v^{m-j}
   \mathbb{E}[I\!E_t^{m-j}|v_0, z_s, 0\le s \le t].
   \end{align*}

We have,  :math:`\forall m \ge 2`,

.. math::
  :label: srjd-IE-moment
  
  \begin{align*}
  &\mathbb{E}[I\!E_t^m|v_0, z(s), 0\le s \le t]\\
  &=  \frac{1}{2}m(m-1)(v_0-\theta)\int_0^te^{ks}\mathbb{E}[I\!E_s^{m-2}
    |v_0, z(s), 0\le s \le t]ds\\
  &\quad + \frac{1}{2}m(m-1)\theta\quad~~~  \int_0^te^{2ks}\mathbb{E}
  [I\!E_s^{m-2}|v_0, z(s), 0\le s \le t]ds\\
  &\quad + \frac{1}{2}m(m-1)\quad~~~ \int_0^t(e^{ks}I\!E\!Z_s)\mathbb{E}
  [I\!E_s^{m-2}|v_0, z(s), 0\le s \le t]ds\\
  &\quad + \frac{1}{2}m(m-1)\sigma_v\quad \int_0^te^{ks}\mathbb{E}
  [I\!E_s^{m-1}|v_0, z(s), 0\le s \le t]ds,
  \end{align*}

where :math:`\mathbb{E}[I\!E_t^0|v_0, z(s), 0\le s \le t] = 1` and
:math:`\mathbb{E}[I\!E_t|v_0, z(s), 0\le s \le t] = 0`.

We decode :math:`\mathbb{E}[I\!E_t^m|v_0, z(s), 0\le s \le t]` as 
a ``Poly`` object of the following form:

.. math::
   
   \begin{align*}
   &\mathbb{E}[I\!E_t^m|v_0, z(s), 0\le s \le t]\\
   &\equiv \sum_{j_{1:5}, l_{1:n}, o_{2:n} } c_{j_{1:5}, l_{1:n}, o_{2:n} }
   v_0^{j_1} k^{-j_2} \theta^{j_3} \sigma_v^{j_4} e^{j_5kt} 
   f_{Z_t}(l_{1:n}, o_{2:n}),
   \end{align*}

where :math:`\forall n >= 2`

.. math::
  :label: fZ_IE
   
   \begin{align*}
   f_{Z_t}(l_{1:n}, o_{2:n}) 
   &\triangleq\sum_{i_1=1}^{N(t)}\cdots 
   \sum_{i_n=1}^{N(t)} e^{l_1ks_{i_1} + \cdots + l_nks_{i_n}} 
   J_{i_1}\cdots J_{i_n} \\
   &\quad \cdot e^{o_2 k (s_{i_1}\vee s_{i_2}) + \cdots 
   + o_{n} k (s_{i_1} \vee\cdots \vee s_{i_n}) },
   \end{align*}

and for :math:`n=1`, :math:`f_{Z_t}(l_{1:n}, o_{2:n})` degenerates into
:math:`f_{Z_t}(l_{1:1}) = \sum_{i_1=1}^{N(t)} e^{l_1 k s_{i_1}}J_{i_1}`.
Lastly, when :math:`n=0`, :math:`f_{Z_t}(l_{1:n}, o_{2:n}) = 1`.

The conditional moments of :math:`I\!E_t` and :math:`v(t)` are implemented
in :py:func:`~ajdmom.mdl_srjd.mom.moment_IE` and 
:py:func:`~ajdmom.mdl_srjd.mom.moment_v`, respectively, in the subpackage
``ajdmom.mdl_srjd``.

For the conditional central moments, define
:math:`\overline{v}(t)\triangleq v(t)-\mathbb{E}[v(t)|v_0, z(s), 0\le s\le t]`,
thus :math:`\overline{v}(t) \equiv e^{-kt}\sigma_vI\!E_t`.

.. math::
   
   \begin{align*}
   &\mathbb{E}[\overline{v}^m(t)|v_0, z(s), 0\le s\le t]\\
   &= e^{-mkt}\sigma_v^m \mathbb{E}[I\!E_t^m|v_0, z(s), 0\le s\le t].
   \end{align*}

The conditional central moments are implemented in 
:py:func:`~ajdmom.mdl_srjd.cmom.cmoment_v` in the subpackage 
``ajdmom.mdl_srjd``.


API
====

.. autosummary::
  :toctree: generated
   
   ajdmom.mdl_srjd.mom
   ajdmom.mdl_srjd.cmom
   ajdmom.mdl_srjd.cond_mom
   ajdmom.mdl_srjd.cond_cmom
   ajdmom.mdl_srjd.cond2_mom
   ajdmom.mdl_srjd.cond2_cmom

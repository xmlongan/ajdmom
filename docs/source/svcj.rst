=====================================================================
:abbr:`SVCJ(Stochastic Volatility with Contemporaneous Jumps)` Model
=====================================================================

In this subpackage (``ajdmom.mdl_svcj``), we consider the following 
:abbr:`SV(Stochastic Volatility)` model, which adds contemporaneous jumps
in the price and variance of the Heston model:

.. math::
   
   d\log s(t) &= (\mu- v(t)/2) dt + \sqrt{v(t)}dw^s(t) + dz^s(t),\\
   dv(t)      &= k(\theta - v(t))dt + \sigma_v \sqrt{v(t)}dw^v(t) + dz^v(t),

where 

- :math:`z^v(t)` is a :abbr:`CPP(Compound Poisson Process)` with 
  constant arrival rate :math:`\lambda_v` and jumps (:math:`J_i^v`) distributed
  according to an exponential distribution with scale parameter 
  :math:`\mu_v` (= 1/rate), i.e.,
  :math:`J_i^v \sim \text{exp}(\mu_v)`
  (which will help to make sure the variance always be non-negative, 
  i.e., :math:`v(t) \ge 0, \forall t\ge 0`).

- :math:`z^s(t)` is another :abbr:`CPP(Compound Poisson Process)` sharing
  the same arrival process with :math:`z^v(t)`, and with jumps distributed 
  according to a normal distribution with mean :math:`\mu_s + \rho_J J_i^v` 
  (dependent on the realized jump :math:`J_i^v` in the varance process) and
  variance :math:`\sigma_s^2`, i.e., 
  :math:`J_i^s|J_i^v \sim \mathcal{N}(\mu_s+\rho_J J_i^v, \sigma_s^2)`.

For models including jumps in the variance, both conditional and unconditional
moments can be derived.

Conditional Moments - I
=========================

We now introduce the following notation:

.. math::

    \begin{align*}
        I\!E_t &\mathrel{:=} \int_0^t e^{ks}\sqrt{v(s)}\mathrm{d} w^v(s),
        &I_t    &\mathrel{:=} \int_0^t \sqrt{v(s)} \mathrm{d} w^v(s),\\
        I\!E\!Z_t &\mathrel{:=} \int_0^t e^{ks}\mathrm{d} z^v(s), ~
        &I\!Z_t &\mathrel{:=} \int_0^t \mathrm{d} z^v(s).
    \end{align*}

Additionally, we define

.. math::

   \quad I\!Z_t^s \mathrel{:=} \int_0^t \mathrm{d} z^s(t), \quad I\!Z_t^* \mathrel{:=} \int_0^t\mathrm{d} z^*(t),

where :math:`z^*(t)` is another compound Poisson process sharing the same arrival process as
:math:`z^v(t)` and :math:`z^s(t)`, but with independent jumps :math:`J_i^*\sim \mathcal{N}(\mu_s, \sigma_s^2)`.
Since the jumps of :math:`z^s(t)` are distributed as
:math:`J_i^s|J_i^v \sim \mathcal{N}(\mu_s + \rho_J J_i^v, \sigma_s^2)`, the compound Poisson process
in the return can be decomposed into another two compound Poisson processes, i.e.,
:math:`I\!Z_{t}^s = \rho_J I\!Z_t + I\!Z_t^*`. The solution to the variance process now is given by:

.. math::

    \begin{equation*}
        e^{kt}v_t = (v_0-\theta) + e^{kt}\theta + \sigma_v I\!E_t + I\!E\!Z_t.
    \end{equation*}

For the return :math:`y_t` (:math:`y_t \mathrel{:=} p(t) - p(0)`), we have the following
decomposition:

.. math::

    \begin{align*}
        y_t
        &= \frac{\sigma_v}{2k} e^{-kt}I\!E_t +
         \left(\rho -\frac{\sigma_v}{2k} \right)I_t + \sqrt{1-\rho^2}I_t^{*}
         + \frac{1}{2k}e^{-kt}I\!E\!Z_t + \left(\rho_J - \frac{1}{2k}\right)I\!Z_t + I\!Z_t^{*}\\
        &\quad + \left(\mu-\frac{\theta}{2}\right)t - (v_0 - \theta)\beta_t,
    \end{align*}

where :math:`\beta_t = (1-e^{-kt})/(2k)` by definition.
The :math:`m`-th conditional moment of :math:`y_t`, given :math:`v_0`, can be derived through

.. math::

    \begin{align*}
        \mathbb{E}[y_t^m|v_0]
        &= \sum_{m_1+\cdots+m_8=m}c_{\boldsymbol{m}} b_{\boldsymbol{m}}
        \mathbb{E}[I\!E_t^{m_1}I_t^{m_2}I_t^{*m_3}I\!E\!Z_t^{m_4}I\!Z_t^{m_5}I\!Z_t^{*m_6}|v_0],
    \end{align*}

where :math:`\boldsymbol{m} \mathrel{:=} (m_1, \dots, m_8)`,
:math:`c(\boldsymbol{m}) \mathrel{:=} \binom{m}{m_1,\dots,m_8}` and

.. math::

    \begin{align*}
        &b(\boldsymbol{m}) \\
        &\mathrel{:=} \frac{e^{-(m_1+m_4)kt}}{(2k)^{m_1+m_4}}\sigma_v^{m_1}\left(\rho - \frac{\sigma_v}{2k}\right)^{m_2}\left(\sqrt{1-\rho^2}\right)^{m_3}\left(\rho_J - \frac{1}{2k} \right)^{m_5}\left[\left(\mu-\frac{\theta}{2}\right)t\right]^{m_7} \left[-(v_0-\theta)\beta_t\right]^{m_8}\\
        &=\sum_{i_1=0}^{m_2}\sum_{i_2=0}^{m_5}\sum_{i_3=0}^{m_7}\sum_{i_4=0}^{m_8} c_{\boldsymbol{i}} b_{\boldsymbol{m},\boldsymbol{i}},
    \end{align*}

where :math:`\boldsymbol{i} \mathrel{:=} (i_1,i_2, i_3,i_4)`,

.. math::

    \begin{align*}
        c_{\boldsymbol{i}}
        &\mathrel{:=} \binom{m_2}{i_1}\binom{m_5}{i_2}\binom{m_7}{i_3}\binom{m_8}{i_4} \frac{(-1)^{i_1+\cdots+i_4}}{2^{m_1+m_4+i_1+i_2+i_3+m_8}},\\
        b_{\boldsymbol{m},\boldsymbol{i}}
        &\mathrel{:=} \frac{e^{-(m_1+m_4+m_8-i_4)kt} t^{m_7}}{k^{m_1+m_4+i_1+i_2+m_8}} \mu^{m_7-i_3}(v_0-\theta)^{m_8}\theta^{i_3}\sigma_v^{m_1+i_1} \rho^{m_2-i_1} \left(\sqrt{1-\rho^2}\right)^{m_3} \rho_J^{m_5-i_2}.
    \end{align*}

The first conditional moment is straightforward to compute and is given by:

.. math::

    \begin{equation*}
        \mathbb{E}[y_t|v_0] %= \frac{1- e^{-kt}}{2k^2}\lambda \mu_v + \left(\rho_J - \frac{1}{2k} \right) \lambda t \mu_v  + \lambda t \mu_s + (\mu -\theta/2)t - (v_0-\theta)\beta_t.
        = (\mu - \mathbb{E}[v]/2)t - (v_0 - \mathbb{E}[v])\beta_{t} + \lambda t (\mu_s + \rho_J\mu_v),
    \end{equation*}

where :math:`\mathbb{E}[v] = \theta + \lambda \mu_v /k`.
To compute higher-order conditional moments of :math:`y_t`, it suffices to evaluate the
conditional joint moment:

.. math::

    \begin{equation}%\label{eqn:joint-ieii-ieziziz}
        \mathbb{E}[I\!E_t^{m_1}I_t^{m_2}I_t^{*m_3}I\!E\!Z_t^{m_4}I\!Z_t^{m_5}I\!Z_t^{*m_6}|v_0].
    \end{equation}

Before addressing the computation of this joint moment, we outline the derivation of conditional
central moment of :math:`y_t`. Define the centralized return as

.. math::

    \begin{equation*}
        \bar{y}_t \mathrel{:=} y_t - \mathbb{E}[y_t|v_0].
    \end{equation*}

The :math:`m`-th conditional central moment of :math:`\bar{y}_t` can then be expressed as:

.. math::

    \begin{align*}
        \mathbb{E}[\bar{y}_t^m|v_0] = \sum_{i=0}^m\binom{m}{i}(-1)^i \mathbb{E}^i[y_t|v_0]\mathbb{E}[y_t^{m-i}|v_0].
    \end{align*}

This decomposition demonstrates that the computation of conditional central moments relies
on the computation of conditional moments.

Unconditional Moments
=========================

Consequently, the conditional moments of the return,
:math:`\mathbb{E}[y_t^m|v_0]`, are also polynomials in :math:`v_0`. This property
allows us to leverage the polynomial structure to compute the unconditional moments
of the return, :math:`\mathbb{E}[y_t^m]`, as demonstrated in :doc:`srjd`.

Conditional Moments - II
==========================

For some circumstances, the conditional is that both the initial variance and
the jumps in the variance are given. This section is devoted to deriving the
conditional moments and central moments under these situations.

Note that :math:`y_t \triangleq \log s(t) - \log s(0)`.
Define :math:`I\!Z_t^s\triangleq \int_0^t dz^s(u)`. Then, we have

.. math::

   y_t = y_{svvj,t} + I\!Z_t^s,

where :math:`y_{svvj,t}` denotes the yield :math:`y_t` in Equation
:eq:`y_svvj_t` from the :abbr:`SVVJ(Stochastic Volatility with
Jumps in the Variance process)` model.

Given the initial variance :math:`v_0` and the 
:abbr:`CPP(Compound Poisson Process)` in the variance over interval 
:math:`[0,t]`, :math:`z^v(u), 0\le u \le t`, we are going to derive
the conditional moments and conditional central moments of return 
over this interval :math:`[0,t]`.

We define two centralized variables

.. math::
   
   \begin{align*}
   \overline{y}_{svvj,t} 
   &\triangleq y_{svvj,t} - 
   \mathbb{E}[y_{svvj,t}|v_0,z^v(u), 0\le u \le t],\\
   \overline{I\!Z^s_t} 
   &\triangleq I\!Z^s_t - \mathbb{E}[I\!Z^s_t|z^v(u), 0\le u \le t]
   \end{align*}

to introduce the (conditionally) centralized return

.. math::
   
   \overline{y}_t 
   \triangleq \overline{y}_{svvj, t} + \overline{I\!Z^s_t}.

Thus, the conditional moments and central moments can be derived through the
following equations,

.. math::
   
   \begin{align*}
   &\mathbb{E}[y_t^m|v_0, z^v(u), 0\le u\le t] \\
   &= \sum_{i=0}^{m}C_m^i \mathbb{E}[y_{svvj, t}^i|v_0, z^v(u), 0\le u\le t]
   \mathbb{E}[(I\!Z^s_t)^{m-i}|z^v(u), 0\le u\le t],\\
   &\mathbb{E}[\overline{y}_t^m|v_0, z^v(u), 0\le u\le t] \\
   &= \sum_{i=0}^{m}C_m^i \mathbb{E}[\overline{y}_{svvj, t}^i
   |v_0, z^v(u), 0\le u\le t] \mathbb{E}[(\overline{I\!Z^s_t})^{m-i}
   |z^v(u), 0\le u\le t].
   \end{align*}

They are implementd in functions :py:func:`~ajdmom.mdl_svcj.mom.moments_y_to`
and :py:func:`~ajdmom.mdl_svcj.cmom.cmoments_y_to` in this subpackage 
(:py:mod:`ajdmom.mdl_svcj`), respectively.

Note that

.. math::
   
  \begin{align*}
  &\mathbb{E}[(I\!Z_t^{s})^m|z^v(u), 0\le u \le t]\\
  &= \mathbb{E}\left[\left(\sum_{i=1}^{N(t)} J_i^{s}|J_i^v \right)^m
  \bigg|z^v(u), 0\le u \le t\right]\\
  &= \sum_{m_1,\cdots, m_{N(t)}} C_m^{m_1}\cdots C_{m-(m_1+\cdots
  +m_{N(t)-1})}^{m_{N(t)}} 
  \mathbb{E}[(J_1^s)^{m_1}|J_1^v] \cdots \mathbb{E}[(J_{N(t)}^s)^{m_{N(t)}}
  |J_{N(t)}^v].
  \end{align*}

And :math:`\mathbb{E}[(\overline{I\!Z_t^{s}})^m|z^v(u), 0\le u \le t]` is
derived similarly. They are implemented in functions 
:py:func:`~ajdmom.mdl_svcj.mom.moment_IZs` and 
:py:func:`~ajdmom.mdl_svcj.cmom.cmoment_IZs` in this subpackage 
(``ajdmom.mdl_svcj``), respectively.

API
====

.. autosummary::
   :toctree: generated
   
   ajdmom.mdl_svcj.cmom
   ajdmom.mdl_svcj.mom
   ajdmom.mdl_svcj.cond_cmom
   ajdmom.mdl_svcj.cond_mom
   ajdmom.mdl_svcj.cond2_cmom
   ajdmom.mdl_svcj.cond2_mom
   ajdmom.mdl_svcj.cond_ieii_ieziziz_mom
   ajdmom.mdl_svcj.ieziziz_mom


=====================================================================
:abbr:`SVCJ(Stochastic Volatility with Contemporaneous Jumps)` Model
=====================================================================

In this subpackage (``ajdmom.mdl_svcj``), we consider the following 
:abbr:`SV(Stochastic Volatility)` model,
which adds contemporaneous jumps in the price and variance of 
the Heston model: 

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

Define :math:`y_t \triangleq \log s(t) - \log s(0)`, and 
:math:`I\!Z_t^s\triangleq \int_0^t dz^s(u)`. Then, we have

.. math::
   
   y_t = y_{svvj,t} + I\!Z_t^s,

where :math:`y_{svvj,t}` denotes the yield :math:`y_t` in Equation
:eq:`y_svvj_t` from the :abbr:`SVVJ(Stochastic Volatility with
Jumps in the Variance process)` model.


For models including jumps in the variance, it seems that only conditional
moments and conditional central moments 
(given :math:`v_0, z^v(u), 0\le u \le t`)
can be derived in closed-form for any order. Therefore, for those models, 
the package will focus on the derivation of conditional moments and conditional
central moments.

Conditional Moments
====================

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


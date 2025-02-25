=====================================================================
:abbr:`SVVJ(Stochastic Volatility with Jumps in the Variance)` Model
=====================================================================

In this subpackage (``ajdmom.mdl_svvj``), we consider the following 
:abbr:`SV(Stochastic Volatility)` model,
which adds a jump component in the variance of the Heston model: 

.. math::
   
   d\log s(t) &= (\mu- v(t)/2) dt + \sqrt{v(t)}dw^s(t),\\
   dv(t)      &= k(\theta - v(t))dt + \sigma_v \sqrt{v(t)}dw^v(t) + dz(t),

where :math:`z(t)` is a :abbr:`CPP(Compound Poisson Process)` with 
constant arrival rate :math:`\lambda` and jumps distributed according to 
distribution :math:`F_j(\cdot,\boldsymbol{\theta}_j)` with parameter 
:math:`\boldsymbol{\theta}_j`. 
Usually, the jump distribution is set as an exponential distribution with
scale parameter :math:`\mu_v` (= 1/rate), which will help to assure the 
variance always be non-negative, i.e., :math:`v(t) \ge 0, \forall t\ge 0`.

Define :math:`y_t \equiv \log s(t) - \log s(0)`. Then, we have

.. math::
  :label: y_svvj_t
   
  \begin{align*}
  y_t
  &= (\mu-\theta/2)t - (v_0 - \theta)\beta_t
   + \frac{1}{2k}e^{-kt}I\!E\!Z_t - \frac{1}{2k}I\!Z_t\\
  &\quad + \frac{\sigma_v}{2k} e^{-kt}I\!E_t + 
    \left(\rho -\frac{\sigma_v}{2k} \right)I_t + \sqrt{1-\rho^2}I_t^{*},
  \end{align*}

where :math:`\beta_t \equiv (1-e^{-kt})/(2k)`. 
Please refer to :doc:`../generated/ajdmom.ito_cond_mom` for the definitions
of :math:`I\!E\!Z_t` and :math:`I\!Z_t`.
Please also note that
:math:`I\!E_t\equiv I\!E_{0,t}, I_t\equiv I_{0,t}, I_t^{*} \equiv I_{0,t}^{*}`,
and refer to :doc:`theory` for the definitions of 
:math:`I\!E_{s,t}, I_{s,t}, I_{s,t}^{*}`.

For models including jumps in the variance, it seems that only conditional
moments and conditional central moments (given :math:`v_0, z_s, 0\le s \le t`)
can be derived in closed-form for any order. Therefore, for those models, 
the package will focus on the derivation of conditional moments and conditional
central moments.

Conditional Moments
====================

Given the initial variance :math:`v_0` and the 
:abbr:`CPP(Compound Poisson Process)` over interval :math:`[0,t]`, 
:math:`z_s, 0\le s \le t`, the conditional mean of return over this interval
is given by

.. math::
   
   \begin{align*}
   &\mathbb{E}[y_t|v_0,z_s, 0\le s\le t] \\
   &= (\mu-\theta/2)t - (v_0 - \theta)\beta_t + \frac{1}{2k} 
   e^{-kt}I\!E\!Z_t - \frac{1}{2k}I\!Z_t.
   \end{align*}

Let us define 
:math:`\overline{y}_t \triangleq y_t - \mathbb{E}[y_t|v_0,z_s, 0\le s\le t]`,
then we have

.. math::
   
   \overline{y}_t = \frac{\sigma_v}{2k} e^{-kt}I\!E_t + 
   \left( \rho -\frac{\sigma_v}{2k} \right)I_t  + \sqrt{1-\rho^2}I_t^{*}.

Conditional Central Moments
----------------------------

For conditional central moments, we have

.. math::
  :label: cmoment-y_svvj
   
   \begin{align*}
   &\mathbb{E}[{\overline{y}_t}^l|v_0,z_s, 0\le s\le t]\\
   &= \sum_{\boldsymbol{n}} c(\boldsymbol{n}) b(\boldsymbol{n}) 
   \mathbb{E}[(e^{-kt}I\!E_t)^{n_1}  I_t^{n_2}  I_t^{*n_3}
   |v_0,z_s, 0\le s\le t],
   \end{align*}

where :math:`\boldsymbol{n} = (n_1, n_2, n_3)`, :math:`n_1+n_2+n_3=l`,
:math:`c(\boldsymbol{n}) = C_l^{n_1} C_{l-n_1}^{n_2}`, and

.. math::
   
   b(\boldsymbol{n}) 
   = \left( \frac{\sigma_v}{2k} \right)^{n_1} 
   \left( \rho -\frac{\sigma_v}{2k} \right)^{n_2} 
   \left( \sqrt{1-\rho^2} \right)^{n_3}

The derivation for 
:math:`\mathbb{E}[E_t^{n_1} I_t^{n_2} I_t^{*n_3}|v_0,z_s, 0\le s\le t]` 
has been implemented in :py:func:`~ajdmom.ito_cond_mom.moment_IEII` 
from module :py:mod:`ajdmom.ito_cond_mom`.
The conditional central moments in :eq:`cmoment-y_svvj` is implemented in
:py:func:`~ajdmom.mdl_svvj.cond2_cmom.cmoments_y_to` in the subpackage
``ajdmom.mdl_svvj``, noting that the function now derives simultaneously
the conditional central moments with orders from 1 to :math:`l`.

Conditional Moments
--------------------

Now we rewrite :math:`y_t` as the following

.. math::
   
   y_t = (\mu - \theta/2)t - (v_0 - \theta)\beta_t + \frac{1}{2k} 
   e^{-kt}I\!E\!Z_t - \frac{1}{2k}I\!Z_t + \overline{y}_t,

to enable us to exploit 
:py:func:`~ajdmom.mdl_svvj.cond2_cmom.cmoment_y` for the derivation of those
involved conditional central moments. With this expression, the conditional
moments can be derived through the following equation

.. math::
  :label: moment-y_svvj
   
   \begin{align*}
   &\mathbb{E}[y_t^l|v_0,z_s, 0\le s\le t] \\
   &= \sum_{\boldsymbol{n}} c_2(\boldsymbol{n}) b_2(\boldsymbol{n})
   (e^{-kt}I\!E\!Z_t)^{n_3} I\!Z_t^{n_4} 
   \mathbb{E}[\overline{y}_t^{n_5}|v_0, z_s, 0\le s \le t],
   \end{align*}

where :math:`\boldsymbol{n} = (n_1, n_2, n_3, n_4, n_5)`, 
:math:`\sum_{i=1}^5 n_i = l`,

.. math::
   
   \begin{align*}
   c_2(\boldsymbol{n}) 
   &= C_l^{n_1} C_{l-n_1}^{n_2} C_{l-n_1-n_2}^{n_3} C_{l-n_1-n_2-n_3}^{n_4}
     C_{l-n_1-n_2-n_3-n_4}^{n_5},\\
   b_2(\boldsymbol{n}) 
   &= (-1)^{n_2}  (2k)^{-(n_3+n_4)} [(\mu-\theta/2)t]^{n_1}
   (v_0-\theta)^{n_2}.
   \end{align*}

The conditional moments in :eq:`moment-y_svvj` is implemented in
:py:func:`~ajdmom.mdl_svvj.cond2_mom.moments_y_to` in the subpackage
``ajdmom.mdl_svvj``.


API
====

.. autosummary::
   :toctree: generated
   
   ajdmom.mdl_svvj.cond2_cmom
   ajdmom.mdl_svvj.cond2_mom


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
Please refer to :doc:`../generated/ajdmom.ito_cond2_mom` for the definitions
of :math:`I\!E\!Z_t` and :math:`I\!Z_t`.
Please also note that
:math:`I\!E_t\equiv I\!E_{0,t}, I_t\equiv I_{0,t}, I_t^{*} \equiv I_{0,t}^{*}`,
and refer to :doc:`theory` for the definitions of 
:math:`I\!E_{s,t}, I_{s,t}, I_{s,t}^{*}`.



Conditional Moments - I
==========================

Given the initial variance :math:`v_0`, the conditional mean of :math:`y_t` is
given as

.. math::

   \begin{align*}
   \mathbb{E}[y_t|v_0]
   &= (\mu-\theta/2)t - (v_0 - \theta)\beta_t + \frac{1}{2k}e^{-kt}
   \mathbb{E}[I\!E\!Z_t] - \frac{1}{2k}\mathbb{E}[I\!Z_t]\\
   &= (\mu-\theta/2)t - (v_0 - \theta)\beta_t + \frac{\lambda \mu_v}{k}\beta_t
   - \frac{1}{2k}\lambda t \mu_v.
   \end{align*}


Let us define
:math:`\overline{y}_t \triangleq y_t - \mathbb{E}[y_t|v_0]`,
then we have

.. math::

   \begin{align*}
   \overline{y}_t
   &= \frac{\sigma_v}{2k} e^{-kt}I\!E_t +
   \left(\rho -\frac{\sigma_v}{2k} \right)I_t + \sqrt{1-\rho^2}I_t^{*}
   + \frac{1}{2k}e^{-kt}I\!E\!Z_t - \frac{1}{2k}I\!Z_t\\
   &\quad - \frac{\lambda \mu_v}{2k^2}(1-e^{-kt}) + \frac{1}{2k}\lambda t\mu_v.
   \end{align*}

Central Moments
--------------------

Then, the :math:`l`-th conditional central moment can be computed via

.. math::

   \mathbb{E}[\overline{y}_t^l|v_0] = \sum_{\mathbf{n}}c(\mathbf{n})
   b(\mathbf{n})\mathbb{E}[(e^{-kt}I\!E_t)^{n_1}I_t^{n_2}I_t^{*n_3}
   (e^{-kt}I\!E\!Z_t)^{n_4} I\!Z_t^{n_5}|v_0],

where :math:`\mathbf{n} = (n_1,n_2,n_3,n_4,n_5,n_6,n_7)`,
:math:`n_1+\cdots+n_7=l`,

.. math::

   \begin{eqnarray*}
   c(\mathbf{n}) &=& \binom{l}{n_1,\cdots, n_7},\\
   b(\mathbf{n}) &=& \left(\frac{\sigma_v}{2k}\right)^{n_1}
   \left(\rho -\frac{\sigma_v}{2k} \right)^{n_2}
   \left(\sqrt{1-\rho^2}\right)^{n_3} \frac{(-1)^{n_5+n_6}}{(2k)^{n_4+n_5+n_7}}
   \left[\frac{\lambda \mu_v}{2k^2}(1-e^{-kt})\right]^{n_6} (\lambda t\mu_v)^{n_7}.
   \end{eqnarray*}


Moments
-------------------

Given conditional central moments, it is easy to compute conditional moments as
the following

.. math::

   \mathbb{E}[y_t^l|v_0] = \sum_{i=0}^l\binom{n}{i} \mathbb{E}^i[y_t|v_0]
   \mathbb{E}[\overline{y}_t^{l-i}|v_0].


Unconditional Moments
==========================

By substituting the terms :math:`\mathbb{E}[(v_0 - \theta)^l]` within the
conditional central moments and conditional moments, we get the unconditional
central moments and unconditional moments. Please refer to the
:abbr:`SRJD(Square-Root Jump Diffusion)` model page for the computation of the
terms :math:`\mathbb{E}[(v_0 - \theta)^l]`.

However, it is a little complicated to derive the unconditional covariances.
It is necessary to introduce more notations as
:math:`y_{n+1} \equiv y((n+1)h) - y(nh)`,

.. math::

   \begin{align*}
   y_{n+1}
   &= - (v_n-\theta)\beta +
     \frac{\sigma_v}{2k} e^{-k(n+1)h}I\!E_{n+1} +
     \left(\rho -\frac{\sigma_v}{2k} \right)I_{n+1} +
     \sqrt{1-\rho^2}I_{n+1}^{*}\\
   &\quad ~ + \frac{1}{2k}e^{-k(n+1)h}I\!E\!Z_{n+1}
          - \frac{1}{2k}I\!Z_{n+1}
          + (\mu - \theta/2) h ,
   \end{align*}

where :math:`\beta \equiv (1-e^{-kh})/(2k)`, and
:math:`v_n - \theta = e^{-kh}(v_{n-1} - \theta) + \sigma_ve^{-knh}I\!E_n + e^{-knh}I\!E\!Z_n`.


When expanding :math:`y_{n+1}^{l_2}`, the indexing (:math:`n_0+\cdots+n_6=l_2`) is organized as

+---------------------+-------------------+----------------+--------------------+----------------------+-------------------+---------------------------+
|:math:`(v_n-\theta)` |:math:`I\!E_{n+1}` |:math:`I_{n+1}` |:math:`I_{n+1}^{*}` |:math:`I\!E\!Z_{n+1}` |:math:`I\!Z_{n+1}` |:math:`(\mu - \theta/2) h` |
+=====================+===================+================+====================+======================+===================+===========================+
|:math:`n_0`          |:math:`n_1`        |:math:`n_2`     |:math:`n_3`         |:math:`n_4`           |:math:`n_5`        |:math:`n_6`                |
+---------------------+-------------------+----------------+--------------------+----------------------+-------------------+---------------------------+


Covariances
---------------------

Covariances can be computed via

.. math::

   cov(y_n^{l_1}, y_{n+1}^{l_2}) = \mathbb{E}[y_n^{l_1}y_{n+1}^{l_2}] -
   \mathbb{E}[y_{n}^{l_1}]\mathbb{E}[y_{n+1}^{l_2}].

Therefore, we only need to compute :math:`\mathbb{E}[y_n^{l_1}y_{n+1}^{l_2}]`.

.. math::

   \begin{align*}
   &\mathbb{E}[y_n^{l_1}y_{n+1}^{l_2}]\\
   &=\sum_{\mathbf{n}} c(\mathbf{n}) b(\mathbf{n})
     \sum_{\mathbf{m}} c(\mathbf{m}) b(\mathbf{m})\\
   &\quad\mathbb{E}[(v_{n-1} - \theta)^{m_0}(e^{-knh}I\!E_n)^{m_1}I_n^{m_2}I_n^{*m_3}
   (e^{-knh}I\!E\!Z_n)^{m_4} I\!Z_n^{m_5}
   \cdot \\
   &\qquad (v_n - \theta)^{n_0}(e^{-k(n+1)h}I\!E_{n+1})^{n_1}I_{n+1}^{n_2}I_{n+1}^{*n_3}
   (e^{-k(n+1)h}I\!E\!Z_{n+1})^{n_4} I\!Z_{n+1}^{n_5}].
   \end{align*}


Note that

.. math::

   \begin{align*}
   &\mathbb{E}[I\!E_{n+1}^{n_1} I_{n+1}^{n_2} I_{n+1}^{*n_3} I\!E\!Z_{n+1}^{n_4}
    I\!Z_{n+1}^{n_5}]\\
   &= \sum_{n_1,n_4,i,j,l,o,p,q,r,s}b_{n_1,n_4,i,j,l,o,p,q,r,s} e^{(n_1+n_4)knh}
     e^{ikh} h^j k^{-l} (v_n-\theta)^o \theta^p \sigma_v^q \lambda^r \mu_v^s,\\
   &(v_n-\theta)^{n_0} e^{-(n_1+n_4)k(n+1)h} \mathbb{E}[I\!E_{n+1}^{n_1}
    I_{n+1}^{n_2} I_{n+1}^{*n_3} I\!E\!Z_{n+1}^{n_4} I\!Z_{n+1}^{n_5}]\\
   &= \sum_{n_1,n_4,i,j,l,o,p,q,r,s}b_{n_1,n_4,i,j,l,o,p,q,r,s} e^{-(n_1+n_4)kh}
     e^{ikh} h^j k^{-l} (v_n-\theta)^{o+n_0} \theta^p \sigma_v^q \lambda^r \mu_v^s.
   \end{align*}

A function is defined to implement the corresponding derivation and the expansion
of :math:`(v_n-\theta)`, resulting in

.. math::

   \begin{align*}
   &ve\_I\!EII\_I\!E\!ZI\!Z\_vn(n_0,n_1,n_2,n_3,n_4,n_5)\\
   &= \sum_{m_1,m_2,i,j,l,o,p,q,r,s} b_{m_1,m_2,i,j,l,o,p,q,r,s} e^{-(m_1+m_2)knh}
     I\!E_n^{m_1} I\!E\!Z_n^{m_2}
     e^{-ikh} h^j k^{-l} (v_n-\theta)^{o} \theta^p \sigma_v^q \lambda^r \mu_v^s.
   \end{align*}

The expansion of :math:`(v_n-\theta)` is done via,

.. math::

   (v_n-\theta)^m
   = \sum_{\mathbf{m}} \binom{m}{m_1,m_2,m_3} [e^{-kh}(v_{n-1} - \theta)]^{m_1}
     (\sigma_ve^{-knh}I\!E_n)^{m_2} (e^{-knh}I\!E\!Z_n)^{m_3}.



Conditional Moments - II
==========================

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
has been implemented in :py:func:`~ajdmom.ito_cond2_mom.moment_IEII`
from module :py:mod:`ajdmom.ito_cond2_mom`.
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

   ajdmom.mdl_svvj.cmom
   ajdmom.mdl_svvj.mom
   ajdmom.mdl_svvj.cov
   ajdmom.mdl_svvj.cond_cmom
   ajdmom.mdl_svvj.cond_mom
   ajdmom.mdl_svvj.cond2_cmom
   ajdmom.mdl_svvj.cond2_mom


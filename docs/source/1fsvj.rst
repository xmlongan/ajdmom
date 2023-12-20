=================================================================
:abbr:`1FSVJ(One-Factor Stochastic Volatility with Jumps)` Model
=================================================================

In this subpackage (``ajdmom.mdl_1fsvj``), we consider the following 
:abbr:`SV(Stochastic Volatility)` model,
which adds a jump component in the log price process of the Heston model: 

.. math::
   
   d\log s(t) &= (\mu- v(t)/2) dt + \sqrt{v(t)}dw^s(t) + dz(t),\\
   dv(t)      &= k(\theta - v(t))dt + \sigma_v \sqrt{v(t)}dw^v(t),

where :math:`z(t)` is a :abbr:`CPP(Compound Poisson Process)` with 
a constant arrival rate :math:`\lambda` and jump distribution
:math:`F_j(\cdot,\boldsymbol{\theta}_j)` with parameter 
:math:`\boldsymbol{\theta}_j`. 
We take normal distribution with mean 
:math:`\mu_j` and variance :math:`\sigma_j^2` as an example of 
:math:`F_j(\cdot,\boldsymbol{\theta}_j)`. For this model,

.. math::
   
   y_n = y_{o,n} + J_n,

where

.. math::
   
  y_{o,n} &\triangleq \mu h - \frac{1}{2}IV_{n} + \rho I_n + 
  \sqrt{1-\rho^2}I_n^{*}, \\
  J_n &\triangleq z(nh) - z((n-1)h) = \sum_{i=N((n-1)h)+1}^{N(nh)}j_i,

where :math:`N(t)` is a Poisson process with rate :math:`\lambda`, 
:math:`j_i\sim\mathcal{N}(\mu_j,\sigma_j^2)`.


Moments
========

Moments and Central Moments

.. math::
   
   E[\overline{y}_{n}^l] 
   &= E[(\overline{y}_{o,n} + \overline{J}_n)^l]
   = \sum_{i=0}^{l} C_l^i E[\overline{y}_{o,n}^i]E[\overline{J}_n^{l-i}],\\
   E[y_n^l]
   &= E[(y_{o,n} + J_n)^l]
   = \sum_{i=0}^{l} C_l^i E[y_{o,n}^i] E[J_n^{l-i}].

Functions :py:func:`~ajdmom.mdl_1fsv.mom.moment_y` and 
:py:func:`~ajdmom.mdl_1fsv.cmom.cmoment_y` can be used to compute
:math:`E[y_{o,n}^i]` and
:math:`E[\overline{y}_{o,n}^i]`, respectively.
Meanwhile, functions :py:func:`~ajdmom.cpp_mom.mcpp` and
:py:func:`~ajdmom.cpp_mom.cmcpp` can be used to compute
:math:`E[J_n^{l-i}]` and
:math:`E[\overline{J}_n^{l-i}]`, respectively.

Covariances
============

.. math::
   
   cov(y_n^{l_1}, y_{n+1}^{l_2})
   = E[y_n^{l_1}y_{n+1}^{l_2}] - E[y_n^{l_1}]E[y_{n+1}^{l_2}]

which reduces to 

.. math::
   
   &E[y_n^{l_1}y_{n+1}^{l_2}]\\
   &= \sum_{i=0}^{l_2}C_{l_2}^i E[y_n^{l_1}y_{o,n+1}^i]E[J_{n+1}^{l_2-i}]\\
   &= \sum_{i=0}^{l_2}C_{l_2}^i \sum_{j=0}^{l_1}C_{l_1}^j 
   E[y_{o,n}^jy_{o,n+1}^i] E[J_n^{l_1-j}]E[J_{n+1}^{l_2-i}].

Function :py:func:`~ajdmom.mdl_1fsv.cov.moment_yy` in module 
:py:mod:`ajdmom.mdl_1fsv.cov` can be used to compute
:math:`E[y_{o,n}^jy_{o,n+1}^i]`.

In summary, I defined

1. :py:func:`~ajdmom.mdl_1fsvj.mom.moment_y` for moment :math:`E[y_n^l]`.

2. :py:func:`~ajdmom.mdl_1fsvj.cmom.cmoment_y` for central moment 
   :math:`E[\overline{y}_{n}^l]`.

3. :py:func:`~ajdmom.mdl_1fsvj.cov.cov_yy` for covariance 
   :math:`cov(y_n^{l_1}, y_{n+1}^{l_2})`.


API
====

.. autosummary::
   :toctree: generated
   
   ajdmom.mdl_1fsvj.cmom
   ajdmom.mdl_1fsvj.mom
   ajdmom.mdl_1fsvj.cov
   ajdmom.mdl_1fsvj.euler

.. automodule:: ajdmom.mdl_1fsvj.mom
   :members:

.. automodule:: ajdmom.mdl_1fsvj.cmom
   :members:

.. automodule:: ajdmom.mdl_1fsvj.cov
   :members:

.. automodule:: ajdmom.mdl_1fsvj.euler
   :members:

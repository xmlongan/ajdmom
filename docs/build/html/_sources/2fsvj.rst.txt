==================================================================
:abbr:`2FSVJ(Two-Factor Stochastic Volatility with Jumps)` Model
==================================================================

In this subpackage (``ajdmom.mdl_2fsvj``), we consider the following 
:abbr:`SV(Stochastic Volatility)` model:

.. math::
   
    d\log s(t) &= (\mu- v(t)/2) dt + \sqrt{v(t)}dw(t) + dz(t),\\
    v(t)       &= v_1(t) + v_2(t),\\
    dv_1(t)    &= k_1(\theta_1 - v_1(t))dt + \sigma_{1v} \sqrt{v_1(t)}dw_1(t),\\
    dv_2(t)    &= k_2(\theta_2 - v_2(t))dt + \sigma_{2v} \sqrt{v_2(t)}dw_2(t),

where :math:`z(t)` is a :abbr:`CPP(Compound Poisson Process)` as that in the 
:doc:`1fsvj` page, all others are set as these in the :doc:`2fsv` page.

We have :math:`y_n = y_{o,n} + J_n` where

.. math::
   
   y_{o,n} &\triangleq \mu h - \frac{1}{2}IV_n + I_n^{*},\\
   J_n &\triangleq z(nh) - z((n-1)h) = \sum_{i=N((n-1)h)+1}^{N(nh)}j_i. 


Central Moments
================

Similarly, I define :math:`\overline{y}_n \triangleq y_n - E[y_n]` and we have

.. math::
   
   E[\overline{y}_n^l]
   = \sum_{i=0}^l C_l^i E[\overline{y}_{o,n}^i] E[\overline{J}_n^{l-i}],

where :math:`E[\overline{y}_{o,n}]= y_{o,n} - E[y_{o,n}]` and
:math:`E[\overline{J}_n]= J_n - E[J_n]`

In summary, I defined

1. :py:func:`~ajdmom.mdl_2fsvj.cmom.cmoment_y`.

Moments
========

.. math::
   
   E[y_n^l]
   = \sum_{i=0}^l C_l^i E[y_{o,n}^i] E[J_n^{l-i}].

I defined

1. :py:func:`~ajdmom.mdl_2fsvj.mom.moment_y`.

Covariances
============

.. math::
   
   cov(y_n^{l_1}, y_{n+1}^{l_2})
   = E[y_n^{l_1}y_{n+1}^{l_2}] - E[y_n^{l_1}]E[y_{n+1}^{l_2}].

.. math::
   
   E[y_n^{l_1}y_{n+1}^{l_2}]
   &= E[(y_{o,n}+J_n)^{l_1}(y_{o,n+1}+J_{n+1})^{l_2}]\\
   &= \sum_{i=0}^{l_1}C_{l_1}^i \sum_{j=0}^{l_2}C_{l_2}^j 
   E[y_{o,n}^i J_n^{l_1-i}y_{o,n+1}^j J_{n+1}^{l_2-j}]\\
   &= \sum_{i=0}^{l_1}\sum_{j=0}^{l_2}C_{l_1}^i C_{l_2}^j
   E[y_{o,n}^iy_{o,n+1}^j]E[J_n^{l_1-i}] E[J_{n+1}^{l_2-j}]

In summary, I defined

1. :py:func:`~ajdmom.mdl_2fsvj.cov.moment_yy`,

2. :py:func:`~ajdmom.mdl_2fsvj.cov.cov_yy`.


API
====

.. autosummary::
   :toctree: generated
   
   ajdmom.mdl_2fsvj.cmom
   ajdmom.mdl_2fsvj.mom
   ajdmom.mdl_2fsvj.cov
   ajdmom.mdl_2fsvj.euler

.. automodule:: ajdmom.mdl_2fsvj.mom
   :members:

.. automodule:: ajdmom.mdl_2fsvj.cmom
   :members:

.. automodule:: ajdmom.mdl_2fsvj.cov
   :members:

.. automodule:: ajdmom.mdl_2fsvj.euler
   :members:

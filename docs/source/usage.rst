============
User Guide
============

Currently, the ``adjmom`` package supports the auto-derivation of moment,
central moment and covariance formulas of any order for

* Heston :abbr:`SV(Stochastic Volatility)` Model (1FSV), a typical
  :abbr:`AJD(Affine Jump Diffusion)` model, and its three AJD extensions,
* :abbr:`SVJ(Stochastic Volatility with Jumps in the price process)` (1FSVJ),
* Two-Factor SV (2FSV), Two-Factor SV with jumps in the price process (2FSVJ).

(Conditional versions are also available with slight adaptation for these 
four models.)

The package also supports the the auto-derivation for moment and central
moment formulas of any order for

* :abbr:`SRJD(Square-Root Jump Diffusion)`,
* :abbr:`SVCJ(Stochastic Volatility with Contemporaneous Jumps in the variance
  and price)`.

(The SVVJ, SVIJ models can also be supported with slight further work.)

The package also supports the auto-derivation of conditional moment and
conditional central moment formulas of any order for models

* :abbr:`SVVJ(Stochastic Volatility with Jumps in the Variance)`,
* :abbr:`SVIJ(Stochastic Volatility with Independent Jumps in the variance and
  price)`,
* :abbr:`SVCJ(Stochastic Volatility with Contemporaneous Jumps in the variance
  and price)`, and
* :abbr:`SRJD(Square-Root Jump Diffusion)`.

Installing ajdmom
==================

Install ``ajdmom`` from the Python Package Index through
:program:`Command Prompt` (cmd.exe) or :program:`Anaconda Prompt`,
depending on how Python is installed in your Windows system:

.. code-block:: console

   pip install ajdmom

The ``ajdmom`` package can be installed similarly through :program:`Terminal` 
if you are working on Linux or macOS system.

Heston :abbr:`SV(Stochastic Volatility)` model
===============================================

The most well-known example of :abbr:`AJD(Affine Jump Diffusion)` model should 
be the Heston :abbr:`SV(Stochastic Volatility)` model.
In the ``ajdmom`` package, 
it is treated as the baseline model and described by the following 
:abbr:`SDEs(Stochastic Differential Equations)` [#f1]_ ,

.. math::
    ds(t) &= \mu s(t)dt + \sqrt{v(t)}s(t)dw^s(t),\\
    dv(t) &= k(\theta - v(t))dt + \sigma_v\sqrt{v(t)}dw^v(t),

where :math:`s(t)` is the asset price at time :math:`t`. 
The details refer to the :doc:`theory` page. The return :math:`y_i` over the 
*i*\ th interval of length :math:`h` is defined as,

.. math::
   y_i \triangleq \log s(ih) - \log s((i-1)h).

The derivation for the Heston SV model is implemented in the ``mdl_1fsv``
subpackage of the ``ajdmom`` package.

Formula Deriving
===================

The moment and covariance formulas are encoded in objects of class
:py:class:`~ajdmom.poly.Poly` which is a customized dictionary data structure
derived from the
:class:`~collections.UserDict` class in the Python Standard Library 
`collections <https://docs.python.org/3/library/collections.html>`_.

**Moment Formulas**

To get the formula for the first moment :math:`E[y_n]`: 

>>> from ajdmom import mdl_1fsv # mdl_1fsv -> mdl_1fsvj, mdl_2fsv, mdl_2fsvj
>>> from pprint import pprint
>>> 
>>> m1 = mdl_1fsv.moment_y(1)   # 1 in moment_y(1) -> 2,3,4...
>>> 
>>> # moment_y() -> cmoment_y()             : central moment
>>> # dpoly(m1, wrt), wrt = 'k','theta',... : partial derivative
>>>
>>> msg = "which is a Poly with attribute keyfor = \n{}"
>>> print("moment_y(1) = "); pprint(m1); print(msg.format(m1.keyfor))
moment_y(1) = 
{(0, 1, 0, 0, 1, 0, 0, 0): Fraction(-1, 2),
 (0, 1, 0, 1, 0, 0, 0, 0): Fraction(1, 1)}
which is a Poly with attribute keyfor = 
('e^{-kh}', 'h', 'k^{-}', 'mu', 'theta', 'sigma_v', 'rho', 'sqrt(1-rho^2)')

where the two key-value pairs within the returned poly of :code:`moment_y(1)` 
stand for

.. math::
   
   -\frac{1}{2}\times & e^{-0kh}h^1k^{-0}\mu^0\theta^1\sigma_v^0\rho^0
   \left(\sqrt{1-\rho^2}\right)^0,\\
   1\times & e^{-0kh}h^1k^{-0}\mu^1\theta^0\sigma_v^0\rho^0
   \left(\sqrt{1-\rho^2}\right)^0,

respectively. Adding together the two terms reproduces the first moment of 
the Heston SV model, i.e., :math:`E[y_n] = (\mu-\theta/2)h`.

**Covariance Formulas**

The covariances considered in this package are that between :math:`y_n` 
and its lag-1 counterpart :math:`y_{n+1}` with orders 
:math:`(l_1,l_2)`, i.e., 
:math:`cov(y_n^{l_1}, y_{n+1}^{l_2})`. 

To derive the formula for the covariance :math:`cov(y_n^2,y_{n+1})`:

>>> from ajdmom import mdl_1fsv # mdl_1fsv -> mdl_1fsvj, mdl_2fsv, mdl_2fsvj
>>> from pprint import pprint
>>> 
>>> cov21 = mdl_1fsv.cov_yy(2,1) # (2,1) -> (1,1), (1,2), (2,2), (3,2), ...
>>> 
>>> # dpoly(cov21, wrt), wrt = 'k','theta',... : partial derivative
>>> 
>>> msg = "which is a Poly with attribute keyfor =\n{}"
>>> print("cov_yy(2,1) = "); pprint(cov21); print(msg.format(cov21.keyfor))
cov_yy(2,1) = 
{(0, 0, 3, 0, 1, 2, 0, 2): Fraction(-1, 4),
 (0, 0, 3, 0, 1, 2, 2, 0): Fraction(-5, 4),
 (0, 0, 4, 0, 1, 3, 1, 0): Fraction(3, 4),
 (0, 0, 5, 0, 1, 4, 0, 0): Fraction(-1, 8),
 (0, 1, 2, 0, 2, 1, 1, 0): Fraction(1, 2),
 (0, 1, 2, 1, 1, 1, 1, 0): Fraction(-1, 1),
 (0, 1, 3, 0, 2, 2, 0, 0): Fraction(-1, 8),
 (0, 1, 3, 1, 1, 2, 0, 0): Fraction(1, 4),
 (1, 0, 3, 0, 1, 2, 0, 2): Fraction(1, 2),
 (1, 0, 3, 0, 1, 2, 2, 0): Fraction(5, 2),
 (1, 0, 4, 0, 1, 3, 1, 0): Fraction(-3, 2),
 (1, 0, 5, 0, 1, 4, 0, 0): Fraction(1, 4),
 (1, 1, 2, 0, 1, 2, 2, 0): Fraction(1, 1),
 (1, 1, 2, 0, 2, 1, 1, 0): Fraction(-1, 1),
 (1, 1, 2, 1, 1, 1, 1, 0): Fraction(2, 1),
 (1, 1, 3, 0, 1, 3, 1, 0): Fraction(-3, 4),
 (1, 1, 3, 0, 2, 2, 0, 0): Fraction(1, 4),
 (1, 1, 3, 1, 1, 2, 0, 0): Fraction(-1, 2),
 (1, 1, 4, 0, 1, 4, 0, 0): Fraction(1, 8),
 (2, 0, 3, 0, 1, 2, 0, 2): Fraction(-1, 4),
 (2, 0, 3, 0, 1, 2, 2, 0): Fraction(-5, 4),
 (2, 0, 4, 0, 1, 3, 1, 0): Fraction(3, 4),
 (2, 0, 5, 0, 1, 4, 0, 0): Fraction(-1, 8),
 (2, 1, 2, 0, 1, 2, 2, 0): Fraction(-1, 1),
 (2, 1, 2, 0, 2, 1, 1, 0): Fraction(1, 2),
 (2, 1, 2, 1, 1, 1, 1, 0): Fraction(-1, 1),
 (2, 1, 3, 0, 1, 3, 1, 0): Fraction(3, 4),
 (2, 1, 3, 0, 2, 2, 0, 0): Fraction(-1, 8),
 (2, 1, 3, 1, 1, 2, 0, 0): Fraction(1, 4),
 (2, 1, 4, 0, 1, 4, 0, 0): Fraction(-1, 8)}
which is a Poly with attribute keyfor =
('e^{-kh}', 'h', 'k^{-}', 'mu', 'theta', 'sigma_v', 'rho', 'sqrt(1-rho^2)')


Moment Values
===================

Given an exact set of parameter values, values of
the central moments, moments and covariances, and their partial derivatives 
:abbr:`w.r.t.(with respect to)` a parameter can also be computed.

**Moments and Central Moments**

To compute the exact value of the third moment :math:`E[y_n^3]`, given
:math:`(\mu=0.125, k=0.1, \theta=0.25, \sigma_v=0.1, \rho=-0.7, h=1)`: 

>>> ## Moments and Central Moments
>>> from ajdmom.mdl_1fsv.mom import m, dm       # for moments
>>> from ajdmom.mdl_1fsv.cmom import cm, dcm    # for central moments
>>>    
>>> parameters = {'mu':0.125, 'k':0.1, 'theta':0.25, 
...   'sigma_v':0.1, 'rho':-0.7, 'h': 1}
>>>   
>>> # 3rd moment as an example
>>> moment = m(l=3, par=parameters)             #  cm: central moment
>>> # partial derivative w.r.t. parameter 'k'
>>> dmoment = dm(l=3, par=parameters, wrt='k')  # dcm: central moment
>>> moment
-0.04489260315929133
>>> dmoment
0.20556366585696395
   

**Covariances**

To compute the exact value of covariance :math:`cov(y_n^2, y_{n+1}^2)`, 
given :math:`(\mu=0.125, k=0.1, \theta=0.25, \sigma_v=0.1, \rho=-0.7, h=1)`: 

>>> ## Covariance
>>> from ajdmom.mdl_1fsv.cov import cov, dcov
>>> 
>>> parameters = {'mu':0.125, 'k':0.1, 'theta':0.25, 
...   'sigma_v':0.1, 'rho':-0.7, 'h': 1}
>>> 
>>> # covariance cov(y_n^2, y_{n+1}^2) as an example
>>> covariance = cov(l1=2, l2=2, par=parameters)
>>> # partial derivative w.r.t. parameter 'k'
>>> dcovariance = dcov(l1=2, l2=2, par=parameters, wrt='k')
>>> covariance
0.0149529894520537
>>> dcovariance
-0.15904979864793667


:abbr:`AJD(Affine Jump Diffusion)` Extensions
==============================================

In addition to the Heston SV model, there are some extensions, which are
summarized in the following table:

+------------+-----------------------------------------------------------------+
| Model      |    Description                                                  |
+============+=================================================================+
|mdl_1fsv    | - baseline model, i.e., the Heston SV model                     |
|            | - refers to :doc:`theory` or :doc:`1fsv`                        |
+------------+-----------------------------------------------------------------+
|mdl_1fsvj   | - with jumps in the return process of the model mdl_1fsv        |
|            | - refers to :doc:`1fsvj`                                        |
+------------+-----------------------------------------------------------------+
|mdl_2fsv    | - with volatility consisting of superposition of two SRDs       |
|            | - refers to :doc:`2fsv`                                         |
+------------+-----------------------------------------------------------------+
|mdl_2fsvj   | - with jumps in the return process of the model mdl_2fsv        |
|            | - refers to :doc:`2fsvj`                                        |
+------------+-----------------------------------------------------------------+
|mdl_svvj    | - with jumps in the variance of the Heston model                |
|            | - refers to :doc:`svvj`                                         |
+------------+-----------------------------------------------------------------+
|mdl_svij    | - with independent jumps in the price and variance of Heston    |
|            | - refers to :doc:`svij`                                         |
+------------+-----------------------------------------------------------------+
|mdl_svcj    | - with contemporaneous jumps in the price and variance of Heston|
|            | - refers to :doc:`svcj`                                         |
+------------+-----------------------------------------------------------------+
|mdl_srjd    | - Square-Root Jump Diffusion                                    |
|            | - refers to :doc:`srjd`                                         |
+------------+-----------------------------------------------------------------+

Notes: SRD is short for Square-Root Diffusion.

The derivation of (central) moments and covariances for the 
:abbr:`SV(Stochastic Volatility)` models are
implemented in the following subpackages of the :code:`ajdmom` package, 
respectively, as

+---------+--------------------------+----------------------------------------+
| Model   | Subpackage               | Modules                                |
+=========+==========================+========================================+
|mdl_1fsv |  :code:`ajdmom.mdl_1fsv` | - :py:mod:`ajdmom.mdl_1fsv.cmom`       |
|         |                          | - :py:mod:`ajdmom.mdl_1fsv.mom`        |
|         |                          | - :py:mod:`ajdmom.mdl_1fsv.cov`        |
+---------+--------------------------+----------------------------------------+
|mdl_1fsvj|  :code:`ajdmom.mdl_1fsvj`| - :py:mod:`ajdmom.mdl_1fsvj.cmom`      |
|         |                          | - :py:mod:`ajdmom.mdl_1fsvj.mom`       |
|         |                          | - :py:mod:`ajdmom.mdl_1fsvj.cov`       |
+---------+--------------------------+----------------------------------------+
|mdl_2fsv |  :code:`ajdmom.mdl_2fsv` | - :py:mod:`ajdmom.mdl_2fsv.cmom`       |
|         |                          | - :py:mod:`ajdmom.mdl_2fsv.mom`        |
|         |                          | - :py:mod:`ajdmom.mdl_2fsv.cov`        |
+---------+--------------------------+----------------------------------------+
|mdl_2fsvj|  :code:`ajdmom.mdl_2fsvj`| - :py:mod:`ajdmom.mdl_2fsvj.cmom`      |
|         |                          | - :py:mod:`ajdmom.mdl_2fsvj.mom`       |
|         |                          | - :py:mod:`ajdmom.mdl_2fsvj.cov`       |
+---------+--------------------------+----------------------------------------+
|mdl_svvj |  :code:`ajdmom.mdl_svvj` | - :py:mod:`ajdmom.mdl_svvj.cond2_cmom` |
|         |                          | - :py:mod:`ajdmom.mdl_svvj.cond2_mom`  |
+---------+--------------------------+----------------------------------------+
|mdl_svij |  :code:`ajdmom.mdl_svij` | - :py:mod:`ajdmom.mdl_svij.cond2_cmom` |
|         |                          | - :py:mod:`ajdmom.mdl_svij.cond2_mom`  |
+---------+--------------------------+----------------------------------------+
|mdl_svcj |  :code:`ajdmom.mdl_svcj` | - :py:mod:`ajdmom.mdl_svcj.cmom`       |
|         |                          | - :py:mod:`ajdmom.mdl_svcj.mom`        |
+---------+--------------------------+----------------------------------------+
|mdl_srjd |  :code:`ajdmom.mdl_srjd` | - :py:mod:`ajdmom.mdl_srjd.cmom`       |
|         |                          | - :py:mod:`ajdmom.mdl_srjd.mom`        |
+---------+--------------------------+----------------------------------------+

The corresponding quantities for other models (mdl_1fsvj, mdl_2fsv, mdl_2fsvj,
mdl_svvj, mdl_svij, mdl_svcj, mdl_srjd)
can be computed by using the counterparts within their subpackages.
It should be noted that only **conditional** moments|central moments 
(given initial variance and jumps in the variance) can be derived for models
including jumps in the variance, i.e., mdl_svvj, mdl_svij, mdl_svcj, mdl_srjd.

----------

.. [#f1] Whose exact equations vary according to different authors. One simplified setting is :math:`dp(t) = \mu dt + \sqrt{v(t)}dw^s(t)` where :math:`p(t) = \log s(t)` while all other settings keep as the same. :math:`v(t)` is the instantaneous return variance at time :math:`t`, and :math:`w^s(t)` and :math:`w^v(t)` are two Wiener processes with correlation :math:`\rho`. In order to make sure :math:`v(t) >0` for :math:`t>0`, it is required that the parameters :math:`k>0,\theta>0,\sigma_v>0` and satisfy :math:`\sigma_v^2 \leq 2k\theta`, along with an initial :math:`v(0)>0`.

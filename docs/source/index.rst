.. ajdmom documentation master file, created by
   sphinx-quickstart on Fri May 19 17:16:26 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

================================================================
Unlocking Explicit Moments for Affine Jump Diffusions
================================================================

:Release: |release|


``ajdmom`` is a Python library for **automatically deriving explicit, closed-form
moment formulas** for well-established Affine Jump Diffusion (AJD) processes.
It significantly enhances the usability of AJD models by providing both
**unconditional moments** and **conditional moments**, up to any positive integer
order. The supported AJD models include Heston
:abbr:`SV(Stochastic Volatility)` model,
:abbr:`SRJD(Square-Root Jump Diffusion)`,
:abbr:`1SVJ(SV with jumps in the price)`,
:abbr:`2FSV(Two-Factor SV)`,
:abbr:`2FSVJ(Two-Factor SV with jumps in the price)`,
:abbr:`SVVJ(SV with jumps in the variance)`,
:abbr:`SVIJ(SV with independent jumps in the price and variance)`, and
:abbr:`SVCJ(SV with contemporaneous jumps in the price and variance)`.
Extensions to other AJD models are also applicable.

This documentation provides a comprehensive guide to understanding, installing, and
utilizing ``ajdmom`` for your quantitative finance and stochastic modeling needs.


Installation
-------------

To install ``ajdmom``, you can use pip from PyPI:

.. code-block:: bash

   pip install ajdmom

Alternatively, you can install directly from the GitHub repository:

.. code-block:: bash

   pip install git+https://github.com/xmlongan/ajdmom


Quick Start
-------------------

To get the formula for the first moment :math:`\mathbb{E}[y_n]`:

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

The two key-value pairs within the returned `Poly` object from :code:`moment_y(1)`
correspond to the following expressions:

.. math::

   -\frac{1}{2}\times & e^{-0kh}h^1k^{-0}\mu^0\theta^1\sigma_v^0\rho^0
   \left(\sqrt{1-\rho^2}\right)^0,\\
   1\times & e^{-0kh}h^1k^{-0}\mu^1\theta^0\sigma_v^0\rho^0
   \left(\sqrt{1-\rho^2}\right)^0,

respectively. Adding together these two terms gives the first moment of
the Heston SV model, i.e., :math:`\mathbb{E}[y_n] = (\mu-\theta/2)h`.


Table of Contents
-------------------

.. toctree::
   :maxdepth: 2

   usage
   theory
   design
   1fsv
   1fsvj
   2fsv
   2fsvj
   svvj
   svij
   svcj
   srjd
   Basic API <api>

Ongoing Development
--------------------

This code is being developed on an on-going basis at the author's
`Github site <https://github.com/xmlongan/ajdmom>`_.

Support
--------

For support in using this software, submit an
`issue <https://github.com/xmlongan/ajdmom/issues/new>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

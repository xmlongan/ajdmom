.. ajdmom documentation master file, created by
   sphinx-quickstart on Fri May 19 17:16:26 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

================================================================
ajdmom: Unlocking Explicit Moments for Affine Jump Diffusions
================================================================

AJD: Affine Jump Diffusion -----------------------------
``ajdmom`` (AJD **mom**\ ents) 
*Release* v3.1

**AJDmom** is a Python Package designed for auto-deriving the moment and covariance
formulas for :abbr:`AJD(Affine Jump Diffusion)` processes. The AJDs include the
:abbr:`SRJD(Square-Root Jump Diffusion)` processes (or called Square-Root Diffusions
with Jumps), the well-known Heston :abbr:`SV(Stochastic Volatility)` model and its AJD
extensions (:abbr:`SVJ(SV with jumps in the price)`, Two-Factor SV,
Two-Factor SV with jumps in the price, :abbr:`SVVJ(SV with jumps in the variance)`,
:abbr:`SVIJ(SV with independent jumps in the price and variance)`,
:abbr:`SVCJ(SV with contemporaneous jumps in the price and variance)`).


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

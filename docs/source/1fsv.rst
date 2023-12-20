=====================================================
:abbr:`1FSV(One-Factor Stochastic Volatility)` Model
=====================================================

In this subpackage (``ajdmom.mdl_1fsv``), we consider the 
:abbr:`SV(Stochastic Volatility)` model in :doc:`theory`, i.e., 

.. math::
    ds(t) &= \mu s(t)dt + \sqrt{v(t)}s(t)dw^s(t),\\
    dv(t) &= k(\theta - v(t))dt + \sigma_v\sqrt{v(t)}dw^v(t),

which is also the baseline SV model of the package.

The derivation of its moments has been explained in :doc:`design` page.


API
====

.. autosummary::
   :toctree: generated
   
   ajdmom.mdl_1fsv.cmom
   ajdmom.mdl_1fsv.mom
   ajdmom.mdl_1fsv.cov
   ajdmom.mdl_1fsv.euler

.. automodule:: ajdmom.mdl_1fsv.mom
   :members:

.. automodule:: ajdmom.mdl_1fsv.cmom
   :members:

.. automodule:: ajdmom.mdl_1fsv.cov
   :members:

.. automodule:: ajdmom.mdl_1fsv.euler
   :members:

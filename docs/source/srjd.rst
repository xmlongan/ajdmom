=====================================================================
:abbr:`SRJD(Square-Root Jump Diffusion)` Model
=====================================================================

In this subpackage (``ajdmom.mdl_srjd``), we consider the following 
Square-Root Jump-Diffusion process: 

.. math::
   
   dv(t) = k(\theta - v(t))dt + \sigma_v\sqrt{v(t)}dw^v(t) + dz(t),
   
which is the same as that in :doc:`../generated/ajdmom.ito_cond_mom`, details
can be refered to therein.
But, in this subpackage (``ajdmom.mdl_srjd``), we are concerned about
the derivation of :math:`\mathbb{E}[I\!E_t^m|v(0),z(s),0\le s\le t]`
and :math:`\mathbb{E}[v^m(t)|v(0),z(s),0\le s\le t]`. 

Please note that
we write :math:`v_t, z_t, v_0` and :math:`v(t), z(t), v(0)` interchangably.

Conditional Moments
====================

Given :math:`v_0` and :math:`z_{s}, 0\le s \le t`,

.. math::
  
  e^{kt}v_t = \mu_{ev} + \sigma_v I\!E_t,

where :math:`\mu_{ev} \triangleq (v_0-\theta) + \theta e^{kt} + I\!E\!Z_t`.
Thus, we have

.. math::
   
   \begin{align*}
   &\mathbb{E}[(e^{kt}v_t)^m|v_0, z_s, 0\le s \le t] \\
   &\quad = \sum_{j=0}^mC_m^j \mu_{ev}^j \sigma_v^{m-j} 
   \mathbb{E}[I\!E_t^{m-j}|v_0, z_s, 0\le s \le t],
   \end{align*}

further,

.. math::
   
   \begin{align*}
   &\mathbb{E}[v_t^m|v_0, z_s, 0\le s \le t] \\
   &= e^{-mkt} \sum_{j=0}^mC_m^j \mu_{ev}^j \sigma_v^{m-j}
   \mathbb{E}[I\!E_t^{m-j}|v_0, z_s, 0\le s \le t].
   \end{align*}

We have,  :math:`\forall m \ge 2`,

.. math::
  :label: srjd-IE-moment
  
  \begin{align*}
  &\mathbb{E}[I\!E_t^m|v_0, z(s), 0\le s \le t]\\
  &=  \frac{1}{2}m(m-1)(v_0-\theta)\int_0^te^{ks}\mathbb{E}[I\!E_s^{m-2}
    |v_0, z(s), 0\le s \le t]ds\\
  &\quad + \frac{1}{2}m(m-1)\theta\quad~~~  \int_0^te^{2ks}\mathbb{E}
  [I\!E_s^{m-2}|v_0, z(s), 0\le s \le t]ds\\
  &\quad + \frac{1}{2}m(m-1)\quad~~~ \int_0^t(e^{ks}I\!E\!Z_s)\mathbb{E}
  [I\!E_s^{m-2}|v_0, z(s), 0\le s \le t]ds\\
  &\quad + \frac{1}{2}m(m-1)\sigma_v\quad \int_0^te^{ks}\mathbb{E}
  [I\!E_s^{m-1}|v_0, z(s), 0\le s \le t]ds,
  \end{align*}

where :math:`\mathbb{E}[I\!E_t^0|v_0, z(s), 0\le s \le t] = 1` and
:math:`\mathbb{E}[I\!E_t|v_0, z(s), 0\le s \le t] = 0`.

We decode :math:`\mathbb{E}[I\!E_t^m|v_0, z(s), 0\le s \le t]` as 
a ``Poly`` object of the following form:

.. math::
   
   \begin{align*}
   &\mathbb{E}[I\!E_t^m|v_0, z(s), 0\le s \le t]\\
   &\equiv \sum_{j_{1:5}, l_{1:n}, o_{2:n} } c_{j_{1:5}, l_{1:n}, o_{2:n} }
   v_0^{j_1} k^{-j_2} \theta^{j_3} \sigma_v^{j_4} e^{j_5kt} 
   f_{Z_t}(l_{1:n}, o_{2:n}),
   \end{align*}

where :math:`\forall n >= 2`

.. math::
  :label: fZ_IE
   
   \begin{align*}
   f_{Z_t}(l_{1:n}, o_{2:n}) 
   &\triangleq\sum_{i_1=1}^{N(t)}\cdots 
   \sum_{i_n=1}^{N(t)} e^{l_1ks_{i_1} + \cdots + l_nks_{i_n}} 
   J_{i_1}\cdots J_{i_n} \\
   &\quad \cdot e^{o_2 k (s_{i_1}\vee s_{i_2}) + \cdots 
   + o_{n} k (s_{i_1} \vee\cdots \vee s_{i_n}) },
   \end{align*}

and for :math:`n=1`, :math:`f_{Z_t}(l_{1:n}, o_{2:n})` degenerates into
:math:`f_{Z_t}(l_{1:1}) = \sum_{i_1=1}^{N(t)} e^{l_1 k s_{i_1}}J_{i_1}`.
Lastly, when :math:`n=0`, :math:`f_{Z_t}(l_{1:n}, o_{2:n}) = 1`.

The conditional moments of :math:`I\!E_t` and :math:`v(t)` are implemented
in :py:func:`~ajdmom.mdl_srjd.mom.moment_IE` and 
:py:func:`~ajdmom.mdl_srjd.mom.moment_v`, respectively, in the subpackage
``ajdmom.mdl_srjd``.

For the conditional central moments, define
:math:`\overline{v}(t)\triangleq v(t)-\mathbb{E}[v(t)|v_0, z(s), 0\le s\le t]`,
thus :math:`\overline{v}(t) \equiv e^{-kt}\sigma_vI\!E_t`.

.. math::
   
   \begin{align*}
   &\mathbb{E}[\overline{v}^m(t)|v_0, z(s), 0\le s\le t]\\
   &= e^{-mkt}\sigma_v^m \mathbb{E}[I\!E_t^m|v_0, z(s), 0\le s\le t].
   \end{align*}

The conditional central moments are implemented in 
:py:func:`~ajdmom.mdl_srjd.cmom.cmoment_v` in the subpackage 
``ajdmom.mdl_srjd``.

Moments (Unconditional)
=======================

Although it is hard to derive unconditional moments up to any order for the 
SRJD model, it is possible to derive some low order unconditional ones.


.. math::
   
   e^{kt}(v(t) - \theta) - (v_0 - \theta) = \sigma_v I\!E_t + I\!E\!Z_t.

It is assumed that :math:`v_0` is distributed according to the steady-state
distribution of the process. Therefore, 
:math:`\mathbb{E}[v_0^m] = \mathbb{E}[v^m(t)], \forall m \ge 1`.

.. math::
   
   (e^{kt}-1)^m\mathbb{E}[(v(t) -\theta)^m] = 
   \mathbb{E}[(\sigma_v I\!E_t + I\!E\!Z_t)^m].

.. math::
   
   \mathbb{E}[(\sigma_v I\!E_t + I\!E\!Z_t)^m]
   = \sum_{i=0}^m C_m^i \sigma_v^i \mathbb{E}[I\!E_t^i I\!E\!Z_t^{m-i}].

When :math:`m=1, 2, 3`, the moments can be derived. However, 
:math:`\forall m \ge 4`, it seems impossible to achieve that.
For this situation, numerical moments (contrast to formulas) should be 
possible by numerical expectation evaluation. We leave it as an open 
problem within this Python package ``ajdmom``.

API
====

.. autosummary::
  :toctree: generated
   
   ajdmom.mdl_srjd.mom
   ajdmom.mdl_srjd.cmom


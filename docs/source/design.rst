===============
Program Design
===============

A Short Story
==============

It's not straightforward for ordinary programmers to write codes to automate
the moment and covariance derivation in the :doc:`theory` page.

The recursive equations :math:numref:`ito-moment` in the :doc:`theory` page contain integral
operations.
The integrals are not that concise. What makes things worse is that the 
integrals grow recursively. Meanwhile, :math:`v_{n-1}` will get buried in
the results. It seems `Symbolic Computation <https://en.wikipedia.org/wiki/Computer_algebra>`_
is needed to tidy the final
expression such that we can make use of equation :eq:`moment-v` in the
:doc:`theory` page. However, to the best of my knowledge, none of the current
programming languages (Mathematica, MATLAB) or packages (SymPy) have offered 
supports in their *Symbolic Computation* for manipulating Itô processes, 
let alone computing their expectations, which are essential in the moment and covariance
derivation of :abbr:`AJD(Affine Jump Diffusion)` models.

Thus, I believed expertise in compiler design is required to solve the problems
from the bottom. And I do have tried to learn corresponding courses over 
several months but finally realized this demands time and lots of practice, 
so I quitted. 

Fortunately, I come up a different solution (may also be simpler) afterwards 
which will be explained in the following sections.

Insights
=========

I observe some features that allow me to bypass the compiler design approach.
One of the features is that all :math:`E[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}|v_{n-1}]`
can be represented as a "Polynomial" of following form

.. _polynomial-representation:

.. math::
   
   &E[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}|v_{n-1}]\\
   &= \sum_{n_3,i,j,l,o,p,q} b_{n_3ijlopq} e^{n_3k(n-1)h} e^{ik[t-(n-1)h]}
   [t-(n-1)h]^jv_{n-1}^l k^{-o}\theta^p\sigma_v^q

where :math:`b_{ijlopq}` is the item coefficient. And
:math:`E[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]` can be 
represented similarly.

To facilitate the representation and corresponding operations, I designed
a new *class* called :py:class:`~ajdmom.poly.Poly` which is a tailored 
dictionary data structure, derived from the
:class:`~collections.UserDict` class in the Python Standard Library 
`collections <https://docs.python.org/3/library/collections.html>`_.


Essential Integral
===================

The essential computation in recursive equation :eq:`ito-moment` of
the :doc:`theory` page is that of the following integral

.. math::
   
   \int_{(n-1)h}^t e^{ik[s-(n-1)h]} [s-(n-1)h]^j ds.


For the indefinite integral, we have

.. math::
   
   \int e^{nkt} t^m dt =
   \begin{cases}
   \sum_{i=0}^m c_{nmi} \frac{1}{k^{i+1}}e^{nkt} t^{m-i} 
    & \text{if } n\neq 0, m \neq 0,\\
   \frac{1}{nk}e^{nkt}t^0 & \text{if } n\neq 0, m = 0,\\
   \frac{1}{m+1}e^{0kt}t^{m+1} & \text{if } n = 0, m \neq 0,\\
   e^{0kt}t^1 & \text{if } n =0 , m=0,
   \end{cases}

where :math:`c_{nm0} \triangleq \frac{1}{n}` and

.. math::
   :label: c-nmi
   
   c_{nmi} \triangleq \frac{(-1)^{i}}{n^{i+1}} \prod_{j=m-i+1}^{m} j,
   \quad 1\le i \le m.

Coefficient :math:`c_{nmi}` is implemented as function
:py:func:`~ajdmom.ito_mom.c_nmi` which returns a 
`Fraction <https://docs.python.org/3/library/fractions.html>`_ instead of a
decimal (float number).

For the definite integral,

.. math::
   
   \int_{(n-1)h}^t e^{ik[s-(n-1)h]}[s-(n-1)h]^jds = F(t-(n-1)h) - F(0)
    
where :math:`F(t) = \int e^{nkt} t^m dt`. The definite integral is implemented 
as function :py:func:`~ajdmom.ito_mom.int_et`.


Polynomial Representation
--------------------------

The result of the integral, returned by :py:func:`~ajdmom.ito_mom.int_et`,
is represented as a "polynomial" of the following form

.. math::
   
   \int_{(n-1)h}^t e^{ik[s-(n-1)h]} [s-(n-1)h]^j ds
    = \sum_{i,j^{'},l}c_{ij^{'}l}e^{ik[t-(n-1)h]}[t-(n-1)h]^{j^{'}}k^{-l}

which is encoded in a :py:class:`~ajdmom.poly.Poly` with 
:code:`keyfor = ('e^{k[t-(n-1)h]}','[t-(n-1)h]','k^{-}')` which is a derived
`UserDict <https://docs.python.org/3/library/collections.html#collections.UserDict>`_
with ``key`` = :math:`(i,j^{'},l)` and ``value`` = :math:`c_{ij^{'}l}`.


Code Design
============

Itô process moment
-----------------------

With :math:`E[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]` represented
as a "polynomial" of the following form

.. math::
   
   &E[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]\\
   &= \sum_{n_3,i,j,l,o,p,q} b_{n_3ijlopq} e^{n_3k(n-1)h} e^{ik[t-(n-1)h]}
   [t-(n-1)h]^jv_{n-1}^l k^{-o}\theta^p\sigma_v^q,

consequently, we have

.. math::
   
   &e^{-kt}E[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]\\
   &= \sum_{n_3,i,j,l,o,p,q} b_{n_3ijlopq} e^{(n_3-1)k(n-1)h}
   e^{(i-1)k[t-(n-1)h]}[t-(n-1)h]^jv_{n-1}^l k^{-o}\theta^p\sigma_v^q,\\
   &e^{kt}E[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]\\
   &= \sum_{n_3,i,j,l,o,p,q} b_{n_3ijlopq} e^{(n_3+1)k(n-1)h}
   e^{(i+1)k[t-(n-1)h]}[t-(n-1)h]^jv_{n-1}^l k^{-o}\theta^p\sigma_v^q,\\
   &e^{2kt}E[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]\\
   &= \sum_{n_3,i,j,l,o,p,q} b_{n_3ijlopq} e^{(n_3+2)k(n-1)h}
   e^{(i+2)k[t-(n-1)h]}[t-(n-1)h]^jv_{n-1}^l k^{-o}\theta^p\sigma_v^q.

Therefore, it's profitable to consider the following generic integral

.. math::
   :label: int-e-poly
   
   &\int_{(n-1)h}^t e^{mks}E[I\!E_{n-1,s}^{n_3}I_{n-1,s}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]ds\\
   &= \sum_{n_3,i,j,l,o,p,q} b_{n_3ijlopq} e^{(n_3+m)k(n-1)h} \cdot int\_et(i+m,j)\cdot v_{n-1}^l k^{-o}\theta^p\sigma_v^q\\
   &= \sum_{n_3+m,i+m,j^{'},l,o^{'},p,q} b_{(n_3+m)(i+m)j^{'}l o^{'}pq} e^{(n_3+m)k(n-1)h} e^{(i+m)k[t-(n-1)h]}
   [t-(n-1)h]^{j^{'}}\\
   &\qquad \cdot v_{n-1}^{l} k^{-o^{'}}\theta^{p}\sigma_v^{q}

where 

.. math::
   
   int\_et(i+m,j) 
   =\sum_{i+m,j^{'},l^{'}} c_{(i+m)j^{'}l^{'}}e^{(i+m)k[t-(n-1)h]}[t-(n-1)h]^{j^{'}} k^{-l^{'}}.

Implementation:

1. Function :py:func:`~ajdmom.ito_mom.int_e_poly` in module
   :py:mod:`~ajdmom.ito_mom` is defined to accomplish the computation in
   equation :eq:`int-e-poly`.

2. Function :py:func:`~ajdmom.ito_mom.recursive_IEII` in module
   :py:mod:`~ajdmom.ito_mom` is defined to realize
   the recursive step in equation :eq:`ito-moment` of the :doc:`theory` page.

3. Function :py:func:`~ajdmom.ito_mom.moment_IEII` in module
   :py:mod:`~ajdmom.ito_mom` is implemented to calculate
   :math:`E[I\!E_n^{n_3}I_n^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]`.

To demonstration, I re-write the following initial three moments in
:ref:`ito-recursive-i` in the :doc:`theory` page according to the "polynomial"
representation

.. math::
   
   E[I\!E_{n-1,t}^2|v_{n-1}]
   &=& \frac{1}{2}&e^{2k(n-1)h} e^{2k[t-(n-1)h]}[t-(n-1)h]^0v_{n-1}^0
   k^{-1}\theta^1\sigma_v^0\\
   && + &e^{2k(n-1)h}e^{k[t-(n-1)h]}[t-(n-1)h]^0v_{n-1}^1
   k^{-1}\theta^0\sigma_v^0\\
   && - &e^{2k(n-1)h}e^{k[t-(n-1)h]}[t-(n-1)h]^0v_{n-1}^0
   k^{-1}\theta^1\sigma_v^0\\
   && - &e^{2k(n-1)h}e^{0k[t-(n-1)h]}[t-(n-1)h]^0v_{n-1}^1
   k^{-1}\theta^0\sigma_v^0\\
   && + \frac{1}{2} &e^{2k(n-1)h}e^{0k[t-(n-1)h]}[t-(n-1)h]^0v_{n-1}^0
   k^{-1}\theta^1\sigma_v^0,\\
   %
   E[I\!E_{n-1,t}I_{n-1,t}|v_{n-1}]
   &=& &e^{k(n-1)h} e^{k[t-(n-1)h]}[t-(n-1)h]^0v_{n-1}^0
   k^{-1}\theta^1\sigma_v^0\\
   && +&e^{k(n-1)h} e^{0k[t-(n-1)h]}[t-(n-1)h]^1v_{n-1}^1
   k^{-0}\theta^0\sigma_v^0\\
   && -&e^{k(n-1)h} e^{0k[t-(n-1)h]}[t-(n-1)h]^1v_{n-1}^0
   k^{-0}\theta^1\sigma_v^0\\
   && -&e^{k(n-1)h} e^{0k[t-(n-1)h]}[t-(n-1)h]^0v_{n-1}^0
   k^{-1}\theta^1\sigma_v^0,\\
   %
   E[I_{n-1,t}^2|v_{n-1}]
   &=&-&e^{0k(n-1)h} e^{-k[t-(n-1)h]}[t-(n-1)h]^0v_{n-1}^1
   k^{-1}\theta^0\sigma_v^0\\
   && +&e^{0k(n-1)h} e^{-k[t-(n-1)h]}[t-(n-1)h]^0v_{n-1}^0
   k^{-1}\theta^1\sigma_v^0\\
   && +&e^{0k(n-1)h} e^{0k[t-(n-1)h]}[t-(n-1)h]^1v_{n-1}^0
   k^{-0}\theta^1\sigma_v^0\\
   && +&e^{0k(n-1)h} e^{0k[t-(n-1)h]}[t-(n-1)h]^0v_{n-1}^1
   k^{-1}\theta^0\sigma_v^0\\
   && -&e^{0k(n-1)h} e^{0k[t-(n-1)h]}[t-(n-1)h]^0v_{n-1}^0
   k^{-1}\theta^1\sigma_v^0.


Moments
==============

Central Moments
----------------

.. math::
   :label: moment_y_central
   
    E[\overline{y}_{n}^l] 
    &= \sum_{\boldsymbol{n}} c(\boldsymbol{n})b(\boldsymbol{n})E\left[v_{n-1}^{n_2}(e^{-knh}I\!E_{n})^{n_3}I_{n}^{n_4}I_{n}^{*n_5}\right]\\
    &=\sum_{\boldsymbol{n}} c(\boldsymbol{n})b(\boldsymbol{n})E\left[v_{n-1}^{n_2}e^{-n_3knh}E[I\!E_{n}^{n_3}I_{n}^{n_4}I_{n}^{*n_5}|v_{n-1}]\right]

where :math:`\boldsymbol{n} = (n_1,n_2,n_3,n_4,n_5)` and 
:math:`\sum_{i=1}^5n_i=l`,

.. math::
   :label: c-n
   
    c(\boldsymbol{n}) = C_{l}^{n_1}C_{l-n_1}^{n_2}C_{l-n_1-n_2}^{n_3}C_{l-n_1-n_2-n_3}^{n_4}

.. math::
   :label: b-n
   
    b(\boldsymbol{n})
    = \theta^{n_1}\cdot(-1)^{n_2}\cdot\left(\frac{1-e^{-kh}}{2k}\right)^{n_1+n_2}\cdot \left(\frac{\sigma_v}{2k}\right)^{n_3} \cdot \left(\rho - \frac{\sigma_v}{2k} \right)^{n_4} \cdot \left(\sqrt{1-\rho^2}\right)^{n_5}

Equation :eq:`b-n` is further represented as a 
:py:class:`~ajdmom.poly.Poly` with 

* attribute :code:`keyfor = 
  ('e^{-kh}', 'k^{-}', 'theta', 'sigma_v', 'rho', 'sqrt(1-rho^2)')`,
* :code:`key` = :math:`(i,n_1+n_2+n_3+j,n_1,n_3+j,n_4-j,n_5)` and 
* :code:`value` = 
  :math:`C_{n_1+n_2}^i C_{n_4}^j (-1)^{n_2+i+j} \frac{1}{2^{n_1+n_2+n_3+j}}`,

i.e.,

.. math::
   
   b(\boldsymbol{n})
   &= \sum_{i=0}^{n_1+n_2} \sum_{j=0}^{n_4} C_{n_1+n_2}^i C_{n_4}^j 
      (-1)^{n_2+i+j} \frac{1}{2^{n_1+n_2+n_3+j}} \\
   &\quad e^{-ikh} k^{-(n_1+n_2+n_3+j)}\theta^{n_1}\sigma_v^{n_3+j}\rho^{n_4-j}
      \left(\sqrt{1-\rho^2}\right)^{n_5}.

And we have

.. math::
   
   e^{-n_3knh}E[I\!E_{n}^{n_3}I_{n}^{n_4}I_{n}^{*n_5}|v_{n-1}]
   = \left(e^{-n_3kt}E[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]\right)_{t=nh}.

Implementation:

1. Define :py:func:`~ajdmom.mdl_1fsv.cmom.c_n` and
   :py:func:`~ajdmom.mdl_1fsv.cmom.b_n` in 
   :py:mod:`ajdmom.mdl_1fsv.cmom` 
   to implement equation :eq:`c-n` and :eq:`b-n`, respectively.

2. Define :py:func:`~ajdmom.mdl_1fsv.cmom.moment_comb` for computing 
   the moment under an exact combination of :math:`(n_1,n_2,n_3,n_4,n_5)`.

3. Define :py:func:`~ajdmom.mdl_1fsv.cmom.sub_v` 
   and :py:func:`~ajdmom.mdl_1fsv.cmom.cmoment_y` for computing 
   the central moment :math:`E[\overline{y}_{n}^l]`. 


Moments
--------

.. math::
   :label: moment_y
   
    E[y_{n}^l] 
    &= \sum_{\boldsymbol{n}} c(\boldsymbol{n})b_2(\boldsymbol{n})E\left[v_{n-1}^{n_2}(e^{-knh}I\!E_{n})^{n_3}I_{n}^{n_4}I_{n}^{*n_5}\right]\\
    &=\sum_{\boldsymbol{n}} c(\boldsymbol{n})b_2(\boldsymbol{n})E\left[v_{n-1}^{n_2}e^{-n_3knh}E[I\!E_{n}^{n_3}I_{n}^{n_4}I_{n}^{*n_5}|v_{n-1}]\right]

where :math:`\boldsymbol{n}` and :math:`c(\boldsymbol{n})` are the same as these
in :eq:`moment_y_central` while

.. math::
   :label: b2-n
   
    b_2(\boldsymbol{n})
    &= \left[(\mu-\theta/2)h + \frac{1-e^{-kh}}{2k}\theta\right]^{n_1}\cdot(-1)^{n_2}\cdot\left(\frac{1-e^{-kh}}{2k}\right)^{n_2}\\
    &\quad \cdot \left(\frac{\sigma_v}{2k}\right)^{n_3} \cdot \left(\rho - \frac{\sigma_v}{2k} \right)^{n_4} \cdot \left(\sqrt{1-\rho^2}\right)^{n_5}.

Implementation:

1. Define :py:func:`~ajdmom.mdl_1fsv.mom.b_n` in module
   :py:mod:`ajdmom.mdl_1fsv.mom` to implement 
   equation :eq:`b2-n`.

2. Define :py:func:`~ajdmom.mdl_1fsv.mom.moment_comb`
   in module :py:mod:`ajdmom.mdl_1fsv.mom` as a counterpart 
   of :py:func:`~ajdmom.mdl_1fsv.cmom.moment_comb` in
   :py:mod:`ajdmom.mdl_1fsv.cmom`.

3. Define :py:func:`~ajdmom.mdl_1fsv.mom.moment_y` for computing 
   the moment :math:`E[y_n^l]`.


Covariances
==================

.. math::
   :label: cov_yy
   
   cov(y_n^{l_1},y_{n+1}^{l_2})
   = E[y_n^{l_1}y_{n+1}^{l_2}] 
    - E[y_n^{l_1}]E[y_{n+1}^{l_2}],

in which :math:`E[y_n^{l_1}]` and :math:`E[y_{n+1}^{l_2}]`
can be computed through :py:func:`~ajdmom.mdl_1fsv.mom.moment_y`
in module :py:mod:`ajdmom.mdl_1fsv.mom`.
Thus, I only need to present the computation of 
:math:`E[y_n^{l_1}y_{n+1}^{l_2}].`

Co-Moments 
-----------

.. math::
   :label: moment_yy
   
    &E[y_n^{l_1}y_{n+1}^{l_2}]\\
    &= \sum_{\boldsymbol{n}}c(\boldsymbol{n})b_2(\boldsymbol{n})E[y_n^{l_1} v_n^{n_2}e^{-n_3k(n+1)h}I\!E_{n+1}^{n_3} I_{n+1}^{n_4} I_{n+1}^{*n_5}]\\
    &= \sum_{\boldsymbol{n}}c(\boldsymbol{n})b_2(\boldsymbol{n})E[y_n^{l_1}\color{teal} v_n^{n_2}e^{-n_3k(n+1)h}E[I\!E_{n+1}^{n_3} I_{n+1}^{n_4} I_{n+1}^{*n_5}|v_n]]\\
    &= \sum_{\boldsymbol{n}}c(\boldsymbol{n})b_2(\boldsymbol{n})E[y_n^{l_1}\color{teal} \text{ve_IEII_vn}(n_2, n_3, n_4, n_5)]\\
    &= \sum_{\boldsymbol{n}}c(\boldsymbol{n})b_2(\boldsymbol{n})\color{magenta}\sum_{\boldsymbol{m}}c(\boldsymbol{m})b_2(\boldsymbol{m})E[v_{n-1}^{m_2}e^{-m_3knh}I\!E_n^{m_3}I_n^{m_4}I_n^{*m_5} \color{teal}\text{ve_IEII_vn}(n_2, n_3, n_4, n_5)]

where I used

.. math::
   
    y_n^{l_1} 
    &= \sum_{\boldsymbol{m}}c(\boldsymbol{m})b_2(\boldsymbol{m})v_{n-1}^{m_2}e^{-m_3knh}I\!E_n^{m_3}I_n^{m_4}I_n^{*m_5},\\
    y_{n+1}^{l_2} 
    &= \sum_{\boldsymbol{n}}c(\boldsymbol{n})b_2(\boldsymbol{n})v_{n}^{n_2}e^{-n_3k(n+1)h}I\!E_{n+1}^{n_3}I_{n+1}^{n_4}I_{n+1}^{*n_5}.

Note that 

.. math::
   
   E[I\!E_{n+1}^{n_3} I_{n+1}^{n_4} I_{n+1}^{*n_5}|v_n]
   &= \sum_{n_3,i,j,l,o,p,q} b_{n_3ijlopq} e^{n_3knh} e^{ikh}
   h^jv_{n}^l k^{-o}\theta^p\sigma_v^q,\\
   v_n^{n_2}e^{-n_3k(n+1)h}E[I\!E_{n+1}^{n_3} I_{n+1}^{n_4} I_{n+1}^{*n_5}|v_n]
   &= \sum_{n_3,i,j,l,o,p,q} b_{n_3ijlopq} e^{-n_3kh} e^{ikh}
   h^jv_{n}^{l+n_2} k^{-o}\theta^p\sigma_v^q.

Function :py:func:`~ajdmom.mdl_1fsv.cov.ve_IEII_vn` is defined to accomplish
above computation and expand :math:`v_n` which returns a poly with 
:code:`keyfor
= (‘e^{-knh}eI_n’,‘e^{-kh}’,‘h’,‘v_{n-1}’,‘k^{-}’,‘theta’,‘sigma_v’)`, i.e.,

.. math::
   
   \text{ve_IEII_vn}(n_2, n_3, n_4, n_5)
   =\sum_{m,i,j,l,o,p,q}b_{mijlopq}e^{-mknh}I\!E_n^m e^{-ikh} h^jv_{n-1}^l
   k^{-o}\theta^p\sigma_v^q.

The expansion of :math:`v_n` is done through, 

.. math::
   :label: vn-expanded
   
   v_n 
   &= e^{-kh}v_{n-1} + (1 - e^{-kh})\theta + \sigma_v e^{-knh}I\!E_{n},\\
   v_n^m 
   &= \sum_{\boldsymbol{m}} c_v(\boldsymbol{m}) b_v(\boldsymbol{m}) \cdot 
   v_{n-1}^{m_1}(e^{-knh}I\!E_n)^{m_3},

(taking :math:`v_n^m` as an example), where 
:math:`\boldsymbol{m} = (m_1,m_2,m_3)`, :math:`m_1+m_2+m_3 = m`, and

.. math::
   
   c_v(\boldsymbol{m})
   \triangleq C_m^{m_1}C_{m-m_1}^{m_2},
   \quad
   b_v(\boldsymbol{m})
   \triangleq e^{-m_1 kh} \cdot [(1-e^{-kh})\theta]^{m_2} \cdot \sigma_v^{m_3}.

Implementation:

1. Define :py:func:`~ajdmom.mdl_1fsv.cov.ve_IEII_vn` in module
   :py:mod:`ajdmom.mdl_1fsv.cov`.

2. Define :py:func:`~ajdmom.mdl_1fsv.cov.moment_inner_comb` 
   (in module :py:mod:`ajdmom.mdl_1fsv.cov`) to compute 
   the moment when the inner combination 
   :math:`\boldsymbol{m}=(m_1,m_2,m_3,m_4,m_5)` is
   fixed under an exact outer combination 
   :math:`\boldsymbol{n}=(n_1,n_2,n_3,n_4,n_5)`.

3. Define :py:func:`~ajdmom.mdl_1fsv.cov.moment_outer_comb` 
   (in module :py:mod:`ajdmom.mdl_1fsv.cov`) to compute the moment when
   only the combination of the :math:`\boldsymbol{n}=(n_1,n_2,n_3,n_4,n_5)`, 
   :math:`\sum_{i=1}^5n_i=l_2` is given.

4. Define :py:func:`~ajdmom.mdl_1fsv.cov.moment_yy` 
   (in module :py:mod:`ajdmom.mdl_1fsv.cov`) for 
   equation :eq:`moment_yy`.

5. Define :py:func:`~ajdmom.mdl_1fsv.cov.cov_yy` 
   (in module :py:mod:`ajdmom.mdl_1fsv.cov`)
   for equation :eq:`cov_yy`.

'''
Itô process moments under Superposition of Two Square-Root Diffusion Processes

See :doc:`../2fsv` for superposition of two square-root diffusion processes.
I will demonstrate how to compute

.. math::
   :label: IEI_IEII

   E[m_4m_5m_6m_7m_8]
   \\triangleq
   E[I\!E_{1,n-1,t}^{m_4} I_{1,n-1,t}^{m_5} I\!E_{2,n-1,t}^{m_6} I_{2,n-1,t}^{m_7}
   I_{n-1,t}^{*m_8}|v_{1,n-1},v_{2,n-1}].

Result
-------

The result is presented first.
Function :py:func:`~ajdmom.itos_mom.moment_IEI_IEII` is defined to compute
equation :eq:`IEI_IEII` which returns a :py:class:`~ajdmom.poly.Poly`
with attribute

:code:`keyfor = ('((n_1m,n_2m,i_m),...,(n_11,n_21,i_1))',
'e^{(m_4*k1+m_6*k2)(n-1)h}','e^{(j_1*k1+j_2*k2)h}','h','v_{1,n-1}','theta1',
'sigma_v1','v_{2,n-1}','theta2','sigma_v2')`,

i.e., with key components standing for

* :code:`key[0]`: :math:`((n_{1m},n_{2m},i_{m}),...,(n_{11},n_{21},i_{1}))` for
  :math:`(n_{1m}k_1+n_{2m}k_2)^{-i_m}\cdots (n_{11}k_1+n_{21}k_2)^{-i_1}`,

* :code:`key[1]`: :math:`(m_4,m_6)` for :math:`e^{(m_4k_1+m_6k_2)(n-1)h}`

* :code:`key[2]`: :math:`(j_1,j_2)` for :math:`e^{(j_1k_1+j_2k_2)[t-(n-1)h]}`,

* :code:`key[3]`: :math:`i` for :math:`[t-(n-1)h]^i`,

* :code:`key[4],key[5],key[6]`: :math:`v_{1,n-1}, \\theta_1, \sigma_{v1}`
  raised to the respective power,

* :code:`key[7],key[8],key[9]`: :math:`v_{2,n-1}, \\theta_2, \sigma_{v2}`
  raised to the respective power.


Therefore, I write the result of equation :eq:`IEI_IEII` as

.. math::

   &E[m_4m_5m_6m_7m_8]\\\\
   &= \sum_{t0,(m_4,m_6),(i,i'),j,l,p,q,l',p',q'}
   b_{t0(m_4,m_6)(i,i')jlpql'p'q'} \cdot \\\\
   &\quad (n_{1m}k_1+n_{2m}k_2)^{-i_m}
   \cdots (n_{11}k_1+n_{21}k_2)^{-i_1}\cdot
   e^{(m_4k_1+m_6k_2)(n-1)h}\cdot\\\\
   &\quad e^{(ik_1+i'k_2)[t-(n-1)h]} [t-(n-1)h]^{j}
   v_{1,n-1}^{l}\\theta_1^{p}\sigma_{v1}^{q}
   v_{2,n-1}^{l'}\\theta_2^{p'}\sigma_{v2}^{q'}

where :math:`t0 = ((n_{1m},n_{2m},i_{m}),...,(n_{11},n_{21},i_{1}))`.

**Note that**:
:math:`E[I\!E_{1,n}^{m_4} I_{1,n}^{m_5} I\!E_{2,n}^{m_6} I_{2,n}^{m_7}
I_{n}^{*m_8}|v_{1,n-1},v_{2,n-1}] = E[m_4m_5m_6m_7m_8|_{t=nh}]`.

I will show the deduction process in what follows.

Deduction
----------

In order to compute equation
:eq:`IEI_IEII`, I expand it by taking derivative as the following equation
shows

.. math::

   &d(I\!E_{1,n-1,t}^{m_4} I_{1,n-1,t}^{m_5} I\!E_{2,n-1,t}^{m_6} I_{2,n-1,t}^{m_7}
   I_{n-1,t}^{*m_8})\\\\
   &\\approx \\frac{1}{2}m_8(m_8-1)I\!E_{1,n-1,t}^{m_4} I_{1,n-1,t}^{m_5} I\!E_{2,n-1,t}^{m_6} I_{2,n-1,t}^{m_7}
   I_{n-1,t}^{*m_8-2})v(t)dt\\\\
   &\quad + c_1(t) I\!E_{2,n-1,t}^{m_6} I_{2,n-1,t}^{m_7} I_{n-1,t}^{*m_8} dt\\\\
   &\quad + c_2(t) I\!E_{1,n-1,t}^{m_4} I_{1,n-1,t}^{m_5} I_{n-1,t}^{*m_8} dt\\\\
   &\\approx \\frac{1}{2}m_8(m_8-1)I\!E_{1,n-1,t}^{m_4} I_{1,n-1,t}^{m_5} I\!E_{2,n-1,t}^{m_6} I_{2,n-1,t}^{m_7} I_{n-1,t}^{*m_8-2})v_1(t)dt\\\\
   &\quad + \\frac{1}{2}m_8(m_8-1)I\!E_{1,n-1,t}^{m_4} I_{1,n-1,t}^{m_5} I\!E_{2,n-1,t}^{m_6} I_{2,n-1,t}^{m_7} I_{n-1,t}^{*m_8-2})v_2(t)dt\\\\
   &\quad + \\frac{1}{2}m_4(m_4-1)e^{2k_1t}I\!E_{1,n-1,t}^{m_4-2}I_{1,n-1,t}^{m_5}I\!E_{2,n-1,t}^{m_6} I_{2,n-1,t}^{m_7} I_{n-1,t}^{*m_8} v_1(t)dt\\\\
   &\quad + \\frac{1}{2}m_5(m_5-1)I\!E_{1,n-1,t}^{m_4}  I_{1,n-1,t}^{m_5-2} I\!E_{2,n-1,t}^{m_6} I_{2,n-1,t}^{m_7} I_{n-1,t}^{*m_8} v_1(t)dt\\\\
   &\quad + m_4m_5e^{k_1t}I\!E_{1,n-1,t}^{m_4-1}I_{1,n-1,t}^{m_5-1}I\!E_{2,n-1,t}^{m_6} I_{2,n-1,t}^{m_7} I_{n-1,t}^{*m_8} v_1(t)dt\\\\
   &\quad + \\frac{1}{2}m_6(m_6-1)e^{2k_2t} I\!E_{2,n-1,t}^{m_6-2}I_{2,n-1,t}^{m_7}I\!E_{1,n-1,t}^{m_4} I_{1,n-1,t}^{m_5} I_{n-1,t}^{*m_8} v_2(t)dt\\\\
   &\quad + \\frac{1}{2}m_7(m_7-1)I\!E_{2,n-1,t}^{m_6}  I_{2,n-1,t}^{m_7-2}I\!E_{1,n-1,t}^{m_4} I_{1,n-1,t}^{m_5} I_{n-1,t}^{*m_8} v_2(t)dt\\\\
   &\quad + m_6m_7e^{k_2t}I\!E_{2,n-1,t}^{m_6-1}I_{2,n-1,t}^{m_7-1}I\!E_{1,n-1,t}^{m_4} I_{1,n-1,t}^{m_5} I_{n-1,t}^{*m_8} v_2(t)dt

where

.. math::

   c_1(t)
   &\\triangleq \\bigg[
     \\frac{1}{2}m_4(m_4-1)I\!E_{1,n-1,t}^{m_4-2}I_{1,n-1,t}^{m_5}e^{2k_1t}
   + \\frac{1}{2}m_5(m_5-1)I\!E_{1,n-1,t}^{m_4}  I_{1,n-1,t}^{m_5-2}\\\\
   &\qquad + m_4m_5I\!E_{1,n-1,t}^{m_4-1}I_{1,n-1,t}^{m_5-1}e^{k_1t}
   \\bigg] v_1(t),\\\\
   c_2(t)
   &\\triangleq \\bigg[
     \\frac{1}{2}m_6(m_6-1)I\!E_{2,n-1,t}^{m_6-2}I_{2,n-1,t}^{m_7}e^{2k_2t}
   + \\frac{1}{2}m_7(m_7-1)I\!E_{2,n-1,t}^{m_6}  I_{2,n-1,t}^{m_7-2}\\\\
   &\qquad + m_6m_7I\!E_{2,n-1,t}^{m_6-1}I_{2,n-1,t}^{m_7-1}e^{k_2t}
   \\bigg] v_2(t),

and

.. math::

   v_{1}(t)
   &= e^{-k_1t}e^{k_1(n-1)h}(v_{1,n-1} - \\theta_1) + \\theta_1 + \sigma_{v1} e^{-k_1t}I\!E_{1,n-1,t},\\\\
   v_{2}(t)
   &= e^{-k_2t}e^{k_2(n-1)h}(v_{2,n-1} - \\theta_2) + \\theta_2 + \sigma_{v2} e^{-k_2t}I\!E_{2,n-1,t}.

Recursive Equation
-------------------

Thus, we have the following *recursive equation*

.. math::
   :label: ito-moment-m4m5m6m7m8

   &E[m_4m_5m_6m_7m_8]&\\\\
   &= \\frac{m_4(m_4-1)}{2}e^{k_1(n-1)h}(v_{1,n-1} - \\theta_1) &\int_{(n-1)h}^t e^{k_1s}E[(m_4-2)m_5m_6m_7m_8]ds\\\\
   &\quad + \\frac{m_4(m_4-1)}{2}\\theta_1 &\int_{(n-1)h}^t e^{2k_1s}E[(m_4-2)m_5m_6m_7m_8]ds\\\\
   &\quad + \\frac{m_4(m_4-1)}{2}\sigma_{v1} &\int_{(n-1)h}^t e^{k_1s}E[(m_4-1)m_5m_6m_7m_8]ds\\\\
   &\quad + \\frac{m_5(m_5-1)}{2}e^{k_1(n-1)h}(v_{1,n-1} - \\theta_1) &\color{blue}\int_{(n-1)h}^t e^{-k_1s}E[m_4(m_5-2)m_6m_7m_8]ds\\\\
   &\quad + \\frac{m_5(m_5-1)}{2}\\theta_1 &\color{blue}\int_{(n-1)h}^t E[m_4(m_5-2)m_6m_7m_8]ds\\\\
   &\quad + \\frac{m_5(m_5-1)}{2}\sigma_{v1} &\color{blue}\int_{(n-1)h}^t e^{-k_1s}E[(m_4+1)(m_5-2)m_6m_7m_8]ds\\\\
   &\quad + m_4m_5e^{k_1(n-1)h}(v_{1,n-1} - \\theta_1) &\int_{(n-1)h}^t E[(m_4-1)(m_5-1)m_6m_7m_8]ds\\\\
   &\quad + m_4m_5\\theta_1 &\int_{(n-1)h}^t e^{k_1s}E[(m_4-1)(m_5-1)m_6m_7m_8]ds\\\\
   &\quad + m_4m_5\sigma_{v1} &\int_{(n-1)h}^t E[m_4(m_5-1)m_6m_7m_8]ds\\\\
   &\quad + \\frac{m_6(m_6-1)}{2}e^{k_2(n-1)h}(v_{2,n-1} - \\theta_2) &\color{blue}\int_{(n-1)h}^t e^{k_2s}E[m_4m_5(m_6-2)m_7m_8]ds\\\\
   &\quad + \\frac{m_6(m_6-1)}{2}\\theta_2 &\color{blue}\int_{(n-1)h}^t e^{2k_2s}E[m_4m_5(m_6-2)m_7m_8]ds\\\\
   &\quad + \\frac{m_6(m_6-1)}{2}\sigma_{v2} &\color{blue}\int_{(n-1)h}^t e^{k_2s}E[m_4m_5(m_6-1)m_7m_8]ds\\\\
   &\quad + \\frac{m_7(m_7-1)}{2}e^{k_2(n-1)h}(v_{2,n-1} - \\theta_2) &\int_{(n-1)h}^t e^{-k_2s}E[m_4m_5m_6(m_7-2)m_8]ds\\\\
   &\quad + \\frac{m_7(m_7-1)}{2}\\theta_2 &\int_{(n-1)h}^t E[m_4m_5m_6(m_7-2)m_8]ds\\\\
   &\quad + \\frac{m_7(m_7-1)}{2}\sigma_{v2} &\int_{(n-1)h}^t e^{-k_2s}E[m_4m_5(m_6+1)(m_7-2)m_8]ds\\\\
   &\quad + m_6m_7e^{k_2(n-1)h}(v_{2,n-1} - \\theta_2) &\color{blue}\int_{(n-1)h}^t E[m_4m_5(m_6-1)(m_7-1)m_8]ds\\\\
   &\quad + m_6m_7\\theta_2 &\color{blue}\int_{(n-1)h}^t e^{k_2s}E[m_4m_5(m_6-1)(m_7-1)m_8]ds\\\\
   &\quad + m_6m_7\sigma_{v2} &\color{blue}\int_{(n-1)h}^t E[m_4m_5m_6(m_7-1)m_8]ds\\\\
   &\quad + \\frac{m_8(m_8-1)}{2}e^{k_1(n-1)h}(v_{1,n-1} - \\theta_1) &\int_{(n-1)h}^t e^{-k_1s}E[m_4m_5m_6m_7(m_8-2)]ds\\\\
   &\quad + \\frac{m_8(m_8-1)}{2}\\theta_1 &\int_{(n-1)h}^t E[m_4m_5m_6m_7(m_8-2)]ds\\\\
   &\quad + \\frac{m_8(m_8-1)}{2}\sigma_{v1} &\int_{(n-1)h}^t e^{-k_1s}E[(m_4+1)m_5m_6m_7(m_8-2)]ds\\\\
   &\quad + \\frac{m_8(m_8-1)}{2}e^{k_2(n-1)h}(v_{2,n-1} - \\theta_2) &\color{blue}\int_{(n-1)h}^t e^{-k_2s}E[m_4m_5m_6m_7(m_8-2)]ds\\\\
   &\quad + \\frac{m_8(m_8-1)}{2}\\theta_2 &\color{blue}\int_{(n-1)h}^t E[m_4m_5m_6m_7(m_8-2)]ds\\\\
   &\quad + \\frac{m_8(m_8-1)}{2}\sigma_{v2} &\color{blue}\int_{(n-1)h}^t e^{-k_2s}E[m_4m_5(m_6+1)m_7(m_8-2)]ds.


Initial Moments
----------------

For order 0, i.e., :math:`m_4+\cdots+m_8=0`, :math:`E[m_4m_5m_6m_7m_8] = 1`.
And for order 1, :math:`m_4+\cdots+m_8=1`, :math:`E[m_4m_5m_6m_7m_8] = 0`.

For order 2, i.e., :math:`m_4+\cdots+m_8=2`, :math:`E[m_4m_5m_6m_7m_8] = 0`,
except for

* :math:`m_4+m_5=2`:

  .. math::

     &E[I\!E_{1,n-1,t}^2|v_{1,n-1}]\\\\
     &= e^{2k_1t}\\frac{1}{2k_1}\\theta_1 + e^{k_1t + k_1(n-1)h}\\frac{1}{k_1}
     (v_{1,n-1}-\\theta_1) - e^{2k_1(n-1)h}\\frac{1}{2k_1}\left(2v_{1,n-1}
     - \\theta_1 \\right),\\\\
     %
     &E[I\!E_{1,n-1,t}I_{1,n-1,t}|v_{1,n-1}]\\\\
     &=e^{k_1t}\\frac{1}{k_1}\\theta_1
       + e^{k_1(n-1)h}(v_{1,n-1}-\\theta_1)[t-(n-1)h]
       - e^{k_1(n-1)h}\\frac{1}{k_1}\\theta_1,\\\\
     %
     &E[I_{1,n-1,t}^2|v_{1,n-1}]\\\\
     &= -e^{-k_1t + k_1(n-1)h}\\frac{1}{k_1}(v_{1,n-1}-\\theta_1)
       + \\theta_1[t-(n-1)h] + (v_{1,n-1}-\\theta_1)\\frac{1}{k_1};

* :math:`m_6+m_7=2`:

  .. math::

     &E[I\!E_{2,n-1,t}^2|v_{2,n-1}]\\\\
     &= e^{2k_2t}\\frac{1}{2k_2}\\theta_2 + e^{k_2t + k_2(n-1)h}\\frac{1}{k_2}
     (v_{2,n-1}-\\theta_2) - e^{2k_2(n-1)h}\\frac{1}{2k_2}\left(2v_{2,n-1}
     - \\theta_2 \\right),\\\\
     %
     &E[I\!E_{2,n-1,t}I_{2,n-1,t}|v_{2,n-1}]\\\\
     &=e^{k_2t}\\frac{1}{k_2}\\theta_2
       + e^{k_2(n-1)h}(v_{2,n-1}-\\theta_2)[t-(n-1)h]
     - e^{k_2(n-1)h}\\frac{1}{k_2}\\theta_2,\\\\
     %
     &E[I_{2,n-1,t}^2|v_{2,n-1}]\\\\
     &= -e^{-k_2t + k_2(n-1)h}\\frac{1}{k_2}(v_{2,n-1}-\\theta_2)
        + \\theta_2[t-(n-1)h] + (v_{2,n-1}-\\theta_2)\\frac{1}{k_2};

* :math:`m_8=2`:

  .. math::

     &E[I_{n-1,t}^{*2}|v_{1,n-1},v_{2,n-1}]\\\\
     &= -e^{k_1(n-1)h}(v_{1,n-1}-\\theta_1)\\frac{1}{k_1}(e^{-k_1t} -
     e^{-k_1(n-1)h}) +\\theta_1 [t-(n-1)h]\\\\
     &\quad -e^{k_2(n-1)h}(v_{2,n-1}-\\theta_2)\\frac{1}{k_2}(e^{-k_2t} -
     e^{-k_2(n-1)h}) +\\theta_2 [t-(n-1)h].

Implementation
---------------

We have [#f1]_,

.. math::

   \int e^{(n_1k_1+n_2k_2)t} t^m dt =
   \\begin{cases}
   \sum_{i=0}^m c_{n_1n_2mi} e^{(n_1k_1+n_2k_2)t} t^{m-i} & \\text{if } n_1k_1+n_2k_2\\neq 0, m \\neq 0,\\\\
   \\frac{1}{n_1k_1+n_2k_2}e^{(n_1k_1+n_2k_2)t}t^0 & \\text{if } n_1k_1+n_2k_2\\neq 0, m = 0,\\\\
   \\frac{1}{m+1}e^{0kt}t^{m+1} & \\text{if } n_1k_1+n_2k_2 = 0, m \\neq 0,\\\\
   e^{0kt}t^1 & \\text{if } n_1k_1+n_2k_2 =0 , m=0,
   \end{cases}

where :math:`c_{n_1n_2m0} \\triangleq \\frac{1}{n_1k_1+n_2k_2}` and

.. math::
   :label: c-n_1n_2mi

   c_{n_1n_2mi} \\triangleq \\frac{(-1)^{i}}{(n_1k_1+n_2k_2)^{i+1}}
   \prod_{j=m-i+1}^{m} j,~~ 1\le i \le m.

The coefficient :math:`c_{n_1n_2mi}` is implemented in function
:py:func:`~ajdmom.itos_mom.c`.

For the definite integral

.. math::

   \int_{(n-1)h}^t e^{(n_1k_1+n_2k_2)[s-(n-1)h]}[s-(n-1)h]^mds
   = F(t-(n-1)h) - F(0)

which is defined in :py:func:`~ajdmom.itos_mom.int_et`,
where :math:`F(t) = \int e^{(n_1k_1+n_2k_2)t} t^m dt`.

In summary, I defined

1. :py:func:`~ajdmom.itos_mom.int_et` which uses
   :py:func:`~ajdmom.itos_mom.c`.

2. :py:func:`~ajdmom.itos_mom.recursive_IEI_IEII` which uses
   :py:func:`~ajdmom.itos_mom.int_mIEI_IEII` and
   :py:func:`~ajdmom.itos_mom.coef_poly`.

3. :py:func:`~ajdmom.itos_mom.moment_IEI_IEII`.

-------------

.. [#f1] It's assumed :math:`n_1k_1 + n_2k_2\\neq 0`.

-------------
'''
import math
from fractions import Fraction as Frac

from ajdmom.poly import Poly

def c(n1, n2, m, i):
  '''Constant :math:`c_{n_1n_2mi}` in :eq:`c-n_1n_2mi`

  :param n1: :math:`n_1` in :math:`\int_{(n-1)h}^t e^{(n_1k_1+n_2k_2)s}s^mds`.
  :param n2: :math:`n_2` in above integral.
  :param m: :math:`m` in above integral.
  :param i: :math:`i` in :math:`t^{m-i}`.

  :return: a tuple of (key,val) with key for :math:`(n_1,n_2,i)` in
     :math:`(n_1k_1+n_2k_2)^{-i}`, val for the value.
  :rtype: tuple
  '''
  prod = 1
  for j in range(m-i+1, m+1):
    prod = prod * j
  key = (n1, n2, i+1)
  val = ((-1)**i) * prod
  return(key, val)

def int_et(n1, n2, m):
  ''':math:`\int_{(n-1)h}^t e^{(n_1k_1+n_2k_2)[s-(n-1)h]}[s-(n-1)h]^mds`

  :param n1: :math:`n_1` in above integral.
  :param n2: :math:`n_2` in above integral.
  :param m: :math:`m` in above integral.

  :return: poly with attribute ``keyfor`` =
     ('(n1*k1+n2*k2)^{-}','e^{(n1*k1+n2*k2)[t-(n-1)h]}','[t-(n-1)h]').
  :rtype: Poly
  '''
  if m < 0:
    msg = f"m in int_et(n1,n2,m) equals {m}, however it shouldn't be negative!"
    raise ValueError(msg)
  #
  poly = Poly()
  kf = ['(n1*k1+n2*k2)^{-}','e^{(n1*k1+n2*k2)[t-(n-1)h]}','[t-(n-1)h]']
  poly.set_keyfor(kf)
  #
  if n1+n2 == 0 and m == 0:
    key = ((0,0,0),(0,0),1)
    val = 1
    poly.add_keyval(key, val)
  elif n1+n2 == 0 and m != 0:
    key = ((0,0,0),(0,0),m+1)
    val = Frac(1, m+1)
    poly.add_keyval(key, val)
  elif n1+n2 != 0 and m == 0:
    key = ((n1,n2,1),(n1,n2),0)
    val = 1
    poly.add_keyval(key, val)
    # - F(0)
    key = ((n1,n2,1),(0,0),0)
    poly.add_keyval(key, -val)
  else:
    key = ((n1,n2,1),(n1,n2),m)
    val = 1
    poly.add_keyval(key, val)
    for i in range(1, m+1):
      kei, val = c(n1, n2, m, i)
      key = (kei,(n1,n2),m-i)
      poly.add_keyval(key, val)
      if i == m: # - F(0)
        key = (kei,(0,0),0)
        poly.add_keyval(key, -val)
  return(poly)

def t_mul_t0(t, t0):
  '''multiply quant t0 with quant t

  :param t: a tuple (n1,n2,i) standing for '(n1*k1+n2*k2)^{-i}'.
  :param t0: a tuple of tuple,
     '(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}'

  :return: a tuple of tuple.
  :rtype: tuple
  '''
  n1, n2, i = t
  if i == 0: return(t0) # t represent 1
  #
  l0 = list(t0)
  # check whether match any existing denumerator
  for n in range(len(t0)):
    m1, m2, j = t0[n]
    if n1 == m1 and n2 == m2: # same denumerator 1/(n1*k1 + n2*k2)
      l0[n] = (n1, n2, i+j) # l0[n][2] = i + j error since l0[n] is a tuple
      return(tuple(l0))
  # none match
  if l0[0][2] != 0: # not 1, such as not being (0,0,0)
    l0.insert(0,t)
  else:             # be  1, such as (0,0,0), then replace it with t
    l0[0] = t
  return(tuple(l0))

def int_mIEI_IEII(i, m, n4, n5, n6, n7, n8, IEI_IEII):
  ''':math:`\int_{(n-1)h}^te^{mk_is}E[n_4n_5n_6n_7n_8]ds`

  :param i: :math:`i` in :math:`e^{mk_is} (i=1,2)`.
  :param m: :math:`m` in :math:`e^{mk_is} (i=1,2)`.
  :param n4: :math:`m_4` in :math:`I\!E_{1,n-1,t}^{m_4}`.
  :param n5: :math:`m_5` in :math:`I_{1,n-1,t}^{m_5}`.
  :param n6: :math:`m_6` in :math:`I\!E_{2,n-1,t}^{m_6}`.
  :param n7: :math:`m_7` in :math:`I_{2,n-1,t}^{m_7}`.
  :param n8: :math:`m_8` in :math:`I_{1,n-1,t}^{*m_8}`.
  :param IEI_IEII: a dict with key (n4,n5,n6,n7,n8) and value poly with
     attribute ``keyfor`` =
     ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
     'e^{(m_4*k1+m_6*k2)(n-1)h}','e^{(j_1*k1+j_2*k2)[t-(n-1)h]}','[t-(n-1)h]',
     'v_{1,n-1}','theta1','sigma_v1', 'v_{2,n-1}','theta2','sigma_v2').

  :return: poly with attribute ``keyfor`` =
     ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
     'e^{(m_4*k1+m_6*k2)(n-1)h}','e^{(j_1*k1+j_2*k2)[t-(n-1)h]}','[t-(n-1)h]',
     'v_{1,n-1}','theta1','sigma_v1', 'v_{2,n-1}','theta2','sigma_v2').
  :rtype: Poly
  '''
  b = IEI_IEII[(n4,n5,n6,n7,n8)]
  #
  poly = Poly()
  kf = ['(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
     'e^{(m_4*k1+m_6*k2)(n-1)h}','e^{(j_1*k1+j_2*k2)[t-(n-1)h]}','[t-(n-1)h]',
     'v_{1,n-1}','theta1','sigma_v1', 'v_{2,n-1}','theta2','sigma_v2']
  poly.set_keyfor(kf)
  #
  for k1 in b:
    m4, m6 = k1[2]
    if i == 1:
      n1 = m4 + m; n2 = m6
    else:
      n2 = m6 + m; n1 = m4
    c = int_et(n1, n2, k1[3])
    # ['(n1*k1+n2*k2)^{-}','e^{(n1*k1+n2*k2)[t-(n-1)h]}','[t-(n-1)h]']
    for k2 in c:
      # t0 = list(k1[0]); t0.insert(0,k2[0]); t0 = tuple(t0)
      t0 = t_mul_t0(k2[0], k1[0])
      # compensate e^{mki[s-(n-1)h]} for e^{ki(n-1)h}
      if i == 1:
        t1 = (k1[1][0]+m, k1[1][1])
      else:
        t1 = (k1[1][0], k1[1][1]+m)
      #
      t2 = k2[1]
      t3 = k2[2]
      key = (t0,t1,t2,t3, k1[4],k1[5],k1[6],k1[7],k1[8],k1[9])
      val = b[k1] * c[k2]
      poly.add_keyval(key, val)
  return(poly)

def coef_poly(coef, poly, tp):
  '''Multiply poly with different type coefficients

  :param coef: integer, such as the leading :math:`m_4(m_4-1)/2`.
  :param poly: poly returned by :py:func:`~ajdmom.itos_mom.int_mIEI_IEII`.
  :param tp: type of the multiplication,

     +---+-------------------------------+---+-------------------------------+
     |tp |multiply with                  |tp |multiply with                  |
     +===+===============================+===+===============================+
     | 1 |:math:`e^{k_1(n-1)h}v_{1,n-1}` | 5 |:math:`e^{k_2(n-1)h}v_{2,n-1}` |
     +---+-------------------------------+---+-------------------------------+
     | 2 |:math:`-e^{k_1(n-1)h}\\theta_1` | 6 |:math:`-e^{k_2(n-1)h}\\theta_2` |
     +---+-------------------------------+---+-------------------------------+
     | 3 |:math:`\\theta_1`               | 7 |:math:`\\theta_2`               |
     +---+-------------------------------+---+-------------------------------+
     | 4 |:math:`\\sigma_{v1}`            | 8 |:math:`\\sigma_{v2}`            |
     +---+-------------------------------+---+-------------------------------+

  :return: poly with attribute ``keyfor`` =
     ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
     'e^{(m_4*k1+m_6*k2)(n-1)h}','e^{(j_1*k1+j_2*k2)[t-(n-1)h]}','[t-(n-1)h]',
     'v_{1,n-1}','theta1','sigma_v1', 'v_{2,n-1}','theta2','sigma_v2').
  :rtype: Poly
  '''
  poln = Poly()
  kf = ['(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
     'e^{(m_4*k1+m_6*k2)(n-1)h}','e^{(j_1*k1+j_2*k2)[t-(n-1)h]}','[t-(n-1)h]',
     'v_{1,n-1}','theta1','sigma_v1', 'v_{2,n-1}','theta2','sigma_v2']
  poln.set_keyfor(kf)
  #
  if tp == 1:
    for k in poly:
      t1 = (k[1][0]+1, k[1][1])
      key = (k[0],t1,k[2],k[3],k[4]+1,k[5],k[6],k[7],k[8],k[9])
      val = coef * poly[k]
      poln.add_keyval(key, val)
  if tp == 2:
    for k in poly:
      t1 = (k[1][0]+1, k[1][1])
      key = (k[0],t1,k[2],k[3],k[4],k[5]+1,k[6],k[7],k[8],k[9])
      val = (-coef) * poly[k]
      poln.add_keyval(key, val)
  if tp == 3:
    for k in poly:
      key = (k[0],k[1],k[2],k[3],k[4],k[5]+1,k[6],k[7],k[8],k[9])
      val = coef * poly[k]
      poln.add_keyval(key, val)
  if tp == 4:
    for k in poly:
      key = (k[0],k[1],k[2],k[3],k[4],k[5],k[6]+1,k[7],k[8],k[9])
      val = coef * poly[k]
      poln.add_keyval(key, val)
  #
  if tp == 5:
    for k in poly:
      t1 = (k[1][0], k[1][1]+1)
      key = (k[0],t1,k[2],k[3],k[4],k[5],k[6],k[7]+1,k[8],k[9])
      val = coef * poly[k]
      poln.add_keyval(key, val)
  if tp == 6:
    for k in poly:
      t1 = (k[1][0], k[1][1]+1)
      key = (k[0],t1,k[2],k[3],k[4],k[5],k[6],k[7],k[8]+1,k[9])
      val = (-coef) * poly[k]
      poln.add_keyval(key, val)
  if tp == 7:
    for k in poly:
      key = (k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7],k[8]+1,k[9])
      val = coef * poly[k]
      poln.add_keyval(key, val)
  if tp == 8:
    for k in poly:
      key = (k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7],k[8],k[9]+1)
      val = coef * poly[k]
      poln.add_keyval(key, val)
  return(poln)

def recursive_IEI_IEII(n4, n5, n6, n7, n8, IEI_IEII):
  '''Recursive equation :eq:`ito-moment-m4m5m6m7m8`

  :param n4: :math:`m_4` in :math:`I\!E_{1,n-1,t}^{m_4}`.
  :param n5: :math:`m_5` in :math:`I_{1,n-1,t}^{m_5}`.
  :param n6: :math:`m_6` in :math:`I\!E_{2,n-1,t}^{m_6}`.
  :param n7: :math:`m_7` in :math:`I_{2,n-1,t}^{m_7}`.
  :param n8: :math:`m_8` in :math:`I_{1,n-1,t}^{*m_8}`.
  :param IEI_IEII: a dict with key (n4,n5,n6,n7,n8) and value poly with
     attribute ``keyfor`` =
     ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
     'e^{(m_4*k1+m_6*k2)(n-1)h}','e^{(j_1*k1+j_2*k2)[t-(n-1)h]}','[t-(n-1)h]',
     'v_{1,n-1}','theta1','sigma_v1', 'v_{2,n-1}','theta2','sigma_v2').

  :return: poly with attribute ``keyfor`` =
     ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
     'e^{(m_4*k1+m_6*k2)(n-1)h}','e^{(j_1*k1+j_2*k2)[t-(n-1)h]}','[t-(n-1)h]',
     'v_{1,n-1}','theta1','sigma_v1', 'v_{2,n-1}','theta2','sigma_v2').
  :rtype: Poly
  '''
  poly = Poly()
  kf = ['(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
     'e^{(m_4*k1+m_6*k2)(n-1)h}','e^{(j_1*k1+j_2*k2)[t-(n-1)h]}','[t-(n-1)h]',
     'v_{1,n-1}','theta1','sigma_v1', 'v_{2,n-1}','theta2','sigma_v2']
  poly.set_keyfor(kf)
  #
  if n4 >= 2 and n5 >= 0 and n6 >= 0 and n7 >= 0 and n8 >= 0:
    c = Frac(n4*(n4-1), 2)
    poly.merge(coef_poly(c, int_mIEI_IEII(1, 1, n4-2,n5,n6,n7,n8,IEI_IEII), 1))
    poly.merge(coef_poly(c, int_mIEI_IEII(1, 1, n4-2,n5,n6,n7,n8,IEI_IEII), 2))
    poly.merge(coef_poly(c, int_mIEI_IEII(1, 2, n4-2,n5,n6,n7,n8,IEI_IEII), 3))
    poly.merge(coef_poly(c, int_mIEI_IEII(1, 1, n4-1,n5,n6,n7,n8,IEI_IEII), 4))
  elif n4 >= 0 and n5 >=2 and n6 >= 0 and n7 >= 0 and n8 >= 0:
    c = Frac(n5*(n5-1), 2)
    poly.merge(coef_poly(c, int_mIEI_IEII(1,-1, n4,n5-2,n6,n7,n8,IEI_IEII), 1))
    poly.merge(coef_poly(c, int_mIEI_IEII(1,-1, n4,n5-2,n6,n7,n8,IEI_IEII), 2))
    poly.merge(coef_poly(c, int_mIEI_IEII(1, 0, n4,n5-2,n6,n7,n8,IEI_IEII), 3))
    poly.merge(coef_poly(c, int_mIEI_IEII(1,-1, n4+1,n5-2,n6,n7,n8,IEI_IEII),4))
  elif n4 >= 1 and n5 >= 1 and n6 >= 0 and n7 >= 0 and n8 >= 0:
    c = n4 * n5
    poly.merge(coef_poly(c, int_mIEI_IEII(1,0, n4-1,n5-1,n6,n7,n8,IEI_IEII),1))
    poly.merge(coef_poly(c, int_mIEI_IEII(1,0, n4-1,n5-1,n6,n7,n8,IEI_IEII),2))
    poly.merge(coef_poly(c, int_mIEI_IEII(1,1, n4-1,n5-1,n6,n7,n8,IEI_IEII),3))
    poly.merge(coef_poly(c, int_mIEI_IEII(1,0, n4  ,n5-1,n6,n7,n8,IEI_IEII),4))
  #
  elif n4 >= 0 and n5 >= 0 and n6 >= 2 and n7 >= 0 and n8 >= 0:
    c = Frac(n6*(n6-1), 2)
    poly.merge(coef_poly(c, int_mIEI_IEII(2, 1, n4,n5,n6-2,n7,n8,IEI_IEII), 5))
    poly.merge(coef_poly(c, int_mIEI_IEII(2, 1, n4,n5,n6-2,n7,n8,IEI_IEII), 6))
    poly.merge(coef_poly(c, int_mIEI_IEII(2, 2, n4,n5,n6-2,n7,n8,IEI_IEII), 7))
    poly.merge(coef_poly(c, int_mIEI_IEII(2, 1, n4,n5,n6-1,n7,n8,IEI_IEII), 8))
  elif n4 >= 0 and n5 >= 0 and n6 >= 0 and n7 >= 2 and n8 >= 0:
    c = Frac(n7*(n7-1), 2)
    poly.merge(coef_poly(c, int_mIEI_IEII(2,-1, n4,n5,n6,n7-2,n8,IEI_IEII), 5))
    poly.merge(coef_poly(c, int_mIEI_IEII(2,-1, n4,n5,n6,n7-2,n8,IEI_IEII), 6))
    poly.merge(coef_poly(c, int_mIEI_IEII(2, 0, n4,n5,n6,n7-2,n8,IEI_IEII), 7))
    poly.merge(coef_poly(c, int_mIEI_IEII(2,-1, n4,n5,n6+1,n7-2,n8,IEI_IEII),8))
  elif n4 >= 0 and n5 >= 0 and n6 >= 1 and n7 >= 1 and n8 >= 0:
    c = n6 * n7
    poly.merge(coef_poly(c, int_mIEI_IEII(2, 0, n4,n5,n6-1,n7-1,n8,IEI_IEII),5))
    poly.merge(coef_poly(c, int_mIEI_IEII(2, 0, n4,n5,n6-1,n7-1,n8,IEI_IEII),6))
    poly.merge(coef_poly(c, int_mIEI_IEII(2, 1, n4,n5,n6-1,n7-1,n8,IEI_IEII),7))
    poly.merge(coef_poly(c, int_mIEI_IEII(2, 0, n4,n5,n6,  n7-1,n8,IEI_IEII),8))
  #
  elif n4 >= 0 and n5 >= 0 and n6 >= 0 and n7 >= 0 and n8 >= 2:
    c = Frac(n8*(n8-1), 2)
    poly.merge(coef_poly(c, int_mIEI_IEII(1,-1, n4,n5,n6,n7,n8-2,IEI_IEII),1))
    poly.merge(coef_poly(c, int_mIEI_IEII(1,-1, n4,n5,n6,n7,n8-2,IEI_IEII),2))
    poly.merge(coef_poly(c, int_mIEI_IEII(1, 0, n4,n5,n6,n7,n8-2,IEI_IEII),3))
    poly.merge(coef_poly(c, int_mIEI_IEII(1,-1,n4+1,n5,n6,n7,n8-2,IEI_IEII),4))
    #
    poly.merge(coef_poly(c, int_mIEI_IEII(2,-1, n4,n5,n6,n7,n8-2,IEI_IEII),5))
    poly.merge(coef_poly(c, int_mIEI_IEII(2,-1, n4,n5,n6,n7,n8-2,IEI_IEII),6))
    poly.merge(coef_poly(c, int_mIEI_IEII(2, 0, n4,n5,n6,n7,n8-2,IEI_IEII),7))
    poly.merge(coef_poly(c, int_mIEI_IEII(2,-1, n4,n5,n6+1,n7,n8-2,IEI_IEII),8))
  #
  return(poly)

def moment_IEI_IEII(n4, n5, n6, n7, n8, return_all=False):
  '''Moment of :math:`E[m_4m_5m_6m_7m_8]`

  Moment of the combination of Ito processes, conditioning on the previous
  volatility components, i.e.,

  .. math::

     E[I\!E_{1,n-1,t}^{m_4} I_{1,n-1,t}^{m_5} I\!E_{2,n-1,t}^{m_6}
     I_{2,n-1,t}^{m_7}I_{n-1,t}^{*m_8}|v_{1,n-1},v_{2,n-1}]

  :param n4: :math:`m_4` in :math:`I\!E_{1,n-1,t}^{m_4}`.
  :param n5: :math:`m_5` in :math:`I_{1,n-1,t}^{m_5}`.
  :param n6: :math:`m_6` in :math:`I\!E_{2,n-1,t}^{m_6}`.
  :param n7: :math:`m_7` in :math:`I_{2,n-1,t}^{m_7}`.
  :param n8: :math:`m_8` in :math:`I_{1,n-1,t}^{*m_8}`.
  :param return_all: whether or not return lower order moments simultaneously,
     defaults to ``False``.

  :return: poly if return_all=False else IEI_IEII
  :rtype: Poly or dict of Poly
  '''
  if n4+n5+n6+n7+n8 < 0:
    raise ValueError(f'moment_IEI_IEII({n_4},{n_5},{n_6},{n_7},{n_8}) is called')
  #
  IEI_IEII = {}
  kf = ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
     'e^{(m_4*k1+m_6*k2)(n-1)h}','e^{(j_1*k1+j_2*k2)[t-(n-1)h]}','[t-(n-1)h]',
     'v_{1,n-1}','theta1','sigma_v1', 'v_{2,n-1}','theta2','sigma_v2')
  # n4+n5+n6+n7+n8 = 0
  poly = Poly({(((0,0,0),),(0,0),(0,0),0, 0,0,0,0,0,0): 1})
  poly.set_keyfor(kf)
  IEI_IEII[(0,0,0,0,0)] = poly
  # n4+n5+n6+n7+n8 = 1
  poly = Poly(); poly.set_keyfor(kf)
  IEI_IEII[(1,0,0,0,0)] = poly
  IEI_IEII[(0,1,0,0,0)] = poly
  IEI_IEII[(0,0,1,0,0)] = poly
  IEI_IEII[(0,0,0,1,0)] = poly
  IEI_IEII[(0,0,0,0,1)] = poly
  # n4+n5+n6+n7+n8 = 2
  #   n4+n5 = 2
  poly = Poly({
    (((2,0,1),),(2,0),(2,0),0, 0,1,0,0,0,0): 1,
    (((1,0,1),),(2,0),(1,0),0, 1,0,0,0,0,0): 1,
    (((1,0,1),),(2,0),(1,0),0, 0,1,0,0,0,0): -1,
    (((1,0,1),),(2,0),(0,0),0, 1,0,0,0,0,0): -1,
    (((2,0,1),),(2,0),(0,0),0, 0,1,0,0,0,0): 1
  })
  poly.set_keyfor(kf)
  IEI_IEII[(2,0,0,0,0)] = poly
  #
  poly = Poly({
    (((1,0,1),),(1,0),(1,0),0, 0,1,0,0,0,0): 1,
    (((0,0,0),),(1,0),(0,0),1, 1,0,0,0,0,0): 1,
    (((0,0,0),),(1,0),(0,0),1, 0,1,0,0,0,0): -1,
    (((1,0,1),),(1,0),(0,0),0, 0,1,0,0,0,0): -1
  })
  poly.set_keyfor(kf)
  IEI_IEII[(1,1,0,0,0)] = poly
  #
  poly = Poly({
    (((1,0,1),),(0,0),(-1,0),0, 1,0,0,0,0,0): -1,
    (((1,0,1),),(0,0),(-1,0),0, 0,1,0,0,0,0): 1,
    (((0,0,0),),(0,0),(0,0), 1, 0,1,0,0,0,0): 1,
    (((1,0,1),),(0,0),(0,0), 0, 1,0,0,0,0,0): 1,
    (((1,0,1),),(0,0),(0,0), 0, 0,1,0,0,0,0): -1
  })
  poly.set_keyfor(kf)
  IEI_IEII[(0,2,0,0,0)] = poly
  #  n4+n5 = 1
  poly = Poly(); poly.set_keyfor(kf)
  IEI_IEII[(1,0,1,0,0)] = poly
  IEI_IEII[(1,0,0,1,0)] = poly
  IEI_IEII[(1,0,0,0,1)] = poly
  #
  IEI_IEII[(0,1,1,0,0)] = poly
  IEI_IEII[(0,1,0,1,0)] = poly
  IEI_IEII[(0,1,0,0,1)] = poly
  #  n6+n7 = 2
  poly = Poly({
    (((0,2,1),),(0,2),(0,2),0, 0,0,0,0,1,0): 1,
    (((0,1,1),),(0,2),(0,1),0, 0,0,0,1,0,0): 1,
    (((0,1,1),),(0,2),(0,1),0, 0,0,0,0,1,0): -1,
    (((0,1,1),),(0,2),(0,0),0, 0,0,0,1,0,0): -1,
    (((0,2,1),),(0,2),(0,0),0, 0,0,0,0,1,0): 1
  })
  poly.set_keyfor(kf)
  IEI_IEII[(0,0,2,0,0)] = poly
  #
  poly = Poly({
    (((0,1,1),),(0,1),(0,1),0, 0,0,0,0,1,0): 1,
    (((0,0,0),),(0,1),(0,0),1, 0,0,0,1,0,0): 1,
    (((0,0,0),),(0,1),(0,0),1, 0,0,0,0,1,0): -1,
    (((0,1,1),),(0,1),(0,0),0, 0,0,0,0,1,0): -1
  })
  poly.set_keyfor(kf)
  IEI_IEII[(0,0,1,1,0)] = poly
  #
  poly = Poly({
    (((0,1,1),),(0,0),(0,-1),0, 0,0,0,1,0,0): -1,
    (((0,1,1),),(0,0),(0,-1),0, 0,0,0,0,1,0): 1,
    (((0,0,0),),(0,0),(0,0), 1, 0,0,0,0,1,0): 1,
    (((0,1,1),),(0,0),(0,0), 0, 0,0,0,1,0,0): 1,
    (((0,1,1),),(0,0),(0,0), 0, 0,0,0,0,1,0): -1
  })
  poly.set_keyfor(kf)
  IEI_IEII[(0,0,0,2,0)] = poly
  #  n6+n7 = 1
  poly = Poly(); poly.set_keyfor(kf)
  IEI_IEII[(0,0,1,0,1)] = poly
  IEI_IEII[(0,0,0,1,1)] = poly
  # n8 = 2
  pol1 = IEI_IEII[(0,2,0,0,0)].copy()
  pol2 = IEI_IEII[(0,0,0,2,0)].copy()
  pol3 = pol1 # pol1 will change
  pol3.merge(pol2) # pol3 will change
  IEI_IEII[(0,0,0,0,2)] = pol3
  #
  if n4+n5+n6+n7+n8 <= 2:
    return(IEI_IEII if return_all else IEI_IEII[(n4,n5,n6,n7,n8)])
  #
  if n4+n5+n6+n7+n8 > 3:
    # compute all lower-order moments to get ready
    for n in range(3, n4+n5+n6+n7+n8):
      for i4 in range(n, -1, -1):
        for i5 in range(n-i4, -1, -1):
          for i6 in range(n-i4-i5, -1, -1):
            for i7 in range(n-i4-i5-i6, -1, -1):
              i8 = n-i4-i5-i6-i7
              poly = recursive_IEI_IEII(i4,i5,i6,i7,i8, IEI_IEII)
              poly.remove_zero()
              IEI_IEII[(i4,i5,i6,i7,i8)] = poly
  # the last step
  poly = recursive_IEI_IEII(n4,n5,n6,n7,n8, IEI_IEII)
  poly.remove_zero()
  IEI_IEII[(n4,n5,n6,n7,n8)] = poly
  return(IEI_IEII if return_all else poly)

if __name__ == "__main__":
  # Example usage of the module, see 'tests/test_itos_mom.py' for more test
  from pprint import pprint
  print('\nExample usage of the module functions\n')
  #
  kf = '(key,val) with \nkey = (n1,n2,i) for (n_1k_1+n_2k_2)^{-i} '
  kf += 'and \nval = the numerator'
  print(f'c(n1,n2,m,i) returns a tuple of {kf}')
  print(f'c(1,2,1,1) = {c(1,2,1,1)}\n')
  #
  kf = "('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',\n"
  kf += "'e^{(m_4*k1+m_6*k2)(n-1)h}','e^{(j_1*k1+j_2*k2)[t-(n-1)h]}',"
  kf += "'[t-(n-1)h]',\n"
  kf += "'v_{1,n-1}','theta1','sigma_v1', 'v_{2,n-1}','theta2','sigma_v2')"
  print(f"moment_IEI_IEII(n4,n5,n6,n7,n8) returns a poly with keyfor =\n{kf}")
  print('moment_IEI_IEII(3,0,0,0,0) = ')
  pprint(moment_IEI_IEII(n4=3,n5=0,n6=0,n7=0,n8=0))

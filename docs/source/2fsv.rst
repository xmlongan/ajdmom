=====================================================
:abbr:`2FSV(Two-Factor Stochastic Volatility)` Model
=====================================================

In this subpackage (``ajdmom.mdl_2fsv``), we consider the following
:abbr:`SV(Stochastic Volatility)` model in which volatility is
a superposition of two :abbr:`SRD(Square-Root Diffusion)` processes,

.. math::
   
    d\log s(t) &= (\mu- v(t)/2) dt + \sqrt{v(t)}dw(t),\\
    v(t)       &= v_1(t) + v_2(t),\\
    dv_1(t)    &= k_1(\theta_1 - v_1(t))dt + \sigma_{1v} \sqrt{v_1(t)}dw_1(t),\\
    dv_2(t)    &= k_2(\theta_2 - v_2(t))dt + \sigma_{2v} \sqrt{v_2(t)}dw_2(t),

where :math:`w_1(t)`, :math:`w_2(t)` are two independent Wiener processes, 
which are also independent of :math:`w(t)`.

Notations
==========

We define

.. math::
   
    I_{1,n} &\triangleq \int_{(n-1)h}^{nh}\sqrt{v_1(t)}dw_1(t),
    &I_{2,n} &\triangleq \int_{(n-1)h}^{nh}\sqrt{v_2(t)}dw_2(t),\\
    I\!E_{1,n} &\triangleq \int_{(n-1)h}^{nh}e^{k_1t}\sqrt{v_1(t)}dw_1(t),\quad
    &I\!E_{2,n} &\triangleq \int_{(n-1)h}^{nh}e^{k_2t}\sqrt{v_2(t)}dw_2(t),\\
    IV_{1,n}&\triangleq \int_{(n-1)h}^{nh}v_1(t)dt,
    &IV_{2,n}&\triangleq \int_{(n-1)h}^{nh}v_2(t)dt,\\
    IV_{n} &\triangleq IV_{1,n} + IV_{2,n},
    &I_n^* &\triangleq \int_{(n-1)h}^{nh}\sqrt{v(t)}dw(t).

Return of the *n*\ th interval :math:`y_n` can be expressed as
:math:`y_n = \mu h - \frac{1}{2}IV_n + I_n^*`.
We have

.. math::
   
   IV_{i,n} 
   = (h-\tilde{h}_i)\theta_i + \tilde{h}v_{i,n-1} - 
   \frac{\sigma_{vi}}{k_i}e^{-k_inh}I\!E_{i,n} + \frac{\sigma_{vi}}{k_i}I_{i,n}, 
   i=1,2,

where :math:`\tilde{h}_i \triangleq (1-e^{-k_ih})/k_i, i=1,2`.


Central Moments
================

Similarly, we define :math:`\overline{y}_n \triangleq y_n - E[y_n]` and we have

.. math::
   
   \overline{y}_n
   &=\frac{1}{2}(\tilde{h}_1\theta_1 + \tilde{h}_2\theta_2)
   - \frac{1}{2}\tilde{h}_1v_{1,n-1} - \frac{1}{2}\tilde{h}_2v_{2,n-1}\\
   &\quad + \frac{\sigma_{v1}}{2k_1}e^{-k_1 nh}I\!E_{1,n} 
    - \frac{\sigma_{v1}}{2k_1}I_{1,n}
    + \frac{\sigma_{v2}}{2k_2}e^{-k_2 nh}I\!E_{2,n}
    - \frac{\sigma_{v2}}{2k_2}I_{2,n}
    + I_{n}^{*}.

According to above expansion, central moment :math:`\overline{y}_n` with order
:math:`l` reduces to

.. math::
   
   &E[\overline{y}_n^l]\\
   &= \sum_{\boldsymbol{m}}c(\boldsymbol{m})b(\boldsymbol{m})
   E[v_{1,n-1}^{m_2}v_{2,n-1}^{m_3} (e^{-k_1 nh}I\!E_{1,n})^{m_4} I_{1,n}^{m_5}
    (e^{-k_2 nh}I\!E_{2,n})^{m_6} I_{2,n}^{m_7} I_{n}^{*m_8}]\\
   &= \sum_{\boldsymbol{m}}c(\boldsymbol{m})b(\boldsymbol{m})
   E[v_{1,n-1}^{m_2}v_{2,n-1}^{m_3}e^{-m_4k_1 nh}e^{-m_6k_2 nh} 
   I\!E_{1,n}^{m_4} I_{1,n}^{m_5} I\!E_{2,n}^{m_6} I_{2,n}^{m_7} I_{n}^{*m_8}]\\
   &= \sum_{\boldsymbol{m}}c(\boldsymbol{m})b(\boldsymbol{m})
   E[v_{1,n-1}^{m_2}v_{2,n-1}^{m_3}e^{-m_4k_1 nh}e^{-m_6k_2 nh}\\
   &\qquad \times \text{moment_IEI_IEII}(m_4,m_5,m_6,m_7,m_8)]\\
   &= \sum_{\boldsymbol{m}}c(\boldsymbol{m})b(\boldsymbol{m})
   \times \text{vvee_IEI_IEII}(m_2, m_3, m_4, m_5, m_6, m_7, m_8)\\
   &= \sum_{\boldsymbol{m}} \text{moment_comb}(l,m_1,m_2,m_3,m_4,m_5,m_6,m_7,m_8)

where :math:`\boldsymbol{m} = (m_1,\cdots,m_8), \sum_{i=1}^8m_i = l`,

.. math::
   
   c(\boldsymbol{m}) 
   &= C_{l}^{m_1}C_{l-m_1}^{m_2}C_{l-m_1-m_2}^{m_3}
   C_{l-m_1-m_2-m_3}^{m_4} C_{l-m_1-m_2-m_3-m_4}^{m_5} 
   C_{l-m_1-m_2-m_3-m_4-m_5}^{m_6} \\
   &\quad C_{l-m_1-m_2-m_3-m_4-m_5-m_6}^{m_7},

.. math::
   
   b(\boldsymbol{m})
   &= (-1)^{m_2+m_3+m_5+m_7}2^{-(l-m_8)} 
      (\tilde{h}_1\theta_1+\tilde{h}_2\theta_2)^{m_1}
      \tilde{h}_1^{m_2} \tilde{h}_2^{m_3}
      (\sigma_{v1}/k_1)^{m_4+m_5} (\sigma_{v2}/k_2)^{m_6+m_7}\\
   &=\sum_{i_1+i_2+i_3+i_4=m_1}\sum_{j_1=0}^{m_2}\sum_{j_2=0}^{m_3}
     c_1(\boldsymbol{i},\boldsymbol{j})
     (-1)^{i_2+i_4+m_2+m_3+j_1+j_2+m_5+m_7}2^{-(l-m_8)}\\
   &\quad e^{-[(i_2+j_1)k_1 + (i_4+j_2)k_2]h}
          k_1^{-(i_1+i_2+m_2+m_4+m_5)} k_2^{-(i_3+i_4+m_3+m_6+m_7)}
          \theta_1^{i_1+i_2}\theta_2^{i_3+i_4} 
          \sigma_{v1}^{m_4+m_5} \sigma_{v2}^{m_6+m_7}

where 

.. math::
   
   c_1(\boldsymbol{i},\boldsymbol{j})
   = C_{m_1}^{i_1}C_{m_1-i_1}^{i_2}C_{m_1-i_1-i_2}^{i_3}
      C_{m_2}^{j_1} C_{m_3}^{j_2}.

Function
:code:`moment_IEI_IEII(m_4,m_5,m_6,m_7,m_8)` returns a poly with attribute 
:code:`keyfor = ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
'e^{(m_4*k1+m_6*k2)(n-1)h}','e^{(j_1*k1+j_2*k2)[t-(n-1)h]}','[t-(n-1)h]',
'v_{1,n-1}','theta1','sigma_v1', 'v_{2,n-1}','theta2','sigma_v2')`.

In summary, I defined

1. :py:func:`~ajdmom.mdl_2fsv.cmom.b_n`,

2. :py:func:`~ajdmom.mdl_2fsv.cmom.vvee_IEI_IEII`,

3. :py:func:`~ajdmom.mdl_2fsv.cmom.moment_comb`,

4. :py:func:`~ajdmom.mdl_2fsv.cmom.sub_v`,

5. :py:func:`~ajdmom.mdl_2fsv.cmom.cmoment_y`


Moments
========

We have :math:`y_n = \overline{y}_n + E[y_n]` and 
:math:`E[y_n] = \mu h  - \frac{1}{2}(\theta_1 + \theta_2)h`, thus

.. math::
   
   y_n 
   &= \frac{1}{2}(\tilde{h}_1\theta_1 + \tilde{h}_2\theta_2)
   - \frac{1}{2}(\theta_1 + \theta_2)h + \mu h 
   - \frac{1}{2}\tilde{h}_1v_{1,n-1} - \frac{1}{2}\tilde{h}_2v_{2,n-1}\\
   &\quad + \frac{\sigma_{v1}}{2k_1}e^{-k_1 nh}I\!E_{1,n} 
    - \frac{\sigma_{v1}}{2k_1}I_{1,n}
    + \frac{\sigma_{v2}}{2k_2}e^{-k_2 nh}I\!E_{2,n}
    - \frac{\sigma_{v2}}{2k_2}I_{2,n}
    + I_{n}^{*}.

Similarly,

.. math::
   
   E[y_n^l]
   &= \sum_{\boldsymbol{m}}c(\boldsymbol{m})b_2(\boldsymbol{m})
   E[v_{1,n-1}^{m_2}v_{2,n-1}^{m_3} (e^{-k_1 nh}I\!E_{1,n})^{m_4} I_{1,n}^{m_5}
    (e^{-k_2 nh}I\!E_{2,n})^{m_6} I_{2,n}^{m_7} I_{n}^{*m_8}]\\
   &= \sum_{\boldsymbol{m}}c(\boldsymbol{m})b_2(\boldsymbol{m})
   E[v_{1,n-1}^{m_2}v_{2,n-1}^{m_3}e^{-m_4k_1 nh}e^{-m_6k_2 nh} 
   I\!E_{1,n}^{m_4} I_{1,n}^{m_5} I\!E_{2,n}^{m_6} I_{2,n}^{m_7} I_{n}^{*m_8}]\\
   &= \sum_{\boldsymbol{m}}c(\boldsymbol{m})b_2(\boldsymbol{m})
   E[v_{1,n-1}^{m_2}v_{2,n-1}^{m_3}e^{-m_4k_1 nh}e^{-m_6k_2 nh}\\
   &\qquad \times \text{moment_IEI_IEII}(m_4,m_5,m_6,m_7,m_8)]\\
   &= \sum_{\boldsymbol{m}}c(\boldsymbol{m})b_2(\boldsymbol{m})
   \times \text{vvee_IEI_IEII}(m_2, m_3, m_4, m_5, m_6, m_7, m_8)\\
   &= \sum_{\boldsymbol{m}} \text{moment_comb}(l,m_1,m_2,m_3,m_4,m_5,m_6,m_7,m_8)

where 

.. math::
   
   b_2(\boldsymbol{m})
   &= (-1)^{m_2+m_3+m_5+m_7}2^{-(l-m_8)} 
      (\tilde{h}_1\theta_1+\tilde{h}_2\theta_2 - h\theta_1 -h\theta_2
       +2\mu h)^{m_1}
      \tilde{h}_1^{m_2} \tilde{h}_2^{m_3}\\
   &\quad (\sigma_{v1}/k_1)^{m_4+m_5} (\sigma_{v2}/k_2)^{m_6+m_7}\\
   &=\sum_{i_1+i_2+i_3+i_4+i_5+i_6+i_7=m_1}\sum_{j_1=0}^{m_2}\sum_{j_2=0}^{m_3}
    c_2(\boldsymbol{i},\boldsymbol{j})
     (-1)^{i_2+i_4+i_5+i_6+m_2+m_3+j_1+j_2+m_5+m_7} 2^{-(l-m_8)+i_7}\\
   &\quad e^{-[(i_2+j_1)k_1 + (i_4+j_2)k_2]h}
          k_1^{-(i_1+i_2+m_2+m_4+m_5)} k_2^{-(i_3+i_4+m_3+m_6+m_7)}
          \theta_1^{i_1+i_2+i_5}\theta_2^{i_3+i_4+i_6} \\
   &\quad \sigma_{v1}^{m_4+m_5} \sigma_{v2}^{m_6+m_7} 
          h^{i_5+i_6+i_7}\mu^{i_7}

where 

.. math::
   
   c_2(\boldsymbol{i},\boldsymbol{j})
   = C_{m_1}^{i_1}C_{m_1-i_1}^{i_2}C_{m_1-i_1-i_2}^{i_3}
      C_{m_1-i_1-i_2-i_3}^{i_4}C_{m_1-i_1-i_2-i_3-i_4}^{i_5}
      C_{m_1-i_1-i_2-i_3-i_4-i_5}^{i_6}
      C_{m_2}^{j_1} C_{m_3}^{j_2}.

In summary, I defined

1. :py:func:`~ajdmom.mdl_2fsv.mom.b_n`,

2. :py:func:`~ajdmom.mdl_2fsv.mom.moment_comb`,

3. :py:func:`~ajdmom.mdl_2fsv.mom.sub_v`,

4. :py:func:`~ajdmom.mdl_2fsv.mom.moment_y`.


One alternative way,

.. math::
   
   E[y_n^l]
   = \sum_{i=0}^l C_l^i E[\overline{y}_n^i] E^{l-i}[y_n],
   \quad
   E^l[y_n]
   = \sum_{i,j} C_l^i C_{l-i}^j (-1)^{l-i} \frac{1}{2^{l-i}} h^l \mu^i \theta_1^j \theta_2^{l-i-j}.


Covariances
============

.. math::
   
   cov(y_n^{l_1}, y_{n+1}^{l_2})
   = E[y_n^{l_1}y_{n+1}^{l_2}] - E[y_n^{l_1}]E[y_{n+1}^{l_2}]

Co-Moments
-----------

.. math::
   
   &E[y_n^{l_1}y_{n+1}^{l_2}]\\
   &= \sum_{\boldsymbol{n}}c(\boldsymbol{n})b_2(\boldsymbol{n})
      E[y_n^{l_1}v_{1,n}^{n_2}v_{2,n}^{n_3} (e^{-k_1 (n+1)h}I\!E_{1,n+1})^{n_4} 
      I_{1,n+1}^{n_5} (e^{-k_2 (n+1)h}I\!E_{2,n+1})^{n_6} I_{2,n+1}^{n_7} 
      I_{n+1}^{*n_8}]\\
   &= \sum_{\boldsymbol{n}}c(\boldsymbol{n})b_2(\boldsymbol{n})
      E[y_n^{l_1}\color{teal}v_{1,n}^{n_2}v_{2,n}^{n_3}e^{-n_4k_1(n+1)h}
      e^{-n_6k_2(n+1)h} \\
   &\quad \color{teal} E[I\!E_{1,n+1}^{n_4} I_{1,n+1}^{n_5} I\!E_{2,n+1}^{n_6}
      I_{2,n+1}^{n_7}I_{n+1}^{*n_8}|v_{1,n},v_{2,n}]]\\
   &= \sum_{\boldsymbol{n}}c(\boldsymbol{n})b_2(\boldsymbol{n})
      E[y_n^{l_1}\color{teal} 
      \text{vvee_IEI_IEII_vnvn}(n_2,n_3,n_4,n_5,n_6,n_7,n_8)]\\
   &= \sum_{\boldsymbol{n}}c(\boldsymbol{n})b_2(\boldsymbol{n})
      \color{magenta}
      \sum_{\boldsymbol{m}}c(\boldsymbol{m})b_2(\boldsymbol{m})
      E[ v_{1,n-1}^{m_2}v_{2,n-1}^{m_3}e^{-m_4k_1nh}e^{-m_6k_2nh}I\!E_{1,n}^{m_4} 
      I_{1,n}^{m_5} I\!E_{2,n}^{m_6} I_{2,n}^{m_7} I_{n}^{*m_8}\\
   &\quad \color{teal} 
      \text{vvee_IEI_IEII_vnvn}(n_2,n_3,n_4,n_5,n_6,n_7,n_8)]

where I used

.. math::
   
   y_n^{l_1}
   &= \sum_{\boldsymbol{m}}c(\boldsymbol{m})b_2(\boldsymbol{m})
      v_{1,n-1}^{m_2}v_{2,n-1}^{m_3}e^{-m_4k_1nh}e^{-m_6k_2nh}I\!E_{1,n}^{m_4} 
      I_{1,n}^{m_5} I\!E_{2,n}^{m_6} I_{2,n}^{m_7} I_{n}^{*m_8},\\
   y_{n+1}^{l_2}
   &= \sum_{\boldsymbol{n}}c(\boldsymbol{n})b_2(\boldsymbol{n})
      v_{1,n}^{n_2}v_{2,n}^{n_3}e^{-n_4k_1(n+1)h}e^{-n_6k_2(n+1)h}
      I\!E_{1,n+1}^{n_4} I_{1,n+1}^{n_5} I\!E_{2,n+1}^{n_6} I_{2,n+1}^{n_7} 
      I_{n+1}^{*n_8}.

Note that

.. math::
   
   &E[I\!E_{1,n+1}^{n_4} I_{1,n+1}^{n_5} I\!E_{2,n+1}^{n_6} I_{2,n+1}^{n_7} 
      I_{n+1}^{*n_8}|v_{1,n},v_{2,n}]\\
   &= \sum_{t0,(n_4,n_6),(i,i'),j,l,p,q,l',p',q'} 
   b_{t0(n_4,n_6)(i,i')jlpql'p'q'} \cdot \\
   &\quad (n_{1m}k_1+n_{2m}k_2)^{-i_m} 
   \cdots (n_{11}k_1+n_{21}k_2)^{-i_1}\cdot
   e^{(n_4k_1+n_6k_2)nh}\cdot\\
   &\quad e^{(ik_1+i'k_2)(t-nh)} (t-nh)^{j}
   v_{1,n}^{l}\theta_1^{p}\sigma_{v1}^{q} 
   v_{2,n}^{l'}\theta_2^{p'}\sigma_{v2}^{q'},\\
   %
   &E[v_{1,n}^{n_2}v_{2,n}^{n_3}e^{-n_4k_1(n+1)h}e^{-n_6k_2(n+1)h}
      I\!E_{1,n+1}^{n_4} I_{1,n+1}^{n_5} I\!E_{2,n+1}^{n_6} I_{2,n+1}^{n_7} 
      I_{n+1}^{*n_8}|v_{1,n},v_{2,n}]\\
   &= \sum_{t0,(n_4,n_6),(i,i'),j,l,p,q,l',p',q'} 
   b_{t0(n_4,n_6)(i,i')jlpql'p'q'} \cdot \\
   &\quad (n_{1m}k_1+n_{2m}k_2)^{-i_m} 
   \cdots (n_{11}k_1+n_{21}k_2)^{-i_1}\cdot
   e^{-(n_4k_1+n_6k_2)h}\cdot\\
   &\quad e^{(ik_1+i'k_2)(t-nh)} (t-nh)^{j}
   v_{1,n}^{l+n_2}\theta_1^{p}\sigma_{v1}^{q} 
   v_{2,n}^{l'+n_3}\theta_2^{p'}\sigma_{v2}^{q'},

where :math:`t0 = ((n_{1m},n_{2m},i_{m}),...,(n_{11},n_{21},i_{1}))` and
:math:`t=(n+1)h`.

Function :py:func:`~ajdmom.mdl_2fsv.cov.vvee_IEI_IEII_vnvn` 
is defined to 
accomplish above computation and expand :math:`v_{1,n}` and
:math:`v_{2,n}`
which returns a poly with attribute 
:code:`keyfor = ('e^{-k1*nh}IE_{1,n}','e^{-k2*nh}IE_{2,n}',
'(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
'e^{-(n1*k1+n2*k2)h}','h',
'v_{1,n-1}','theta1','sigma_v1', 'v_{2,n-1}','theta2','sigma_v2')`, i.e.,

.. math::
   
   &\text{vvee_IEI_IEII_vnvn}(n_2,n_3,n_4,n_5,n_6,n_7,n_8)\\
   &=\sum_{o,o',t0,i,i',j,l,p,q,l',p',q'}b_{oo't0(i,i')jlpql'p'q'}
     e^{-ok_1nh}I\!E_{1,n}^o e^{-o'k_2nh}I\!E_{2,n}^{o'}\\
   &\quad (n_{1m}k_1+n_{2m}k_2)^{-i_m} \cdots (n_{11}k_1+n_{21}k_2)^{-i_1}
     e^{-(ik_1+i'k_2)h} h^j v_{1,n-1}^{l}\theta_1^{p}\sigma_{v1}^{q}
     v_{2,n-1}^{l'}\theta_2^{p'}\sigma_{v2}^{q'}.

Expansion of :math:`v_{1,n}` is done through

.. math::
   
   v_{1,n} 
   &= e^{-k_1h}v_{1,n-1} + (1 - e^{-k_1h})\theta_1 + \sigma_{v1} 
      e^{-k_1nh}I\!E_{1,n},\\
   v_{1,n}^m 
   &= \sum_{\boldsymbol{m}} c_v(\boldsymbol{m}) b_v(\boldsymbol{m}) \cdot 
      v_{1,n-1}^{m_1}(e^{-k_1nh}I\!E_{1,n})^{m_3},

(taking :math:`v_{1,n}^m` as an example), where 
:math:`\boldsymbol{m} = (m_1,m_2,m_3)`, :math:`m_1+m_2+m_3 = m`, and

.. math::
   
   c_v(\boldsymbol{m})
   \triangleq C_m^{m_1}C_{m-m_1}^{m_2},
   \quad
   b_v(\boldsymbol{m})
   \triangleq e^{-m_1 k_1h} \cdot [(1-e^{-k_1h})\theta_1]^{m_2} \cdot
   \sigma_{v1}^{m_3}.

Expansion of :math:`v_{2,n}` is done similarly. 

In summary, I defined

1. :py:func:`~ajdmom.mdl_2fsv.cov.vvee_IEI_IEII_vnvn`,

2. :py:func:`~ajdmom.mdl_2fsv.cov.moment_inner_comb`,

3. :py:func:`~ajdmom.mdl_2fsv.cov.moment_outer_comb`,

4. :py:func:`~ajdmom.mdl_2fsv.cov.moment_yy`,

5. :py:func:`~ajdmom.mdl_2fsv.cov.cov_yy`.


API
====

.. autosummary::
   :toctree: generated
   
   ajdmom.mdl_2fsv.cmom
   ajdmom.mdl_2fsv.mom
   ajdmom.mdl_2fsv.cov
   ajdmom.mdl_2fsv.euler


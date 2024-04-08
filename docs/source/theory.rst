Theory
========

:abbr:`AJD(Affine Jump Diffusion)` Model
-----------------------------------------

AJD process is a class of n-dimensional Markov processes, denoted by 
:math:`\boldsymbol{x}(t)`, in some state space :math:`D\subset \mathbb{R}^n` and described by the following stochastic differential equation: 

.. math::
   d \boldsymbol{x}(t) = \boldsymbol{\mu}(\boldsymbol{x}(t))dt + \boldsymbol{\sigma}(\boldsymbol{x}(t))d \boldsymbol{w}(t) + d\boldsymbol{z}(t),

where the driving processes :math:`\boldsymbol{w}(t)` is an n-dimensional standard Wiener process, and :math:`\boldsymbol{z}(t)` is an inhomogeneous Compound Poisson Process (CPP) whose jumps are distributed according to :math:`F_{\boldsymbol{j}}(\cdot)` on :math:`\mathbb{R}^n` and arrive with intensity :math:`\lambda(\boldsymbol{x}(t)): D\rightarrow \mathbb{R}_{\geqslant 0}`.
All of the drift :math:`\boldsymbol{\mu}(\cdot)`, instantaneous covariance matrix :math:`\boldsymbol{\sigma}(\cdot)\boldsymbol{\sigma}(\cdot)^T` and jump intensity :math:`\lambda(\cdot)` have affine dependence on the state vector :math:`\boldsymbol{x}(t)`, i.e., with the same notations as in Duffie et al. (2000) [#f1]_ , determined by coefficients :math:`(K,H,l)` as

* :math:`\boldsymbol{\mu}(\boldsymbol{x}) = K_0 + K_1\boldsymbol{x}`, for :math:`K = (K_0,K_1)\in \mathbb{R}^n\times \mathbb{R}^{n\times n}`.
* :math:`(\boldsymbol{\sigma}(\boldsymbol{x})\boldsymbol{\sigma}(\boldsymbol{x})^T)_{ij} = (H_0)_{ij} + (H_1)_{ij}\cdot \boldsymbol{x}`, for :math:`H=(H_0,H_1)\in \mathbb{R}^{n\times n}\times \mathbb{R}^{n\times n\times n}`.
* :math:`\lambda(\boldsymbol{x}) = l_0 + l_1\cdot \boldsymbol{x}`, for :math:`l=(l_0,l_1)\in \mathbb{R}\times\mathbb{R}^{n}`.


Since AJD models mainly find applications in a range of financial asset valuation and econometric problems due to its tractability, we take the Heston SV model and its AJD extensions as examples. Under the settings of SV models, including the Heston one, only part of the state is observable because the other part of the state, i.e., volatility, is usually latent. The partially observed state is no longer Markovian and the tractability of the Heston SV models has been limited to only having a closed-form conditional :abbr:`CF(Characteristic Function)`. However, we have found a recursive way to derive the moments and covariances in closed-form of the Heston SV models and its AJD extensions. 

As a typical :abbr:`AJD(Affine Jump Diffusion)` process, the Heston :abbr:`SV(Stochastic Volatility)` model is seen as a baseline model hereafter. 
And I will focus on the derivation of moments and covariances for this baseline model.

Heston :abbr:`SV(Stochastic Volatility)` Model
-----------------------------------------------

The so-called Heston :abbr:`SV(Stochastic Volatility)` in this package 
is described by the
following :abbr:`SDEs(Stochastic Differential Equations)` [#f2]_ ,

.. math::
    ds(t) &= \mu s(t)dt + \sqrt{v(t)}s(t)dw^s(t),\\
    dv(t) &= k(\theta - v(t))dt + \sigma_v\sqrt{v(t)}dw^v(t),

where :math:`s(t)` is the asset price at time :math:`t`, :math:`v(t)` is the
instantaneous return variance at time :math:`t`, and :math:`w^s(t)` and 
:math:`w^v(t)` are two Wiener processes with correlation :math:`\rho`. 
Assume that the initial values :math:`s(0)` and :math:`v(0)` are independent 
of each other and also independent of :math:`w^s(t)` and :math:`w^v(t)`.  
The variance process :math:`v(t)` is a :abbr:`CIR(Cox-Ingersoll-Ross)` process, 
which is also called square-root diffusion. The parameters 
:math:`\theta>0, k>0,\sigma_v>0`
determine the long-run mean, the mean reversion velocity, and the volatility 
of the variance process :math:`v(t)`, respectively, and 
satisfy the condition :math:`\sigma_v^2 \le 2k\theta`.

The variance process :math:`v(t)` is a Markov  process 
that has a steady-state gamma distribution with mean :math:`\theta` and 
variance :math:`\theta \sigma_v^2/(2k)`, e.g., see [#f3]_ .  
Without loss of generality, throughout this package we assume that 
:math:`v(0)` is distributed according to the steady-state distribution of 
:math:`v(t)`, which implies that :math:`v(t)` is strictly stationary and 
ergodic (see [#f4]_ ).

Based on Itô formula, the log price process can be written as:

.. math::
   d\log s(t) = (\mu-\frac{1}{2}v(t))dt + \sqrt{v(t)}dw^s(t).


Let :math:`s_i \triangleq s(ih)` [#f5]_ . For the return over the
*i*\ th interval, denoted by :math:`y_i`, it's defined as 

.. math::
   y_i \triangleq \log s_i - \log s_{i-1}.


Notations
----------

I decompose :math:`w^s(t)` as 
:math:`w^s(t) = \rho w^v(t) + \sqrt{1-\rho^2}w(t)`, where :math:`w(t)` is 
another Wiener process which is independent of
:math:`w^v(t)`. 
For notational simplicity, I define:

.. math::
   
   I_{s,t} \triangleq & \int_{s}^{t}\sqrt{v(u)}dw^v(u), \quad
   & I_{s,t}^* \triangleq & \int_{s}^{t}\sqrt{v(u)}dw(u),\\
   I\!E_{s,t} \triangleq & \int_{s}^{t}e^{ku}\sqrt{v(u)}dw^v(u),\quad
   & IV_{s,t} \triangleq & \int_{s}^{t}v(u)du,

and

.. math::
   
   I_i&\triangleq I_{(i-1)h,ih}, 
   &I_i^*&\triangleq I^*_{(i-1)h,ih},
   &I\!E_i&\triangleq I\!E_{(i-1)h,ih},
   &IV_i&\triangleq IV_{(i-1)h,ih},\\
   I_{i,t}&\triangleq I_{ih,t},
   &I^*_{i,t}&\triangleq I^*_{ih,t},
   &I\!E_{i,t}&\triangleq I\!E_{ih,t},
   &IV_{i,t}&\triangleq IV_{ih,t}.

Usually, :math:`IV_{s,t}, IV_i`, and :math:`IV_{i,t}` are referred to 
as *Integrated Variance* (volatility).  

Then, :math:`y_i` can be expressed as

.. math::
   
   y_i = \mu h - \frac{1}{2}IV_i + \rho I_i + \sqrt{1-\rho^2} I_i^*.

Variance (volatility) process :math:`v(t)` can be re-written as:

.. math::
   
   v(t) = e^{-k(t-s)}v(s)  +\theta \left[1-e^{-k(t-s)}\right] +
   \sigma_v e^{-kt}I\!E_{s,t},

whose moment of order :math:`m` is given as

.. _volatility-moments:

.. math::
   :label: moment-v
   
   E[v(t)^m] = \prod_{j=0}^{m-1}\left(\theta + \frac{j\sigma_v^2}{2k}\right).

Integrated Variance can be re-written as

.. math::
   
   IV_{s,t} = \theta (t-s) - \frac{v(t) - v(s)}{k} + \frac{\sigma_v}{k} I_{s,t}.

Moment Derivation
------------------

Here I discuss how moments and covariances of :math:`y_n` can be
derived.  Define 

.. math::
   
    y_{n-1,t} 
    \triangleq \mu [t-(n-1)h] - \frac{1}{2}IV_{n-1,t} + \rho I_{n-1,t} + 
    \sqrt{1-\rho^2}I_{n-1,t}^*,

then

.. math::
   
   \bar{y}_{n-1,t}
   = \beta_{n-1,t}\textcolor{blue}{\theta} - 
      \beta_{n-1,t}\textcolor{blue}{v_{n-1}} + 
      \frac{\sigma_v}{2k}\textcolor{blue}{e^{-kt}I\!E_{n-1,t}} + 
   \left(\rho - \frac{\sigma_v }{2k}\right)\textcolor{blue}{I_{n-1,t}}
    + \sqrt{1-\rho^2} \textcolor{blue}{I_{n-1,t}^*}

where :math:`\bar{y}_{n-1,t} = y_{n-1,t} - E[y_{n-1,t}]` and 
:math:`\beta_{n-1,t} = (1-e^{-k[t-(n-1)h]})/(2k)`.

The *l*\ th central moment of :math:`y_{n-1,t}`, denoted by 
:math:`cm_l(y_{n-1,t})`, can be computed based on the following quantities:

.. math::
   :label: comb-moment
   
   E[\theta^{n_1}v_{n-1}^{n_2}(e^{-kt}I\!E_{n-1,t})^{n_3}I_{n-1,t}^{n_4}
   I_{n-1,t}^{*n_{5}}],

where :math:`n_i\geq 0` ( :math:`i=1,2,3,4,5` ) are integers and
:math:`\sum_{i=1}^{5}n_i=l`.
I can calculate quantity :eq:`comb-moment` via a two-step process:

1. Compute the conditional expectation by fixing :math:`v_{n-1}`:
  
  .. math::
     E[(e^{-kt}I\!E_{n-1,t})^{n_3}I_{n-1,t}^{n_4}
     I_{n-1,t}^{*n_{5}}|v_{n-1}].

2. Follow with the unconditional expectation with respect to :math:`v_{n−1}`:

  .. math::
     E[\theta^{n_1}v_{n-1}^{n_2}E[(e^{-kt}I\!E_{n-1,t})^{n_3}I_{n-1,t}^{n_4}
     I_{n-1,t}^{*n_{5}}|v_{n-1}]].

It will be shown later that the conditional 
moment :math:`E[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_{5}}|v_{n-1}]` is 
a polynomial of :math:`v_{n-1}`, which implies that quantity 
:eq:`comb-moment` can be expressed as a function of moments of 
:math:`v_{n-1}`.  
By using equation :math:numref:`moment-v`, I can compute :math:`v_{n-1}`'s 
moment of any order, further I can compute that of :eq:`comb-moment` as well. 

First, I consider 
:math:`E[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_{5}}|v_{n-1}]`. 
I separate :math:`I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_{5}}` into two 
parts: :math:`I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}` and :math:`I_{n-1,t}^{*n_{5}}`, 
since they are driven by two different Wiener processes :math:`w^v(t)` and 
:math:`w(t)`, respectively. For :math:`I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}`, 
I have

.. math::
   
   d(I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}) = c_w(t) dw^v(t)+ c(t) dt

where

.. math::
   
   c_w(t) 
   &\triangleq n_3 I\!E_{n-1,t}^{n_3-1}I_{n-1,t}^{n_4}\sqrt{v(t)} + 
   n_4 I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4-1}e^{kt}\sqrt{v(t)},\\
   c(t)
   &\triangleq \bigg[\frac{1}{2}n_3(n_3-1)I\!E_{n-1,t}^{n_3-2}
   I_{n-1,t}^{n_4}e^{2kt} + \frac{1}{2}n_4(n_4-1)I\!E_{n-1,t}^{n_3}
   I_{n-1,t}^{n_4-2}\\
   &\qquad + n_3n_4I\!E_{n-1,t}^{n_3-1}I_{n-1,t}^{n_4-1}e^{kt} \bigg] v(t).

Therefore, the conditional expectation

.. math::
   
   E[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}|v_{n-1}] = \int_{(n-1)h}^t 
   E[c(s)|v_{n-1}]ds.


.. _ito-recursive-i:

Itô process Moments - I
------------------------

If :math:`v(t)` is expanded as

.. math::
   
   v(t) = e^{-k[t-(n-1)h]}v_{n-1} + (1-e^{-k[t-(n-1)h]})\theta + 
   \sigma_v e^{-kt}I\!E_{n-1,t},

then, :math:`E[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}|v_{n-1}]` can be expressed as

.. math::
   :label: ito-moment-i
   
   &E[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}|v_{n-1}]\\
   &= \frac{n_3(n_3-1)}{2}e^{k(n-1)h}(v_{n-1} - \theta) 
   &\color{blue}\int_{(n-1)h}^t e^{ks} E[I\!E_{n-1,s}^{n_3-2}I_{n-1,s}^{n_4}|v_{n-1}]ds\\
   &\quad + \frac{n_3(n_3-1)}{2} \theta 
   &\color{blue}\int_{(n-1)h}^t e^{2ks} E[I\!E_{n-1,s}^{n_3-2}I_{n-1,s}^{n_4}|v_{n-1}]ds\\
   &\quad + \frac{n_3(n_3-1)}{2} \sigma_v 
   &\color{blue}\int_{(n-1)h}^t e^{ks} E[I\!E_{n-1,s}^{n_3-1}I_{n-1,s}^{n_4}|v_{n-1}]ds\\
   &\quad + \frac{n_4(n_4-1)}{2}e^{k(n-1)h}(v_{n-1} - \theta) 
   &\int_{(n-1)h}^t e^{-ks} E[I\!E_{n-1,s}^{n_3}I_{n-1,s}^{n_4-2}|v_{n-1}]ds\\
   &\quad + \frac{n_4(n_4-1)}{2} \theta 
   &\int_{(n-1)h}^t E[I\!E_{n-1,s}^{n_3}I_{n-1,s}^{n_4-2}|v_{n-1}]ds\\
   &\quad + \frac{n_4(n_4-1)}{2} \sigma_v 
   &\int_{(n-1)h}^t e^{-ks} E[I\!E_{n-1,s}^{n_3+1}I_{n-1,s}^{n_4-2}|v_{n-1}]ds\\
   &\quad + n_3n_4e^{k(n-1)h}(v_{n-1}- \theta)  
   &\color{blue}\int_{(n-1)h}^t E[I\!E_{n-1,s}^{n_3-1}I_{n-1,s}^{n_4-1}|v_{n-1}]ds\\
   &\quad + n_3n_4\theta 
   &\color{blue}\int_{(n-1)h}^t e^{ks}E[I\!E_{n-1,s}^{n_3-1}I_{n-1,s}^{n_4-1}|v_{n-1}]ds\\
   &\quad + n_3n_4\sigma_v 
   &\color{blue}\int_{(n-1)h}^t E[I\!E_{n-1,s}^{n_3}I_{n-1,s}^{n_4-1}|v_{n-1}]ds.

Moments of Low Orders 
^^^^^^^^^^^^^^^^^^^^^^

Order 1, i.e., :math:`n_3 + n_4 = 1`.

* :math:`(n_3,n_4) = (1,0): E[I\!E_{n-1,t}|v_{n-1}] = 0`
* :math:`(n_3,n_4) = (0,1): E[I_{n-1,t}|v_{n-1}] = 0`

Order 2, i.e., :math:`n_3 + n_4 = 2`.

+-------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
| :math:`(n_3,n_4)` | Moment :math:`E[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}|v_{n-1}]`                                                                                         |
+===================+=====================================================================================================================================================+
|(2,0)              | :math:`e^{2kt}\frac{1}{2k}\theta + e^{kt+k(n-1)h}\frac{1}{k}(v_{n-1}-\theta) - e^{2k(n-1)h} \left( \frac{1}{k}v_{n-1} - \frac{1}{2k}\theta \right)` |
+-------------------+----------------------------------------------------------+------------------------------------------------------------------------------------------+
|(1,1)              | :math:`e^{kt}\frac{1}{k}\theta + [t-(n-1)h]e^{k(n-1)h}(v_{n-1}-\theta) - e^{k(n-1)h}\frac{1}{k}\theta`                                              |
+-------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
|(0,2)              | :math:`- e^{-kt+k(n-1)h}\frac{1}{k}(v_{n-1}-\theta) + [t-(n-1)h]\theta + (v_{n-1}-\theta)\frac{1}{k}`                                               |
+-------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+

.. _ito-recursive-ii:

Itô process Moments - II
-------------------------

For :math:`I_{n-1,t}^{*n_5}`, its derivative

.. math::
   
   dI_{n-1,t}^{*n_5}
   = n_5I_{n-1,t}^{*n_5-1}\sqrt{v(t)} dw(t) + \frac{1}{2}n_5(n_5-1)
   I_{n-1,t}^{*n_5-2}v(t)dt.

Note that :math:`d(I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4})dI_{n-1,t}^{*n_5} = 0` 
because :math:`dw^v(t)dw(t) = 0`.
Hence, 

.. math::
   
   & d(I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}) \\
   &= (I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4})dI_{n-1,t}^{*n_5} + I_{n-1,t}^{*n_5}
   d(I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4})\\
   &= n_5I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5-1}\sqrt{v(t)} 
   dw(t) + c_w(t)I_{n-1,t}^{*n_5}dw^v(t)\\
   &\quad + \left[\frac{1}{2}n_5(n_5-1) I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}
   I_{n-1,t}^{*n_5-2}v(t)+ c(t)I_{n-1,t}^{*n_5}\right]dt.

Therefore,

.. math::
   
   &E[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]\\
   & = \int_{(n-1)h}^tE\left[\frac{1}{2}n_5(n_5-1) I\!E_{n-1,s}^{n_3}
   I_{n-1,s}^{n_4}I_{n-1,s}^{*n_5-2}v(s)+ c(s)I_{n-1,s}^{*n_5}|v_{n-1}\right]ds

where quantities having :math:`dw(t)` and :math:`dw^v(t)` have been deleted
because their expectations are 0.


Hence, :math:`E[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]` can 
be expressed as

.. math::
   :label: ito-moment-ii

   &E[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*n_5}|v_{n-1}]\\
   &= \frac{n_3(n_3-1)}{2}e^{k(n-1)h}(v_{n-1}-\theta) 
   &\color{blue}\int_{(n-1)h}^t e^{ks} E[I\!E_{n-1,s}^{n_3-2}I_{n-1,s}^{n_4}I_{n-1,s}^{*n_5}|v_{n-1}]ds\\
   &\quad + \frac{n_3(n_3-1)}{2} \theta 
   &\color{blue}\int_{(n-1)h}^t e^{2ks} E[I\!E_{n-1,s}^{n_3-2}I_{n-1,s}^{n_4}I_{n-1,s}^{*n_5}|v_{n-1}]ds\\
   &\quad + \frac{n_3(n_3-1)}{2} \sigma_v 
   &\color{blue}\int_{(n-1)h}^t e^{ks} E[I\!E_{n-1,s}^{n_3-1}I_{n-1,s}^{n_4}I_{n-1,s}^{*n_5}|v_{n-1}]ds\\
   &\quad + \frac{n_4(n_4-1)}{2}e^{k(n-1)h}(v_{n-1}-\theta) 
   &\int_{(n-1)h}^t e^{-ks} E[I\!E_{n-1,s}^{n_3}I_{n-1,s}^{n_4-2}I_{n-1,s}^{*n_5}|v_{n-1}]ds\\
   &\quad + \frac{n_4(n_4-1)}{2} \theta 
   &\int_{(n-1)h}^t E[I\!E_{n-1,s}^{n_3}I_{n-1,s}^{n_4-2}I_{n-1,s}^{*n_5}|v_{n-1}]ds\\
   &\quad + \frac{n_4(n_4-1)}{2} \sigma_v 
   &\int_{(n-1)h}^t e^{-ks} E[I\!E_{n-1,s}^{n_3+1}I_{n-1,s}^{n_4-2}I_{n-1,s}^{*n_5}|v_{n-1}]ds\\
   &\quad + n_3n_4e^{k(n-1)h}(v_{n-1}-\theta) 
   &\color{blue}\int_{(n-1)h}^t  E[I\!E_{n-1,s}^{n_3-1}I_{n-1,s}^{n_4-1}I_{n-1,s}^{*n_5}|v_{n-1}]ds\\
   &\quad + n_3n_4\theta 
   &\color{blue}\int_{(n-1)h}^t  e^{ks}E[I\!E_{n-1,s}^{n_3-1}I_{n-1,s}^{n_4-1}I_{n-1,t}^{*n_5}|v_{n-1}]ds\\
   &\quad + n_3n_4\sigma_v 
   &\color{blue}\int_{(n-1)h}^t E[I\!E_{n-1,s}^{n_3}I_{n-1,s}^{n_4-1}I_{n-1,s}^{*n_5}|v_{n-1}]ds\\
   &\quad + \frac{n_5(n_5-1)}{2}e^{k(n-1)h}(v_{n-1}-\theta) 
   &\int_{(n-1)h}^t e^{-ks} E[I\!E_{n-1,s}^{n_3}I_{n-1,s}^{n_4}I_{n-1,s}^{*n_5-2}|v_{n-1}]ds\\
   &\quad + \frac{n_5(n_5-1)}{2}\theta 
   &\int_{(n-1)h}^t E[I\!E_{n-1,s}^{n_3}I_{n-1,s}^{n_4}I_{n-1,s}^{*n_5-2}|v_{n-1}]ds\\
   &\quad + \frac{n_5(n_5-1)}{2}\sigma_v 
   &\int_{(n-1)h}^t e^{-ks} E[I\!E_{n-1,s}^{n_3+1}I_{n-1,s}^{n_4}I_{n-1,s}^{*n_5-2}|v_{n-1}]ds.


It should be noted that 
:math:`E[I_{n-1,t}^{*n_5}|v_{n-1}] = E[I_{n-1,t}^{n_5}|v_{n-1}]`.


Low Order Moments
^^^^^^^^^^^^^^^^^^

Order 1, i.e., :math:`n_3 + n_4 + n_5= 1`.

* :math:`(n_3,n_4,n_5) = (1,0,0): E[I\!E_{n-1,t}|v_{n-1}] = 0`.
* :math:`(n_3,n_4,n_5) = (0,1,0): E[I_{n-1,t}|v_{n-1}] = 0`.
* :math:`(n_3,n_4,n_5) = (0,0,1): E[I_{n-1,t}^{*}|v_{n-1}] = 0`.

Order 2, i.e., :math:`n_3 + n_4 + n_5= 2`.

* :math:`(n_3,n_4,n_5=0)` reduces to :math:`(n_3,n_4)`, 
  i.e., :math:`E[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}|v_{n-1}]`.
* :math:`(n_3,n_4,n_5=1)`: 
  :math:`E[I\!E_{n-1,t}^{n_3}I_{n-1,t}^{n_4}I_{n-1,t}^{*}|v_{n-1}] = 0`. 
* :math:`(n_3,n_4,n_5)=(0,0,2)` reduces to :math:`(n_3, n_4)=(0,2)`,
  i.e., :math:`E[I_{n-1,t}^{2}|v_{n-1}]`.

.. note:: The two recursive equations :math:numref:`ito-moment-i` and 
   :eq:`ito-moment-ii` can be used to compute the central moment of any order 
   of :math:`y_{n-1,t}` recursively, from lower order ones to high
   order ones. For example, we can start with the combinations 
   :math:`\{(n_3,n_4,n_5), l=1\}`, then 
   :math:`\{(n_3,n_4,n_5), l=2\}`, so on and so forth, where :math:`n_3+n_4+n_5=l`. 
   The computations are fairly straightforward but computationally intensive, 
   which can be automated as implemented in the ``ajdmom`` package 
   which is explained in the :doc:`design` page. 

Covariance Derivation
----------------------

Similarly, we can compute

.. math::
   
   cov(y_n^{l_1}, y_{n+1}^{l_2})
   = E[y_n^{l_1}y_{n+1}^{l_2}] - E[y_n^{l_1}]E[y_{n+1}^{l_2}]

in which :math:`y_n = y_{n-1,t}` with :math:`t=nh` where

.. math::
   
   y_{n-1,t}
   &= (\mu -\theta/2)[t-(n-1)h] + \theta\beta_{n-1,t} - v_{n-1}\beta_{n-1,t}\\
   &\quad + \frac{\sigma_v}{2k}e^{-kt}I\!E_{n-1,t} + 
   \left(\rho - \frac{\sigma_v}{2k}\right)I_{n-1,t} + \sqrt{1-\rho^2}I_{n-1,t}^{*}

which also equals to :math:`\overline{y}_{n-1,t} + (\mu -\theta/2)[t-(n-1)h]`.


-------------

.. [#f1] Duffie, D., Pan, J., & Singleton, K. (2000). Transform analysis and asset pricing for affine jump‐diffusions. Econometrica, 68(6), 1343-1376.
.. [#f2] Whose exact equations vary according to different authors. One 
  alternative setting for :math:`ds(t)` is :math:`dp(t) = \mu dt + \sqrt{v(t)}dw^s(t)` 
  where :math:`p(t) = \log s(t)` which is simpler than the settings in 
  our package. The procedures in the ``ajdmom`` package can be easily adapted to this
  simpler settings.
.. [#f3] Cox, J. C., Ingersoll Jr, J. E., and Ross, S. A. (1985). A theory of the term structure of interest rates. Econometrica, 53:385–407.
.. [#f4] Overbeck, L. and Rydén, T. (1997). Estimation in the cox-ingersoll-ross model. Econometric Theory, 13(3):430–461
.. [#f5] Though modeled as a continuous-time process, the asset price is 
  observed at discrete-time instances. Assume we have observations of 
  :math:`s(t)` at discrete-time :math:`ih` (:math:`i=0,1,\cdots,N`). Similarly, 
  let :math:`v_i \triangleq v(ih)`, however, it should be noted that 
  :math:`v_i` is not observable.
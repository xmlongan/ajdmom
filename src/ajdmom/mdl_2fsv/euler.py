'''
Module for generating samples from mdl_2fsv by Euler approximation

To facilitate the verification of the correctness of our codes 
by comparison the population moments (given by our package) and
their sample counterparts. 
Here I define function :py:func:`~ajdmom.mdl_2fsv.euler.r2FSV`.
'''
import math
import numpy as np

def r2FSV(v_0, par, N, n_segment = 10, h = 1):
  '''Generate samples from mdl_2fsv by Euler approximation
  
  :param v_0: values of the initial variances :math:`(v_1(0),v_2(0))`.
  :param par: parameters in a dict.
  :param N: target length of samples to generate.
  :param n_segment: number of segments each interval is splitted into.
  :param h: time interval between each two consecutive samples.
  
  :return: a sequence of samples.
  :rtype: numpy 1-D array.
  '''
  mu = par['mu']
  k1 = par['k1']; theta1 = par['theta1']; sigma1 = par['sigma_v1']
  k2 = par['k2']; theta2 = par['theta2']; sigma2 = par['sigma_v2']
  # 
  dlt = h/n_segment; std = math.sqrt(dlt)
  # 
  y_series = np.empty(N)
  # 
  v1_0 = v_0[0]
  v2_0 = v_0[1]
  # 
  for i in range(N):
    I = 0; IV1 = 0; IV2 = 0
    for j in range(n_segment):
      eps  = math.sqrt(v1_0+v2_0) * np.random.normal(0, std)
      eps1 = math.sqrt(v1_0) * np.random.normal(0, std)
      eps2 = math.sqrt(v2_0) * np.random.normal(0, std)
      v1   = v1_0 + k1*(theta1-v1_0)*dlt + sigma1*eps1
      v2   = v2_0 + k2*(theta2-v2_0)*dlt + sigma2*eps2
      # 
      if v1 < 0:
        v1 = 0   # eps1 = -(v1_0 + k1*(theta1-v1_0)*dlt)/sigma1
      if v2 < 0:
        v2 = 0   # eps2 = -(v2_0 + k2*(theta2-v2_0)*dlt)/sigma2
      # 
      IV1 += v1_0*dlt
      IV2 += v2_0*dlt
      I += eps
      v1_0 = v1
      v2_0 = v2
    y = mu*h - (IV1+IV2)/2 + I
    y_series[i] = y
  return(y_series)

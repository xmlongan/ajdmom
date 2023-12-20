'''
Module for generating samples from Heston SV model by Euler approximation

To facilitate the verification of the correctness of our codes 
by comparison the population moments (given by our package) and
their sample counterparts. 
Here I define function :py:func:`~ajdmom.mdl_1fsv.euler.r1FSV`.
'''
import math
import numpy as np

def r1FSV(v_0, par, N, n_segment = 10, h = 1):
  '''Generate samples from mdl_1fsv by Euler approximation
  
  :param v_0: value of the initial variance :math:`v(0)`.
  :param par: parameters in a dict.
  :param N: target length of samples to generate.
  :param n_segment: number of segments each interval is splitted into.
  :param h: time interval between each two consecutive samples.
  
  :return: a sequence of samples.
  :rtype: numpy 1-D array.
  '''
  mu = par['mu']; k = par['k']; theta = par['theta']
  sigma = par['sigma_v']; rho = par['rho']
  # 
  dlt = h/n_segment; std = math.sqrt(dlt)
  # 
  y_series = np.empty(N)
  # 
  for i in range(N):
    I = 0; I2 = 0; IV = 0
    for j in range(n_segment):
      eps  = math.sqrt(v_0) * np.random.normal(0, std)
      eps2 = math.sqrt(v_0) * np.random.normal(0, std)
      v    = v_0 + k*(theta-v_0)*dlt + sigma*eps
      if v < 0:
        v = 0   # eps = -(v_0 + k*(theta-v_0)*dlt)/sigma
      I += eps; I2 += eps2; IV += v_0*dlt
      v_0 = v
    y = mu*h - IV/2 + rho*I + math.sqrt(1-rho**2)*I2
    y_series[i] = y
  # 
  return(y_series)

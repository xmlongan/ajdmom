'''
Module for Moments for Two-Factor SV with jumps
'''
import math
from fractions import Fraction as Frac

from ajdmom.poly import Poly
from ajdmom.cpp_mom import mcpp
from ajdmom.mdl_2fsv.mom import moment_y as m_y
from ajdmom.mdl_2fsv.cmom import dt0

def moment_y(l):
  '''Moment of :math:`y_n` with order :math:`l`

  :param l: order of the moment.

  :return: poly with attribute ``keyfor`` =
     ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
     'e^{-(n1*k1+n2*k2)h}','h','mu',
     'theta1','sigma_v1','theta2','sigma_v2', 'lambda','mu_j','sigma_j').
  :rtype: Poly
  '''
  poly = Poly()
  kf = ['(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
    'e^{-(n1*k1+n2*k2)h}','h','mu',
    'theta1','sigma_v1','theta2','sigma_v2', 'lambda','mu_j','sigma_j']
  poly.set_keyfor(kf)
  #
  for i in range(l, -1, -1):
    pol1 = m_y(i)
    # keyfor = ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
    #   'e^{-(n1*k1+n2*k2)h}','h','mu',
    #   'theta1','sigma_v1', 'theta2','sigma_v2')
    pol2 = mcpp(l-i)
    # keyfor = ('lambda*h','mu_j','sigma_j')
    c = math.comb(l,i)
    for k1 in pol1:
      for k2 in pol2:
        key = (k1[0], k1[1], k1[2]+k2[0], k1[3], k1[4],k1[5], k1[6],k1[7],
          k2[0], k2[1], k2[2])
        val = c * pol1[k1] * pol2[k2]
        poly.add_keyval(key, val)
  return(poly.remove_zero())

##########
# scalar and (partial) derivative
##########

def dpoly(poly, wrt):
  '''Partial derivative of moment w.r.t. parameter wrt

  :param poly: poly with attribute ``keyfor`` =
     ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
     'e^{-(n1*k1+n2*k2)h}','h','mu',
     'theta1','sigma_v1','theta2','sigma_v2','lambda','mu_j','sigma_j').
  :param wrt: with respect to.

  :return: poly with attribute ``keyfor`` =
     ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
     'e^{-(n1*k1+n2*k2)h}','h','mu',
     'theta1','sigma_v1','theta2','sigma_v2','lambda','mu_j','sigma_j').
  :rtype: Poly
  '''
  pold = Poly()
  kf = ['(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
    'e^{-(n1*k1+n2*k2)h}','h','mu',
    'theta1','sigma_v1','theta2','sigma_v2','lambda','mu_j','sigma_j']
  pold.set_keyfor(kf)
  #
  # partial derivative w.r.t. k1 or k2
  if wrt == 'k1' or wrt == 'k2':
    for k in poly:
      t0 = k[0]
      if len(t0) != 0:
        deriv = dt0(t0, wrt)
        if len(deriv) != 0:
          for t in deriv:
            t0_new, val_new = t
            knw = list(k); knw[0] = t0_new
            val = val_new * poly[k]
            pold.add_keyval(tuple(knw), val)
    for k in poly:
      if wrt == 'k1':
        n = k[1][0]
      else:
        n = k[1][1]
      if n != 0:
        knw = list(k); knw[2] += 1
        val = (-n) * poly[k]
        pold.add_keyval(tuple(knw), val)
  # partial derivative w.r.t. mu
  elif wrt in ['mu','theta1','sigma_v1','theta2','sigma_v2','lambda','mu_j',
    'sigma_j']:
    if wrt == 'mu': i = 3
    if wrt == 'theta1': i = 4
    if wrt == 'sigma_v1': i = 5
    if wrt == 'theta2': i = 6
    if wrt == 'sigma_v2': i = 7
    if wrt == 'lambda': i = 8
    if wrt == 'mu_j': i = 9
    if wrt == 'sigma_j': i = 10
    for k in poly:
      if k[i] != 0:
        knw = list(k); knw[i] -= 1
        val = k[i] * poly[k]
        pold.add_keyval(tuple(knw), val)
  else:
    candidates = "'k1','k2','mu','theta1','sigma_v1','theta2','sigma_v2'"
    candidates += ",'lambda','mu_j','sigma_j'"
    raise ValueError(f"wrt must be one of {candidates}!")
  return(pold)

def poly2num(poly, par):
  '''Decode poly back to scalar

  :param poly: poly to be decoded with attribute ``keyfor`` =
     ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
     'e^{-(n1*k1+n2*k2)h}','h','mu',
     'theta1','sigma_v1','theta2','sigma_v2','lambda','mu_j','sigma_j').
  :param par: parameters in dict.

  :return: scalar of the poly.
  :rtype: float
  '''
  k1 = par['k1']
  k2 = par['k2']
  h = par['h']
  mu = par['mu']
  theta1 = par['theta1']
  theta2 = par['theta2']
  sigma_v1 = par['sigma_v1']
  sigma_v2 = par['sigma_v2']
  lmbd = par['lambda']
  mu_j = par['mu_j']
  sigma_j = par['sigma_j']
  #
  value = 0
  for k in poly:
    val = poly[k]
    t0 = k[0]
    for t in t0:
      val *= (t[0]*k1 + t[1]*k2) ** (-t[2])
    t1 = k[1]
    val *= math.exp(-(t1[0]*k1 + t1[1]*k2)*h)
    val *= h ** k[2]
    val *= mu ** k[3]
    val *= theta1 ** k[4]
    val *= sigma_v1 ** k[5]
    val *= theta2 ** k[6]
    val *= sigma_v2 ** k[7]
    val *= lmbd ** k[8]
    val *= mu_j ** k[9]
    val *= sigma_j ** k[10]
    value += val
  return(value)

def m(l, par):
  '''Moment in scalar

  :param l: order of the moment.
  :param par: parameters in dict.

  :return: scalar of the moment.
  :rtype: float
  '''
  moment = moment_y(l)
  value = poly2num(moment, par)
  return(value)

def dm(l, par, wrt):
  '''Partial derivative of moment w.r.t. parameter wrt

  :param l: order of the moment.
  :param par: parameters in dict.
  :param wrt: with respect to.

  :return: scalar of the partial derivative.
  :rtype: float
  '''
  moment = moment_y(l)
  pold = dpoly(moment, wrt)
  value = poly2num(pold, par)
  return(value)


if __name__ == "__main__":
  # Example usage of the module, see 'tests/test_mdl_2fsvj.py' for more test
  from pprint import pprint
  print('\nExample usage of the module function\n')
  #
  kf = ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
     'e^{-(n1*k1+n2*k2)h}','h','mu',
     'theta1','sigma_v1','theta2','sigma_v2', 'lambda','mu_j','sigma_j')
  print(f"moment_y(l) returns a poly with keyfor = \n{kf}")
  print("moment_y(2) = "); pprint(moment_y(2))

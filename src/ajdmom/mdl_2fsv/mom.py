'''
Moments for Two-Factor SV
'''
import math
from fractions import Fraction as Frac

from ajdmom.poly import Poly
from ajdmom.ito_mom import moment_v
from ajdmom.itos_mom import moment_IEI_IEII, t_mul_t0
from ajdmom.mdl_2fsv.cmom import (
  c_n, 
  vvee_IEI_IEII,
  dt0
)

def b_n(n1, n2, n3, n4, n5, n6, n7):
  '''Coefficient corresponding to the combination
  
  :param n1: times of constant being selected.
  :param n2: times of :math:`v_{1,n-1}` being selected.
  :param n3: times of :math:`v_{2,n-1}` being selected.
  :param n4: times of :math:`eI_{1,n}` being selected.
  :param n5: times of :math:`I_{1,n}` being selected.
  :param n6: times of :math:`eI_{2,n}` being selected.
  :param n7: times of :math:`I_{2,n}` being selected.
  
  :return: poly with attribute ``keyfor`` =
     ('e^{-k1h}','k1^{-}','theta1','sigma_v1',
     'e^{-k2h}','k2^{-}','theta2','sigma_v2', 'h','mu').
  :rtype: Poly
  '''
  b = Poly()
  kf = ['e^{-k1h}','k1^{-}','theta1','sigma_v1',
    'e^{-k2h}','k2^{-}','theta2','sigma_v2', 'h','mu']
  b.set_keyfor(kf)
  # 
  for i1 in range(n1, -1, -1):                          # 1/2k1 * theta1
    for i2 in range(n1-i1, -1, -1):                     # -e^{-k1h}/2k1 *theta1
      for i3 in range(n1-i1-i2, -1, -1):                # 1/2k2 * theta2
        for i4 in range(n1-i1-i2-i3, -1, -1):           # -e^{-k2h}/2k2 *theta2
          for i5 in range(n1-i1-i2-i3-i4, -1, -1):      # -theta1 h/2
            for i6 in range(n1-i1-i2-i3-i4-i5, -1, -1): # -theta2 h/2
              i7 = n1-i1-i2-i3-i4-i5-i6                 # mu h
              for j in range(n2+1):                     # -e^{-k1h}/2k1
                for q in range(n3+1):                   # -e^{-k2h}/2k2
                  num  = math.comb(n1,i1) 
                  num *= math.comb(n1-i1,i2) 
                  num *= math.comb(n1-i1-i2,i3)
                  num *= math.comb(n1-i1-i2-i3,i4)
                  num *= math.comb(n1-i1-i2-i3-i4,i5)
                  num *= math.comb(n1-i1-i2-i3-i4-i5,i6)
                  num *= math.comb(n2,j)
                  num *= math.comb(n3,q)
                  # 
                  key = (i2+j, i1+i2+n2+n4+n5, i1+i2 + i5, n4+n5,
                         i4+q, i3+i4+n3+n6+n7, i3+i4 + i6, n6+n7, 
                         i5+i6+i7, i7)
                  num *= (-1)**(i2+i4+i5+i6+(n2+j)+(n3+q)+n5+n7)
                  den = 2**(n1-i7+n2+n3+n4+n5+n6+n7)
                  b.add_keyval(key, Frac(num, den))
  return(b)

def moment_comb(n, n1, n2, n3, n4, n5, n6, n7, n8):
  '''Moment for this combination of given n1 to n8 with n.
  
  :param n: l in :math:`E[y_{n}^l]`.
  :param n1: times of constant being selected.
  :param n2: times of :math:`v_{1,n-1}` being selected.
  :param n3: times of :math:`v_{2,n-1}` being selected.
  :param n4: times of :math:`eI_{1,n}` being selected.
  :param n5: times of :math:`I_{1,n}` being selected.
  :param n6: times of :math:`eI_{2,n}` being selected.
  :param n7: times of :math:`I_{2,n}` being selected.
  :param n8: times of :math:`I_{n}^{*}` being selected.
  
  :return: poly with attribute ``keyfor`` = 
     ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
     'e^{-(n1*k1+n2*k2)h}','h','mu',
     'v_{1,n-1}','theta1','sigma_v1', 'v_{2,n-1}','theta2','sigma_v2').
  :rtype: Poly
  '''
  poly = vvee_IEI_IEII(n2, n3, n4, n5, n6, n7, n8)
  # ['(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
  #  'e^{-(n1*k1+n2*k2)h}','h',
  #  'v_{1,n-1}','theta1','sigma_v1', 'v_{2,n-1}','theta2','sigma_v2']
  b = b_n(n1, n2, n3, n4, n5, n6, n7)
  # ['e^{-k1h}','k1^{-}','theta1','sigma_v1',
  #  'e^{-k2h}','k2^{-}','theta2','sigma_v2','h','mu']
  kf = ['(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
    'e^{-(n1*k1+n2*k2)h}','h','mu',
    'v_{1,n-1}','theta1','sigma_v1', 'v_{2,n-1}','theta2','sigma_v2']
  #
  poln = Poly()
  poln.set_keyfor(kf)
  # 
  for k1 in poly:
    for k2 in b:
      # t0 = list(k1[0]); t0.insert(0,(1,0,k2[1])); t0.insert(0,(0,1,k2[5]))
      # t0 = tuple(t0)
      t0 = t_mul_t0((1,0,k2[1]), k1[0])
      t0 = t_mul_t0((0,1,k2[5]), t0)
      # 
      t1 = (k1[1][0]+k2[0], k1[1][1]+k2[4])
      #                   h        mu  v_{1,n-1}  theta1       sigma_v1
      key = (t0, t1, k1[2]+k2[8], k2[9], k1[3], k1[4]+k2[2], k1[5]+k2[3],
        # v_{2,n-1} theta2     sigma_v2
        k1[6], k1[7]+k2[6], k1[8]+k2[7])
      val = poly[k1] * b[k2]
      poln.add_keyval(key, val)
  c = c_n(n, n1,n2,n3,n4,n5,n6,n7,n8)
  return(c * poln)

def sub_v(poly):
  '''Substitute :math:`v_{1,n-1}` and :math:`v_{2,n-1}` with their moments
  
  :param poly: poly with attribute ``keyfor`` = 
     ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
     'e^{-(n1*k1+n2*k2)h}','h','mu',
     'v_{1,n-1}','theta1','sigma_v1', 'v_{2,n-1}','theta2','sigma_v2').
  
  :return: poly with attribute ``keyfor`` = 
     ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
     'e^{-(n1*k1+n2*k2)h}','h','mu', 'theta1','sigma_v1','theta2','sigma_v2').
  :rtype: Poly
  '''
  poly_sum = Poly()
  kf = ['(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
    'e^{-(n1*k1+n2*k2)h}','h','mu', 'theta1','sigma_v1','theta2','sigma_v2']
  poly_sum.set_keyfor(kf)
  # 
  poln = Poly()
  for k in poly:
    pol1 = moment_v(k[4]) # [theta1, sigma_v1^2/k1]
    for key in pol1:
      i, j = key
      # t0 = list(k[0]); t0.insert(0, (1,0,j)); t0 = tuple(t0)
      t0 = t_mul_t0((1,0,j), k[0])
      # 
      knw = (t0, k[1], k[2], k[3], k[5]+i, k[6]+2*j, k[7], k[8], k[9])
      val = poly[k] * pol1[key]
      poln.add_keyval(knw, val)
  for k in poln:
    pol2 = moment_v(k[6]) # [theta2, sigma_v2^2/k2]
    for key in pol2:
      i, j = key
      # t0 = list(k[0]); t0.insert(0, (0,1,j)); t0 = tuple(t0)
      t0 = t_mul_t0((0,1,j), k[0])
      # 
      knw = (t0, k[1], k[2], k[3], k[4], k[5], k[7]+i, k[8]+2*j)
      val = poln[k] * pol2[key]
      poly_sum.add_keyval(knw, val)
  return(poly_sum)

def moment_y(l):
  '''Moment of :math:`y_n` with order :math:`l`
  
  :param l: order of the moment.
  
  :return: poly with attribute ``keyfor`` = 
     ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
     'e^{-(n1*k1+n2*k2)h}','h','mu', 'theta1','sigma_v1','theta2','sigma_v2').
  :rtype: Poly
  '''
  poly = Poly()
  kf = ['(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
    'e^{-(n1*k1+n2*k2)h}','h','mu',
    'v_{1,n-1}','theta1','sigma_v1', 'v_{2,n-1}','theta2','sigma_v2']
  poly.set_keyfor(kf)
  # 
  for n1 in range(l, -1, -1):
    for n2 in range(l-n1, -1, -1):
      for n3 in range(l-n1-n2, -1, -1):
        for n4 in range(l-n1-n2-n3, -1, -1):
          for n5 in range(l-n1-n2-n3-n4, -1, -1):
            for n6 in range(l-n1-n2-n3-n4-n5, -1, -1):
              for n7 in range(l-n1-n2-n3-n4-n5-n6, -1, -1):
                n8 = l-n1-n2-n3-n4-n5-n6-n7
                poly.merge(moment_comb(l, n1,n2,n3,n4,n5,n6,n7,n8))
  poly_sum = sub_v(poly)
  poly_sum.remove_zero()
  return(poly_sum)

##########
# scalar and (partial) derivative
##########

def dpoly(poly, wrt):
  '''Partial derivative of moment w.r.t. parameter wrt
  
  :param poly: poly with attribute ``keyfor`` = 
     ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
     'e^{-(n1*k1+n2*k2)h}','h','mu', 'theta1','sigma_v1','theta2','sigma_v2').
  :param wrt: with respect to.
  
  :return: poly with attribute ``keyfor`` = 
     ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
     'e^{-(n1*k1+n2*k2)h}','h','mu', 'theta1','sigma_v1','theta2','sigma_v2').
  :rtype: Poly
  '''
  pold = Poly()
  kf = ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
    'e^{-(n1*k1+n2*k2)h}','h','mu', 'theta1','sigma_v1','theta2','sigma_v2')
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
  elif wrt in ['mu','theta1','sigma_v1','theta2','sigma_v2']:
    if wrt == 'mu': i = 3
    if wrt == 'theta1': i = 4
    if wrt == 'sigma_v1': i = 5
    if wrt == 'theta2': i = 6
    if wrt == 'sigma_v2': i = 7
    for k in poly:
      if k[i] != 0:
        knw = list(k); knw[i] -= 1
        val = k[i] * poly[k]
        pold.add_keyval(tuple(knw), val)
  else:
    candidates = "'k1','k2','mu','theta1','sigma_v1','theta2','sigma_v2'"
    raise ValueError(f"wrt must be one of {candidates}!")
  return(pold)

def poly2num(poly, par):
  '''Decode poly back to scalar
  
  :param poly: poly to be decoded with attribute ``keyfor`` = 
     ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
     'e^{-(n1*k1+n2*k2)h}','h','mu', 'theta1','sigma_v1','theta2','sigma_v2').
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
  # Example usage of the module, see 'tests/test_mdl_2fsv.py' for more test
  from pprint import pprint
  print('\nExample usage of the module function\n')
  # 
  kf = ('(n_1m*k1+n_2m*k2)^{-i_m},...,(n_11*k1+n_21*k2)^{-i_1}',
     'e^{-(n1*k1+n2*k2)h}','h','mu', 'theta1','sigma_v1','theta2','sigma_v2')
  print(f"moment_y(l) returns a poly with keyfor = \n{kf}")
  print("moment_y(2) = "); pprint(moment_y(2))

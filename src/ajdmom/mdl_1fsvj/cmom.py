'''
Central Moments for One-Factor SV with jumps
'''
import math

from ajdmom.poly import Poly
from ajdmom.mdl_1fsv.cmom import cmoment_y as cm_y
# ['e^{-kh}','h','k^{-}','theta','sigma_v','rho','sqrt(1-rho^2)']
from ajdmom.cpp_mom import cmcpp
# ['lambda*h','mu_j','sigma_j^2']

def cmoment_y(l):
  '''Central moment of :math:`y_n` of order :math:`l`

  :param l: order of the moment.

  :return: poly with attribute ``keyfor`` =
     ('e^{-kh}','h','k^{-}','theta','sigma_v','rho','sqrt(1-rho^2)','lambda',
     'mu_j','sigma_j').
  :rtype: Poly
  '''
  poly = Poly()
  kf = ['e^{-kh}','h','k^{-}','theta','sigma_v','rho','sqrt(1-rho^2)','lambda',
    'mu_j','sigma_j']
  poly.set_keyfor(kf)
  # if l == 0:
  #   poln = Poly({(0,0,0,0,0,0,0,0,0,0): 1}); poln.set_keyfor(kf); return(poln)
  #
  for i in range(l+1):
    coef = math.comb(l, i)
    pol1 = cm_y(i)
    # ('e^{-kh}','h','k^{-}','theta','sigma_v','rho','sqrt(1-rho^2)')
    pol2 = cmcpp(l-i)
    # ('lambda*h','mu','sigma')
    keyIndexes = [(0,1,2,3,4,5,6,-1,-1,-1),(-1,0,-1,-1,-1,-1,-1,0,1,2)]
    poln = pol1.mul_poly(pol2, keyIndexes, kf)
    poly.merge(coef * poln)
  return(poly)

##########
# scalar and (partial) derivative
##########

def dpoly(poly, wrt):
  '''Partial derivative of central moment w.r.t. parameter wrt

  :param poly: poly with attribute ``keyfor`` =
     ('e^{-kh}','h','k^{-}','theta','sigma_v','rho','sqrt(1-rho^2)','lambda',
     'mu_j','sigma_j')
  :param wrt: with respect to.

  :return: poly with attribute ``keyfor`` =
     ('e^{-kh}','h','k^{-}','theta','sigma_v','rho','sqrt(1-rho^2)','lambda',
     'mu_j','sigma_j').
  :rtype: Poly
  '''
  pold = Poly()
  kf = ('e^{-kh}','h','k^{-}','theta','sigma_v','rho','sqrt(1-rho^2)','lambda',
    'mu_j','sigma_j')
  pold.set_keyfor(kf)
  #
  # partial derivative w.r.t. mu
  if wrt == 'mu':
    return(pold)
  # partial derivative w.r.t. k
  elif wrt == 'k':
    for k in poly:
      if k[0] != 0:
        knw = list(k); knw[1] += 1; knw = tuple(knw)
        val = (-k[0]) * poly[k]
        pold.add_keyval(knw, val)
      if k[2] != 0:
        knw = list(k); knw[2] += 1; knw = tuple(knw)
        val = (-k[2]) * poly[k]
        pold.add_keyval(knw, val)
  # partial derivative w.r.t. theta or sigma_v
  elif wrt in ['theta','sigma_v','lambda','mu_j','sigma_j']:
    if wrt == 'theta': i = 3
    if wrt == 'sigma_v': i = 4
    if wrt == 'lambda': i = 7
    if wrt == 'mu_j': i = 8
    if wrt == 'sigma_j': i = 9
    for k in poly:
      if k[i] != 0:
        knw = list(k); knw[i] -= 1; knw = tuple(knw)
        val = k[i] * poly[k]
        pold.add_keyval(knw, val)
  # partial derivative w.r.t. rho
  elif wrt == 'rho':
    for k in poly:
      if k[5] != 0:
        knw = list(k); knw[5] -= 1; knw = tuple(knw)
        val = k[5] * poly[k]
        pold.add_keyval(knw, val)
      if k[6] != 0:
        knw = list(k); knw[5] += 1; knw[6] -=2; knw = tuple(knw)
        val = (-k[6]) * poly[k]
        pold.add_keyval(knw, val)
  else:
    candidates = "'k','theta','sigma_v','rho','lambda','mu_j','sigma_j'"
    raise ValueError(f"wrt must be one of {candidates}!")
  return(pold)

def poly2num(poly, par):
  '''Decode poly back to scalar

  :param poly: poly to be decoded with attribute ``keyfor`` =
     ('e^{-kh}','h','k^{-}','theta','sigma_v','rho','sqrt(1-rho^2)','lambda',
     'mu_j','sigma_j')
  :param par: parameters in dict.

  :return: scalar of the poly.
  :rtype: float
  '''
  k = par['k']
  h = par['h']
  theta = par['theta']
  sigma_v = par['sigma_v']
  rho = par['rho']
  lmbd = par['lambda']
  mu_j = par['mu_j']
  sigma_j = par['sigma_j']
  #
  value = 0
  for K in poly:
    val = poly[K] * math.exp(-K[0]*k*h) * (h ** K[1]) * (k ** (-K[2]))
    val *= (theta ** K[3]) * (sigma_v ** K[4]) * (rho ** K[5])
    val *= (1-rho**2) ** (K[6]/2)
    val *= (lmbd ** K[7]) * (mu_j ** K[8]) * (sigma_j ** K[9])
    value += val
  return(value)

def cm(l, par):
  '''Central moment in scalar

  :param l: order of the central moment.
  :param par: parameters in dict.

  :return: scalar of the central moment.
  :rtype: float
  '''
  cmoment = cmoment_y(l)
  value = poly2num(cmoment, par)
  return(value)

def dcm(l, par, wrt):
  '''Partial derivative of central moment w.r.t. parameter wrt

  :param l: order of the central moment.
  :param par: parameters in dict.
  :param wrt: with respect to.

  :return: scalar of the partial derivative.
  :rtype: float
  '''
  cmoment = cmoment_y(l)
  pold = dpoly(cmoment, wrt)
  value = poly2num(pold, par)
  return(value)


if __name__ == "__main__":
  # Example usage of the module, see 'tests/test_mdl_1fsvj.py' for more test
  from pprint import pprint
  print('\nExample usage of the module function\n')
  #
  kf = ('e^{-kh}','h','k^{-}','theta','sigma_v','rho','sqrt(1-rho^2)',
    'lambda','mu_j','sigma_j^2')
  print(f"cmoment_y(l) returns a poly with keyfor = \n{kf}")
  print("cmoment_y(3) = "); pprint(cmoment_y(3))

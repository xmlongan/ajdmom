'''
Central Moments for One-Factor SV
'''
# from pprint import pprint
import math
from fractions import Fraction as Frac

from ajdmom.poly import Poly
from ajdmom.ito_mom import moment_v, moment_IEII

def c_n(l, n1, n2, n3, n4, n5):
  ''':math:`c(\\boldsymbol{n})` in :eq:`c-n`
  
  :param l: l in :math:`E[\overline{y}_{n}^l]`.
  :param n1: times of :math:`\\theta` being selected.
  :param n2: times of :math:`v_{n-1}` being selected.
  :param n3: times of :math:`eI_n` being selected.
  :param n4: times of :math:`I_n` being selected.
  :param n5: times of :math:`I_n^{*}` being selected.
  
  :return: number of this special combination.
  :rtype: int
  '''
  num  = math.comb(l, n1)
  num *= math.comb(l-n1, n2)
  num *= math.comb(l-n1-n2, n3)
  num *= math.comb(l-n1-n2-n3, n4)
  return(num)

def b_n(n1, n2, n3, n4, n5):
  '''Represent :math:`b(\\boldsymbol{n})` in :eq:`b-n` in Poly
  
  :param n1: times of :math:`\\theta` being selected.
  :param n2: times of :math:`v_{n-1}` being selected.
  :param n3: times of :math:`eI_n` being selected.
  :param n4: times of :math:`I_n` being selected.
  :param n5: times of :math:`I_n^{*}` being selected.
  
  :return: a poly with attribute ``keyfor`` = 
     ('e^{-kh}','k^{-}','theta','sigma_v','rho','sqrt(1-rho^2)').
  :rtype: Poly
  '''
  b = Poly()
  b.set_keyfor(['e^{-kh}','k^{-}','theta','sigma_v','rho','sqrt(1-rho^2)'])
  # 
  for i in range(n1+n2+1):
    for j in range(n4+1):
      key = (i, n1+n2+n3+j, n1, n3+j, n4-j, n5)
      num = math.comb(n1+n2, i) * math.comb(n4, j) * ((-1)**(n2+i+j))
      den = 2**(n1+n2+n3+j)
      b.add_keyval(key, Frac(num, den))
  return(b)

def ve_IEII(n2, n3, n4, n5):
  ''':math:`v_{n-1}^{n_2} e^{-n_3knh} IEII`
  
  :param n2: :math:`n_2` in above formula.
  :param n3: :math:`n_3` in :math:`IEII` which is
     :math:`E[eI_n^{n_3}I_n^{n_4}I_n^{*n_5}|v_{n-1}]`.
  :param n4: :math:`n_4` in :math:`IEII`.
  :param n5: :math:`n_5` in :math:`IEII`.
  
  :return: poly with attribute ``keyfor`` = 
     ('e^{-kh}','h','v_{n-1}','k^{-}','theta','sigma_v').
  :rtype: Poly
  '''
  poln = Poly()
  poln.set_keyfor(['e^{-kh}','h','v_{n-1}','k^{-}','theta','sigma_v'])
  # 
  poly = moment_IEII(n3, n4, n5)
  # with keyfor
  # ['e^{k(n-1)h}','e^{kh}','h','v_{n-1}','k^{-}','theta','sigma_v']
  # and the power for the first one always = n3
  for k in poly.keys():
    # × e^{-n_3knh}      × v_{n-1}^{n_2}
    key = (n3-k[1], k[2], n2+k[3], k[4], k[5], k[6])
    poln.add_keyval(key, poly[k])
  return(poln)

def moment_comb(n, n1, n2, n3, n4, n5):
  '''Moment for this combination in expansion of :math:`\overline{y}_n^l`
  
  :param n: l in :math:`E[\overline{y}_{n}^l]`.
  :param n1: times of :math:`\\theta` being selected.
  :param n2: times of :math:`v_{n-1}` being selected.
  :param n3: times of :math:`eI_n` being selected.
  :param n4: times of :math:`I_n` being selected.
  :param n5: times of :math:`I_n^{*}` being selected.
  
  :return: poly with attribute ``keyfor`` = 
     ('e^{-kh}','h','v_{n-1}','k^{-}','theta','sigma_v','rho','sqrt(1-rho^2)').
  :rtype: Poly
  '''
  poly = ve_IEII(n2, n3, n4, n5)
  # ['e^{-kh}','h','v_{n-1}','k^{-}','theta','sigma_v']
  b = b_n(n1, n2, n3, n4, n5)
  # ['e^{-kh}','k^{-}','theta','sigma_v','rho','sqrt(1-rho^2)']
  #
  keyfor = ['e^{-kh}','h','v_{n-1}','k^{-}','theta','sigma_v','rho',
    'sqrt(1-rho^2)'] # keyfor for new poly
  #
  keyIndexes = ([0,1,2,3,4,5,-1,-1], [0,-1,-1,1,2,3,4,5])
  # -1 for not having this term
  poly = poly.mul_poly(b, keyIndexes, keyfor)
  #
  c = c_n(n, n1, n2, n3, n4, n5)
  return(c * poly)

def sub_v(poly):
  '''Substitute :math:`v_{n-1}` with its moment (poly) in Central Moment of y_n
  
  :param poly: poly with attribute ``keyfor`` = 
     ('e^{-kh}','h','v_{n-1}','k^{-}','theta','sigma_v','rho','sqrt(1-rho^2)').
  
  :return: poly with attribute ``keyfor`` =
     ('e^{-kh}','h','k^{-}','theta','sigma_v','rho','sqrt(1-rho^2)')
  :rtype: Poly
  '''
  poly_sum = Poly()
  kf = ['e^{-kh}','h','k^{-}','theta','sigma_v','rho','sqrt(1-rho^2)']
  poly_sum.set_keyfor(kf)
  # 
  for k in poly:
    poln = moment_v(k[2]) # ['theta','sigma_v^2/k']
    for key in poln:
      i, j = key
      knw = (k[0], k[1], k[3]+j, k[4]+i, k[5]+2*j, k[6], k[7])
      val = poln[key] * poly[k]
      poly_sum.add_keyval(knw, val)
  return(poly_sum)

def cmoment_y(l):
  '''Central moment of :math:`y_n` of order :math:`l`
  
  :param l: order of the central moment.
  
  :return: poly with attribute ``keyfor`` =
     ('e^{-kh}','h','k^{-}','theta','sigma_v','rho','sqrt(1-rho^2)').
  :rtype: Poly
  '''
  poly = Poly()
  kf = ['e^{-kh}','h','v_{n-1}','k^{-}','theta','sigma_v','rho','sqrt(1-rho^2)']
  poly.set_keyfor(kf)
  # 
  # if l == 0: 
  #   poln = Poly({(0,0,0,0,0,0,0,0):1}); poln.set_keyfor(kf); return(poln)
  # 
  for n1 in range(l, -1, -1):
    for n2 in range(l-n1, -1, -1):
      for n3 in range(l-n1-n2, -1, -1):
        for n4 in range(l-n1-n2-n3, -1, -1):
          n5 = l - n1 - n2 - n3 - n4
          poly.merge(moment_comb(l, n1, n2, n3, n4, n5))
  poly_sum = sub_v(poly)
  poly_sum.remove_zero()
  return(poly_sum)

##########
# scalar and (partial) derivative
##########

def dpoly(poly, wrt):
  '''Partial derivative of central moment w.r.t. parameter wrt
  
  :param poly: central moment represented by the poly.
  :param wrt: with respect to.
  
  :return: poly with attribute ``keyfor`` = 
     ('e^{-kh}','h','k^{-}','theta','sigma_v','rho','sqrt(1-rho^2)').
  :rtype: Poly
  '''
  pold = Poly()
  kf = ('e^{-kh}','h','k^{-}','theta','sigma_v','rho','sqrt(1-rho^2)')
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
  elif wrt in ['theta','sigma_v']:
    if wrt == 'theta': i = 3
    if wrt == 'sigma_v': i = 4
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
        knw = list(k); knw[5] += 1; knw[6] -= 2; knw = tuple(knw)
        val = (-k[6]) * poly[k]
        pold.add_keyval(knw, val)
  else:
    raise ValueError("wrt must be either 'k', 'theta', 'sigma_v' or 'rho'!")
  return(pold)

def poly2num(poly, par):
  '''Decode poly back to scalar
  
  :param poly: poly to be decoded with attibute ``keyfor`` = 
     ('e^{-kh}','h','k^{-}','theta','sigma_v','rho','sqrt(1-rho^2)').
  :param par: parameters in dict.
  
  :return: scalar of the poly.
  :rtype: float
  '''
  # ('e^{-kh}','h','k^{-}','theta','sigma_v','rho','sqrt(1-rho^2)')
  k = par['k']
  h = par['h']
  theta = par['theta']
  sigma_v = par['sigma_v']
  rho = par['rho']
  # 
  value = 0
  for K in poly:
    val = poly[K] * math.exp(-K[0]*k*h) * (h ** K[1]) * (k ** (-K[2]))
    val *= (theta ** K[3]) * (sigma_v ** K[4]) * (rho ** K[5])
    val *= (1-rho**2) ** (K[6]/2)
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
  # Example usage of the module, see 'tests/test_mdl_1fsv.py' for more test
  from pprint import pprint
  print('\nExample usage of the module function\n')
  # 
  kf = ('e^{-kh}','h','k^{-}','theta','sigma_v','rho','sqrt(1-rho^2)')
  print(f"cmoment_y(l) returns a poly with keyfor = \n{kf}")
  print("cmoment_y(3) = "); pprint(cmoment_y(3))

'''
Covariance for One-Factor SV with jumps
'''
import math

from ajdmom.poly import Poly
from ajdmom.mdl_1fsv.cov import moment_yy as m_yy
from ajdmom.cpp_mom import mcpp

from ajdmom.mdl_1fsvj.mom import (
  moment_y,
  dpoly,
  poly2num
)
def moment_yy(l1, l2):
  '''Moment :math:`E[y_n^{l_1}y_{n+1}^{l_2}]`

  :param l1: *l1* in :math:`E[y_n^{l_1}y_{n+1}^{l_2}]`.
  :param l2: *l2* in :math:`E[y_n^{l_1}y_{n+1}^{l_2}]`.

  :return: poly with attribute ``keyfor`` =
     ('e^{-kh}','h','k^{-}','mu','theta','sigma_v','rho','sqrt(1-rho^2)',
     'lambda','mu_j','sigma_j').
  :rtype: Poly
  '''
  poly = Poly()
  kf = ['e^{-kh}','h','k^{-}','mu','theta','sigma_v','rho','sqrt(1-rho^2)',
    'lambda','mu_j','sigma_j']
  poly.set_keyfor(kf)
  #
  for i in range(l2+1):
    for j in range(l1+1):
      coef = math.comb(l2, i) * math.comb(l1, j)
      pol1 = m_yy(j, i)
      # ('e^{-kh}','h','k^{-}','mu','theta','sigma_v','rho','sqrt(1-rho^2)')
      pol2 = mcpp(l2-i) # ('lambda*h','mu','sigma')
      pol3 = mcpp(l1-j) # ('lambda*h','mu','sigma')
      keyIndexes = [(0,1,2,3,4,5,6,7,-1,-1,-1),(-1,0,-1,-1,-1,-1,-1,-1,0,1,2)]
      poln = pol1.mul_poly(pol2*pol3, keyIndexes, kf)
      poly.merge(coef * poln)
  return(poly)

def cov_yy(l1, l2):
  '''Moment :math:`cov(y_n^{l_1},y_{n+1}^{l_2})`

  :param l1: *l1* in :math:`E[y_n^{l_1}y_{n+1}^{l_2}]`.
  :param l2: *l2* in :math:`E[y_n^{l_1}y_{n+1}^{l_2}]`.

  :return: poly with attribute ``keyfor`` =
     ('e^{-kh}','h','k^{-}','mu','theta','sigma_v','rho','sqrt(1-rho^2)',
     'lambda','mu_j','sigma_j').
  :rtype: Poly
  '''
  cov = moment_yy(l1, l2) - (moment_y(l1)*moment_y(l2))
  cov.remove_zero()
  return(cov)

##########
# scalar and (partial) derivative
##########

def cov(l1, l2, par):
  '''Covariance in scalar

  :param l1: *l1* in :math:`E[y_n^{l_1}y_{n+1}^{l_2}]`.
  :param l2: *l2* in :math:`E[y_n^{l_1}y_{n+1}^{l_2}]`.
  :param par: parameters in dict.

  :return: scalar of the covariance.
  :rtype: float
  '''
  covariance = cov_yy(l1, l2)
  value = poly2num(covariance, par)
  return(value)

def dcov(l1, l2, par, wrt):
  '''Partial derivative of covariance w.r.t. parameter wrt

  :param l1: *l1* in :math:`E[y_n^{l_1}y_{n+1}^{l_2}]`.
  :param l2: *l2* in :math:`E[y_n^{l_1}y_{n+1}^{l_2}]`.
  :param par: parameters in dict.
  :param wrt: with respect to.

  :return: scalar of the partial derivative.
  :rtype: float
  '''
  covariance = cov_yy(l1, l2)
  pold = dpoly(covariance, wrt)
  value = poly2num(pold, par)
  return(value)


if __name__ == "__main__":
  # Example usage of the module, see 'tests/test_mdl_1fsvj.py' for more test
  from pprint import pprint
  print('\nExample usage of the module function\n')
  #
  kf = ('e^{-kh}','h','k^{-}','mu','theta','sigma_v','rho','sqrt(1-rho^2)',
    'lambda','mu_j','sigma_j^2')
  print(f"cov_yy(l1,l2) returns a poly with keyfor = \n{kf}")
  print("cov_yy(2,1) = "); pprint(cov_yy(2,1))
  # # usually l1 >= 1 and l2 >= 1
  # print("cov_yy(l1=0,l2=0): "); pprint(cov_yy(l1=0,l2=0))
  # print("cov_yy(l1=0,l2=1): "); pprint(cov_yy(l1=0,l2=1))
  # print("cov_yy(l1=1,l2=0): "); pprint(cov_yy(l1=1,l2=0))
  # #

from fractions import Fraction as Frac
import pytest

from ajdmom.poly import Poly
from ajdmom.ito_mom import moment_v
from ajdmom.ito_mom import moment_IEI
from ajdmom.ito_mom import moment_IEII

@pytest.mark.parametrize(
  "n, poly",
  [
    (0, {(0,0): Frac(1,1)}),
    (1, {(1,0): Frac(1,1)}),
    (2, {(2,0): Frac(1,1), (1,1): Frac(1,2)}),
    (3, {(3,0): Frac(1,1), (2,1): Frac(3,2), (1,2): Frac(1,2)})
  ]
)
def test_moment_v(n, poly):
  '''Test moment_v()'''
  expected = Poly(poly); 
  expected.set_keyfor(['theta','sigma_v^2/k'])
  # 
  assert moment_v(n) == expected

@pytest.mark.parametrize(
  "n3n4, poly",
  [
    ((0,0), {(0, 0, 0, 0, 0, 0, 0): Frac(1, 1)}),
    ((1,0), {}),
    ((0,1), {}),
    ((2,0), {(2, 0, 0, 0, 1, 1, 0): Frac(1, 2),
             (2, 0, 0, 1, 1, 0, 0): Frac(-1, 1),
             (2, 1, 0, 0, 1, 1, 0): Frac(-1, 1),
             (2, 1, 0, 1, 1, 0, 0): Frac(1, 1),
             (2, 2, 0, 0, 1, 1, 0): Frac(1, 2)}),
    ((1,1), {(1, 0, 0, 0, 1, 1, 0): Frac(-1, 1),
             (1, 0, 1, 0, 0, 1, 0): Frac(-1, 1),
             (1, 0, 1, 1, 0, 0, 0): Frac(1, 1),
             (1, 1, 0, 0, 1, 1, 0): Frac(1, 1)}),
    ((0,2), {(0, -1, 0, 0, 1, 1, 0): Frac(1, 1),
             (0, -1, 0, 1, 1, 0, 0): Frac(-1, 1),
             (0, 0, 0, 0, 1, 1, 0): Frac(-1, 1),
             (0, 0, 0, 1, 1, 0, 0): Frac(1, 1),
             (0, 0, 1, 0, 0, 1, 0): Frac(1, 1)}),
    ((3,0), {(3, 0, 0, 0, 2, 1, 1): Frac(-1, 2),
             (3, 0, 0, 1, 2, 0, 1): Frac(3, 2),
             (3, 1, 0, 0, 2, 1, 1): Frac(3, 2),
             (3, 1, 0, 1, 2, 0, 1): Frac(-3, 1),
             (3, 2, 0, 0, 2, 1, 1): Frac(-3, 2),
             (3, 2, 0, 1, 2, 0, 1): Frac(3, 2),
             (3, 3, 0, 0, 2, 1, 1): Frac(1, 2)}),
    ((2,1), {(2, 0, 0, 0, 2, 1, 1): Frac(1, 1),
             (2, 0, 0, 1, 2, 0, 1): Frac(-1, 1),
             (2, 0, 1, 0, 1, 1, 1): Frac(1, 1),
             (2, 0, 1, 1, 1, 0, 1): Frac(-2, 1),
             (2, 1, 0, 0, 2, 1, 1): Frac(-2, 1),
             (2, 1, 0, 1, 2, 0, 1): Frac(1, 1),
             (2, 1, 1, 0, 1, 1, 1): Frac(-1, 1),
             (2, 1, 1, 1, 1, 0, 1): Frac(1, 1),
             (2, 2, 0, 0, 2, 1, 1): Frac(1, 1)}),
    ((1,2), {(1, -1, 0, 0, 2, 1, 1): Frac(-1, 2),
             (1, -1, 0, 1, 2, 0, 1): Frac(1, 1),
             (1, 0, 0, 0, 2, 1, 1): Frac(-2, 1),
             (1, 0, 0, 1, 2, 0, 1): Frac(-1, 1),
             (1, 0, 1, 0, 1, 1, 1): Frac(-3, 1),
             (1, 0, 1, 1, 1, 0, 1): Frac(1, 1),
             (1, 0, 2, 0, 0, 1, 1): Frac(-1, 1),
             (1, 0, 2, 1, 0, 0, 1): Frac(1, 1),
             (1, 1, 0, 0, 2, 1, 1): Frac(5, 2)}),
    ((0,3), {(0, -1, 0, 0, 2, 1, 1): Frac(6, 1),
             (0, -1, 0, 1, 2, 0, 1): Frac(-3, 1),
             (0, -1, 1, 0, 1, 1, 1): Frac(3, 1),
             (0, -1, 1, 1, 1, 0, 1): Frac(-3, 1),
             (0, 0, 0, 0, 2, 1, 1): Frac(-6, 1),
             (0, 0, 0, 1, 2, 0, 1): Frac(3, 1),
             (0, 0, 1, 0, 1, 1, 1): Frac(3, 1)}),
    ((4,0), {(4, 0, 0, 0, 2, 2, 0): Frac(3, 4),
             (4, 0, 0, 0, 3, 1, 2): Frac(3, 4),
             (4, 0, 0, 1, 2, 1, 0): Frac(-3, 1),
             (4, 0, 0, 1, 3, 0, 2): Frac(-3, 1),
             (4, 0, 0, 2, 2, 0, 0): Frac(3, 1),
             (4, 1, 0, 0, 2, 2, 0): Frac(-3, 1),
             (4, 1, 0, 0, 3, 1, 2): Frac(-3, 1),
             (4, 1, 0, 1, 2, 1, 0): Frac(9, 1),
             (4, 1, 0, 1, 3, 0, 2): Frac(9, 1),
             (4, 1, 0, 2, 2, 0, 0): Frac(-6, 1),
             (4, 2, 0, 0, 2, 2, 0): Frac(9, 2),
             (4, 2, 0, 0, 3, 1, 2): Frac(9, 2),
             (4, 2, 0, 1, 2, 1, 0): Frac(-9, 1),
             (4, 2, 0, 1, 3, 0, 2): Frac(-9, 1),
             (4, 2, 0, 2, 2, 0, 0): Frac(3, 1),
             (4, 3, 0, 0, 2, 2, 0): Frac(-3, 1),
             (4, 3, 0, 0, 3, 1, 2): Frac(-3, 1),
             (4, 3, 0, 1, 2, 1, 0): Frac(3, 1),
             (4, 3, 0, 1, 3, 0, 2): Frac(3, 1),
             (4, 4, 0, 0, 2, 2, 0): Frac(3, 4),
             (4, 4, 0, 0, 3, 1, 2): Frac(3, 4)})
  ]
)
def test_moment_IEI(n3n4, poly):
  '''Test moment_IEI'''
  expected = Poly(poly)
  kf = ('e^{k(n-1)h}','e^{k[t-(n-1)h]}','[t-(n-1)h]','v_{n-1}','k^{-}',
    'theta','sigma_v')
  expected.set_keyfor(kf)
  # 
  n3, n4 = n3n4
  assert moment_IEI(n3, n4) == expected

@pytest.mark.parametrize(
  "n3n4n5, poly",
  [
    ((0,0,0), {(0, 0, 0, 0, 0, 0, 0): Frac(1, 1)}),
    # 
    ((1,0,0), {}),
    ((0,1,0), {}),
    ((0,0,1), {}),
    # 
    ((2,0,0), {(2, 0, 0, 0, 1, 1, 0): Frac(1, 2),
               (2, 0, 0, 1, 1, 0, 0): Frac(-1, 1),
               (2, 1, 0, 0, 1, 1, 0): Frac(-1, 1),
               (2, 1, 0, 1, 1, 0, 0): Frac(1, 1),
               (2, 2, 0, 0, 1, 1, 0): Frac(1, 2)}),
    ((1,1,0), {(1, 0, 0, 0, 1, 1, 0): Frac(-1, 1),
               (1, 0, 1, 0, 0, 1, 0): Frac(-1, 1),
               (1, 0, 1, 1, 0, 0, 0): Frac(1, 1),
               (1, 1, 0, 0, 1, 1, 0): Frac(1, 1)}),
    ((1,0,1), {}),
    ((0,2,0), {(0, -1, 0, 0, 1, 1, 0): Frac(1, 1),
               (0, -1, 0, 1, 1, 0, 0): Frac(-1, 1),
               (0, 0, 0, 0, 1, 1, 0): Frac(-1, 1),
               (0, 0, 0, 1, 1, 0, 0): Frac(1, 1),
               (0, 0, 1, 0, 0, 1, 0): Frac(1, 1)}),
    ((0,1,1), {}),
    ((0,0,2), {(0, -1, 0, 0, 1, 1, 0): Frac(1, 1),
               (0, -1, 0, 1, 1, 0, 0): Frac(-1, 1),
               (0, 0, 0, 0, 1, 1, 0): Frac(-1, 1),
               (0, 0, 0, 1, 1, 0, 0): Frac(1, 1),
               (0, 0, 1, 0, 0, 1, 0): Frac(1, 1)}),
    # 
    ((3,0,0), {(3, 0, 0, 0, 2, 1, 1): Frac(-1, 2),
               (3, 0, 0, 1, 2, 0, 1): Frac(3, 2),
               (3, 1, 0, 0, 2, 1, 1): Frac(3, 2),
               (3, 1, 0, 1, 2, 0, 1): Frac(-3, 1),
               (3, 2, 0, 0, 2, 1, 1): Frac(-3, 2),
               (3, 2, 0, 1, 2, 0, 1): Frac(3, 2),
               (3, 3, 0, 0, 2, 1, 1): Frac(1, 2)}),
    ((2,1,0), {(2, 0, 0, 0, 2, 1, 1): Frac(1, 1),
               (2, 0, 0, 1, 2, 0, 1): Frac(-1, 1),
               (2, 0, 1, 0, 1, 1, 1): Frac(1, 1),
               (2, 0, 1, 1, 1, 0, 1): Frac(-2, 1),
               (2, 1, 0, 0, 2, 1, 1): Frac(-2, 1),
               (2, 1, 0, 1, 2, 0, 1): Frac(1, 1),
               (2, 1, 1, 0, 1, 1, 1): Frac(-1, 1),
               (2, 1, 1, 1, 1, 0, 1): Frac(1, 1),
               (2, 2, 0, 0, 2, 1, 1): Frac(1, 1)}),
    ((2,0,1), {}),
    ((1,2,0), {(1, -1, 0, 0, 2, 1, 1): Frac(-1, 2),
               (1, -1, 0, 1, 2, 0, 1): Frac(1, 1),
               (1, 0, 0, 0, 2, 1, 1): Frac(-2, 1),
               (1, 0, 0, 1, 2, 0, 1): Frac(-1, 1),
               (1, 0, 1, 0, 1, 1, 1): Frac(-3, 1),
               (1, 0, 1, 1, 1, 0, 1): Frac(1, 1),
               (1, 0, 2, 0, 0, 1, 1): Frac(-1, 1),
               (1, 0, 2, 1, 0, 0, 1): Frac(1, 1),
               (1, 1, 0, 0, 2, 1, 1): Frac(5, 2)}),
    ((1,1,1), {}),
    ((1,0,2), {(1, -1, 0, 0, 2, 1, 1): Frac(-1, 2),
               (1, -1, 0, 1, 2, 0, 1): Frac(1, 1),
               (1, 0, 0, 1, 2, 0, 1): Frac(-1, 1),
               (1, 0, 1, 0, 1, 1, 1): Frac(-1, 1),
               (1, 0, 1, 1, 1, 0, 1): Frac(1, 1),
               (1, 1, 0, 0, 2, 1, 1): Frac(1, 2)}),
    ((0,3,0), {(0, -1, 0, 0, 2, 1, 1): Frac(6, 1),
               (0, -1, 0, 1, 2, 0, 1): Frac(-3, 1),
               (0, -1, 1, 0, 1, 1, 1): Frac(3, 1),
               (0, -1, 1, 1, 1, 0, 1): Frac(-3, 1),
               (0, 0, 0, 0, 2, 1, 1): Frac(-6, 1),
               (0, 0, 0, 1, 2, 0, 1): Frac(3, 1),
               (0, 0, 1, 0, 1, 1, 1): Frac(3, 1)}),
    ((0,2,1), {}),
    ((0,1,2), {(0, -1, 0, 0, 2, 1, 1): Frac(2, 1),
               (0, -1, 0, 1, 2, 0, 1): Frac(-1, 1),
               (0, -1, 1, 0, 1, 1, 1): Frac(1, 1),
               (0, -1, 1, 1, 1, 0, 1): Frac(-1, 1),
               (0, 0, 0, 0, 2, 1, 1): Frac(-2, 1),
               (0, 0, 0, 1, 2, 0, 1): Frac(1, 1),
               (0, 0, 1, 0, 1, 1, 1): Frac(1, 1)}),
    ((0,0,3), {}),
    # 
    ((4,0,0), {(4, 0, 0, 0, 2, 2, 0): Frac(3, 4),
               (4, 0, 0, 0, 3, 1, 2): Frac(3, 4),
               (4, 0, 0, 1, 2, 1, 0): Frac(-3, 1),
               (4, 0, 0, 1, 3, 0, 2): Frac(-3, 1),
               (4, 0, 0, 2, 2, 0, 0): Frac(3, 1),
               (4, 1, 0, 0, 2, 2, 0): Frac(-3, 1),
               (4, 1, 0, 0, 3, 1, 2): Frac(-3, 1),
               (4, 1, 0, 1, 2, 1, 0): Frac(9, 1),
               (4, 1, 0, 1, 3, 0, 2): Frac(9, 1),
               (4, 1, 0, 2, 2, 0, 0): Frac(-6, 1),
               (4, 2, 0, 0, 2, 2, 0): Frac(9, 2),
               (4, 2, 0, 0, 3, 1, 2): Frac(9, 2),
               (4, 2, 0, 1, 2, 1, 0): Frac(-9, 1),
               (4, 2, 0, 1, 3, 0, 2): Frac(-9, 1),
               (4, 2, 0, 2, 2, 0, 0): Frac(3, 1),
               (4, 3, 0, 0, 2, 2, 0): Frac(-3, 1),
               (4, 3, 0, 0, 3, 1, 2): Frac(-3, 1),
               (4, 3, 0, 1, 2, 1, 0): Frac(3, 1),
               (4, 3, 0, 1, 3, 0, 2): Frac(3, 1),
               (4, 4, 0, 0, 2, 2, 0): Frac(3, 4),
               (4, 4, 0, 0, 3, 1, 2): Frac(3, 4)})
  ]
)
def test_moment_IEII(n3n4n5, poly):
  expected = Poly(poly)
  kf = ('e^{k(n-1)h}','e^{k[t-(n-1)h]}','[t-(n-1)h]','v_{n-1}','k^{-}',
    'theta','sigma_v')
  expected.set_keyfor(kf)
  # 
  n3, n4, n5 = n3n4n5
  assert moment_IEII(n3, n4, n5) == expected


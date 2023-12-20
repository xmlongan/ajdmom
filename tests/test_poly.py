import pytest
from ajdmom.poly import Poly

@pytest.fixture
def two_polys():
  pol1 = Poly({(1,0): 1, (0,1): 1})
  pol2 = Poly({(1,0): 1, (0,2): 1})
  pol1.set_keyfor(['e','h'])
  pol2.set_keyfor(['e','h'])
  # 
  return((pol1,pol2))

def test_add(two_polys):
  '''Test Poly addition operation'''
  expected = Poly({(1,0): 2, (0,1): 1, (0,2): 1})
  expected.set_keyfor(['e','h'])
  #
  pol1, pol2 = two_polys
  actual = pol1 + pol2
  # 
  assert actual == expected, "Poly addition operation is wrong!"
  # 
  msg = "commutative law of addition is broken!"
  assert pol1 + pol2 == pol2 + pol1, msg

def test_merge(two_polys):
  '''Test Poly merge operation'''
  expected = Poly({(1,0): 2, (0,1): 1, (0,2): 1})
  expected.set_keyfor(['e','h'])
  #
  pol1, pol2 = two_polys
  pol1.merge(pol2)
  actual = pol1
  # 
  assert actual == expected, "Poly merge operation is wrong!"

def test_sub(two_polys):
  '''Test Poly subtraction operation'''
  expected = Poly({(0,1): 1, (0,2):-1})
  expected.set_keyfor(['e','h'])
  # 
  pol1, pol2 = two_polys
  actual = pol1 - pol2
  # 
  assert actual == expected, "Poly subtraction operation is wrong!"

def test_mul(two_polys):
  '''Test Poly multiplication operation'''
  expected = Poly({(2,0): 1, (1,2): 1, (1,1): 1, (0,3):1})
  expected.set_keyfor(['e','h'])
  # 
  pol1, pol2 = two_polys
  actual = pol1 * pol2
  # 
  assert actual == expected, "Poly multiplication operation is wrong!"

def test_rmul(two_polys):
  '''Test Poly reverse multiplication operation'''
  expected = Poly({(1,0): 4, (0,1): 4})
  expected.set_keyfor(['e','h'])
  # 
  pol1, pol2 = two_polys
  actual = 4 * pol1
  # 
  assert actual == expected, "Poly reverse multiplication is wrong!"

def test_pow(two_polys):
  '''Test Poly power operation'''
  expected = Poly({(2,0): 1, (1,1): 2, (0,2): 1})
  expected.set_keyfor(['e','h'])
  # 
  pol1, pol2 = two_polys
  actual = pol1 ** 2
  # 
  assert actual == expected, "Poly power is wrong!"
  # 
  expected = Poly({(3,0): 1, (2,1): 3, (1,2): 3, (0,3): 1})
  expected.set_keyfor(['e','h'])
  # 
  actual = pol1 ** 3
  # 
  assert actual == expected, "Poly power is wrong still!"

def test_mul_poly(two_polys):
  '''Test Poly multiplication with a different poly'''
  poln = Poly({(1,): 100, (2,0): 100})
  # 
  pol1, pol2 = two_polys
  # 
  expected = Poly({(2,0): 100, (1,1): 100, (3,0): 100, (2,1): 100})
  # 
  actual = pol1.mul_poly(poln, keyIndexes=([0,1],[0,-1]), keyfor=('e','h'))
  # 
  msg = "Poly multiplication of different type Poly is wrong!"
  assert actual == expected, msg


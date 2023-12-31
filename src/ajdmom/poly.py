'''
Class Poly, a customized dictionary, derived from :code:`UserDict`

I defined a new class :py:class:`~ajdmom.poly.Poly`
to extend the :class:`~collections.UserDict` class
from the Python Standard Library :code:`collections`, within which
I defined an attribute :py:attr:`~ajdmom.poly.Poly.keyfor` and the
following magic methods:

+-------------+---------------------+---------------------------------------+
|Magic Method | Example             |Function                               |
+=============+=====================+=======================================+
|__add__()    |:code:`poly1 + poly2`|Addition between two polynomials       |
+-------------+---------------------+---------------------------------------+
|__sub__()    |:code:`poly1 - poly2`|Subtraction between two polynomials    |
+-------------+---------------------+---------------------------------------+
|__mul__()    |:code:`poly1 * poly2`|Multiplication between two polynomials |
+-------------+---------------------+---------------------------------------+
|__rmul__()   |:code:`c * poly2`    |Reversely multiply by a constant       |
+-------------+---------------------+---------------------------------------+
|__pow__()    |:code:`poly ** 2`    |Raise a polynomial to  power n         |
+-------------+---------------------+---------------------------------------+

It should be noted that:

* Addition, subtraction and multiplication are only for polynomials having
  the same :py:attr:`~ajdmom.poly.Poly.keyfor` attribute.

* The original poly or polys remain unchanged after these operations,
  all of which return a new poly.

Besides, I defined following methods:

+------------------------------------------+------------------------------------------------------------------+
|Method                                    |Function                                                          |
+==========================================+==================================================================+
|:py:meth:`~ajdmom.poly.Poly.merge`        |Merge with another poly having the same keyfor attribute          |
+------------------------------------------+------------------------------------------------------------------+
|:py:meth:`~ajdmom.poly.Poly.add_keyval`   |Insert a new key-val or add val to the existing key               |
+------------------------------------------+------------------------------------------------------------------+
|:py:meth:`~ajdmom.poly.Poly.remove_zero`  |Remove any item having value 0                                    |
+------------------------------------------+------------------------------------------------------------------+
|:py:meth:`~ajdmom.poly.Poly.set_keyfor`   |Set the ``keyfor`` attribute for the poly                         |
+------------------------------------------+------------------------------------------------------------------+
|:py:meth:`~ajdmom.poly.Poly.mul_poly`     |Multiply by another poly of different type and return a new one   |
+------------------------------------------+------------------------------------------------------------------+
|:py:meth:`~ajdmom.poly.Poly.is_exact_type`|Check whether the supplied argument is a Poly with the same keyfor|
+------------------------------------------+------------------------------------------------------------------+

Note that :py:meth:`~ajdmom.poly.Poly.merge` (in-place) is different from
the magic method :code:`__add__()` (not-in-place). The former changes the
original poly, while the latter does not.
'''
from collections import UserDict

def kv(i, key):
  '''get key[i] where key is a tuple

  :param i: index of key or -1 for not having the element.
  :param key: tuple as a key for Poly.

  :return: key[i] if i in range(len(k)), otherwise 0.
  :rtype: int
  '''
  if i in range(len(key)):
    return(key[i])
  else:
    return(0)

class Poly(UserDict):
  '''Class for different versions of "Polynomial" '''

  keyfor = ()
  "Indicating the purpose for each key component, a tuple of str."

  def __add__(self, other):
    '''Add another poly having the same keyfor'''
    if not self.is_exact_type(other):
      msg = "The operands must be Polys having the same 'keyfor' attribute."
      raise NotImplementedError(msg)
    #
    # initialize a new poly to isolate the operation from affecting
    # the original one
    #
    poly = Poly(self) # UserDict initialization
    poly.set_keyfor(self.keyfor)
    #
    for k in other:
      poly.add_keyval(k, other[k])
    #
    poly.remove_zero() # remove any item with 0 value
    return(poly)

  def merge(self, other):
    '''Merge another poly having the same keyfor attribute

    Insert new key-value or add value for the existing key.

    :param other: poly having the same keyfor.

    :return: the updated first poly.
    :rtype: Poly
    '''
    if not self.is_exact_type(other):
      msg = "The operands must be Polys having the same 'keyfor' attribute."
      raise NotImplementedError(msg)
    #
    for k in other:
      self.add_keyval(k, other[k])
    #
    self.remove_zero() # remove any item with 0 value

  def __sub__(self, other):
    '''Subtract another poly having the same keyfor'''
    if not self.is_exact_type(other):
      msg = "The operands must be Polys having the same 'keyfor' attribute."
      raise NotImplementedError(msg)
    #
    # initialize a new poly to isolate the operation from affecting
    # the original one
    #
    poly = Poly(self) # UserDict initialization
    poly.set_keyfor(self.keyfor)
    #
    for k in other:
      if k in self:
        poly[k] -= other[k]
      else:
        poly[k] = -other[k]
    #
    poly.remove_zero() # remove any item with 0 value
    return(poly)

  def __mul__(self, other):
    '''Multiply by another poly having the same keyfor attribute'''
    if not self.is_exact_type(other):
      msg = "The operands must be Polys having the same 'keyfor' attribute."
      raise NotImplementedError(msg)
    #
    # initialize a new poly to isolate the operation from affecting
    # the original one
    #
    poly = Poly()
    poly.set_keyfor(self.keyfor)
    #
    for k1 in self:
      for k2 in other:
        key = tuple(k1[i]+k2[i] for i in range(len(k1)))
        val = self[k1] * other[k2]
        poly.add_keyval(key, val)
    #
    poly.remove_zero() # remove any item with 0 value
    return(poly)

  def __rmul__(self, c):
    '''Reversely multiply by a constant'''
    # Should we consider the special case c = 0?
    #
    # initialize a new poly to isolate the operation from affecting
    # the original one
    #
    poly = Poly()
    poly.set_keyfor(self.keyfor)
    #
    for k in self:
      val = self[k] * c
      poly.add_keyval(k, val)
    #
    poly.remove_zero() # remove any item with 0 value
    return(poly)

  def __pow__(self, n):
    '''Raise the poly to power n'''
    #
    if n <= 0:
      raise NotImplementedError("n must be an integer and >= 1.")
    #
    # initialize a new poly to isolate the operation from affecting
    # the original one
    #
    poly = Poly(self)
    poly.set_keyfor(self.keyfor)
    #
    for i in range(n-1):
      poly = poly * self
    return(poly)

  def add_keyval(self, key, val):
    '''Insert a new key-val or add val to the existing key

    :param key: key for the poly, a tuple.
    :param val: corresponding value.
    '''
    if key in self:
      self[key] += val
    else:
      self[key]  = val

  def remove_zero(self):
    '''Remove any item having value 0'''
    # val is n/m, fraction number
    ks = [k for k in self if self[k] == 0] # works for fraction number
    for k in ks:
      del self[k]

  def set_keyfor(self, names):
    '''Set the ``keyfor`` attribute for the poly

    :param names: a sequence of names,
       each corresponding to the key tuple counterpart,
       i.e., names[i] v.s. key[i].
    '''
    self.keyfor = tuple(names)

  def mul_poly(self, other, keyIndexes, keyfor):
    '''Multiply by another poly of different type and return a new poly

    :param other: another poly having a different 'keyfor' attribute.
    :param keyIndexes: a tuple with two lists,
       keyIndexes[0] for self,
       keyIndexes[1] for other.
    :param keyfor: the 'keyfor' attribute for the returned poly.

    :return: a poly with 'keyfor' attribute.
    :rtype: Poly
    '''
    #
    # initialize a new poly to isolate the operation from affecting
    # the original one
    #
    poly = Poly()
    poly.set_keyfor(keyfor)
    #
    idx1, idx2 = keyIndexes
    kN = len(idx1)
    #
    for k1 in self:
      for k2 in other:
        key = tuple(kv(idx1[i],k1) + kv(idx2[i],k2) for i in range(kN))
        val = self[k1] * other[k2]
        poly.add_keyval(key, val)
    #
    poly.remove_zero()
    return(poly)

  def is_exact_type(self, other):
    '''Check whether the supplied argument is a Poly having the same keyfor'''
    flag = True
    if not isinstance(other, Poly):
      flag = False
    else:
      try:
        if len(self.keyfor) != len(other.keyfor):
          flag = False
        else:
          for i in range(len(self.keyfor)):
            if self.keyfor[i] != other.keyfor[i]:
              flag = False
              break
      except AttributeError:
        flag = False
    return(flag)

if __name__ == "__main__":
  # Example usage of class Poly, see 'tests/test_poly.py' for more test
  print('\nExample usage of class Poly\n')
  #
  pol1 = Poly({(1,0): 1, (0,1): 1}); pol1.set_keyfor(['e', 'h'])
  pol2 = Poly({(1,0): 1, (0,2): 1}); pol2.set_keyfor(['e', 'h'])
  print(f'Before addition,\npol1 = {pol1}\npol2 = {pol2}\n')
  print(f'pol1 + pol2 = {pol1 + pol2}\n')
  print(f'After addition,\npol1 = {pol1}\npol2 = {pol2}')

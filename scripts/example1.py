## Formula Deriviation - Moment
from ajdmom import mdl_1fsv
from pprint import pprint

m1 = mdl_1fsv.moment_y(1)
msg = f"which is a Poly with attribute keyfor = \n{m1.keyfor}"
print("moment_y(1) = ")
pprint(m1)
print(msg)

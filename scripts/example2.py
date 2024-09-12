## Formula Deriviation - Covariance
from ajdmom import mdl_1fsv
from pprint import pprint

cov21 = mdl_1fsv.cov_yy(2, 1)
msg = f"which is a Poly with attribute keyfor =\n{cov21.keyfor}"
print("cov_yy(2,1) = \n")
pprint(cov21)
print(msg)

## Moments and Central Moments
from ajdmom.mdl_1fsv.mom import m, dm  # for moments
from ajdmom.mdl_1fsv.cmom import cm, dcm  # for central moments

parameters = {'mu': 0.125, 'k': 0.1, 'theta': 0.25,
              'sigma_v': 0.1, 'rho': -0.7, 'h': 1}

# 3rd moment as an example
moment = m(l=3, par=parameters)  # cm: central moment
# partial derivative w.r.t. parameter 'k'
dmoment = dm(l=3, par=parameters, wrt='k')  # dcm: central moment

## Covariance
from ajdmom.mdl_1fsv.cov import cov, dcov

parameters = {'mu': 0.125, 'k': 0.1, 'theta': 0.25,
              'sigma_v': 0.1, 'rho': -0.7, 'h': 1}

# covariance cov(y_n^2, y_{n+1}^2) as an example
covariance = cov(l1=2, l2=2, par=parameters)
# partial derivative w.r.t. parameter 'k'
dcovariance = dcov(l1=2, l2=2, par=parameters, wrt='k')

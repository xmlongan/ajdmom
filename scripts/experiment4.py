import numpy as np
import math

from ajdmom.mdl_svcj.euler import rSVCJ
from ajdmom.mdl_svcj.cond2_mom import m
from experiment3 import comp_moments, print_comp

rng = np.random.default_rng()


v0, h = 0.007569, 1
k, theta, sigma, rho = 3.46, 0.008, 0.14, -0.82
mu = 0.0319
lmbd_v, mu_v = 0.47, 0.05
mu_b = -0.1
lmbd_s, sigma_s, rhoJ = lmbd_v, 0.0001, -0.38
#
mu_s = math.log((1 + mu_b) * (1 - rhoJ * mu_v)) - sigma_s ** 2 / 2
# same as that in SVJ in Broadie-Kaya (2006)
par = {'v0': v0, 'h': h, 'mu': mu, 'k': k, 'theta': theta,
       'sigma': sigma, 'rho': rho, 'lmbd_v': lmbd_v, 'mu_v': mu_v,
       'lmbd_s': lmbd_s, 'mu_s': mu_s, 'sigma_s': sigma_s, 'rhoJ': rhoJ}

for numJ in [2]:  # numJ = rng.poisson(lmbd * h): the actual number of jumps
    jsize_v = rng.exponential(mu_v, numJ)
    jtime_v = rng.uniform(0, h, numJ)  # unsorted
    jsize_v, jtime_v = tuple(jsize_v), tuple(sorted(jtime_v))  # set as tuples
    par['jumpsize'] = jsize_v
    par['jumptime'] = jtime_v
    #
    print(f"\nWith conditions v0 = {v0}, \njumpsize = {jsize_v}, \njumptime = {jtime_v}")
    #
    n, n_segment = 4000 * 1000, 100
    y = rSVCJ(v0, mu, k, theta, sigma, rho, h, jsize_v, jtime_v,
              mu_s, sigma_s, rhoJ, n, n_segment)
    orders = [1, 2, 3, 4, 5]
    mom_comp = comp_moments(orders, par, y, m)
    print(f"------------------------------moment_ie------------------------------")
    print_comp(mom_comp)

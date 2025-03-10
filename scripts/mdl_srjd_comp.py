import math
import numpy as np

from ajdmom.mdl_srjd.euler import rSRJD
from ajdmom.mdl_srjd.cond2_mom import m
from ajdmom.mdl_srjd.cond2_cmom import cm

rng = np.random.default_rng()


def comp_moments(orders, par, y_series):
    mnames = []
    tms = []
    sms = []
    for n in orders:
        tm = m(n, par)  # theoretial moment calculated by the package
        sm = np.mean(y_series ** n)  # sample moment
        mnames.append(f'moment({n})')
        tms.append(tm)
        sms.append(sm)
    return ((mnames, tms, sms))


def comp_cmoments(orders, par, y_series):
    mnames = []
    tms = []
    sms = []
    diff = y_series - np.mean(y_series)
    for n in orders:
        tm = cm(n, par)  # theoretial cmoment calculated by the package
        sm = np.mean(diff ** n)  # sample cmoment
        mnames.append(f'cmoment({n})')
        tms.append(tm)
        sms.append(sm)
    return ((mnames, tms, sms))


def print_comp(comp):
    txt = ''.ljust(10)
    txt += 'derived moment'.rjust(15) + 'sample moment'.rjust(15)
    txt += 'diff'.rjust(12) + 'diff(in %)'.rjust(12) + '\n'
    #
    for i in range(len(comp[0])):
        name = comp[0][i]
        tm = comp[1][i]
        sm = comp[2][i]
        diff = abs(tm - sm)
        if tm == 0:
            diff_percent = math.nan
        else:
            diff_percent = diff / abs(tm)
        #
        txt += '{:<10}{:>15.4f}{:>15.4f}{:>12.4f}{:>12.0%}\n'.format(name,
                                                                     tm, sm, diff, diff_percent)
    print(txt)


# h should be in par, since it is needed for m(), cm()
v0, h = 0.007569, 1
k, theta, sigma = 3.46, 0.008, 0.14
mu_v, lmbd = 0.05, 0.47
par = {'v0': v0, 'h': h, 'k': k, 'theta': theta, 'sigma': sigma, 'mu_v': mu_v, 'lmbd': lmbd}
#
for numJ in [0, 1, 2, 3]:  # numJ = rng.poisson(lmbd * h): the actual number of jumps
    jumpsize = rng.exponential(mu_v, numJ)
    jumptime = rng.uniform(0, h, numJ)  # unsorted
    jumpsize, jumptime = tuple(jumpsize), tuple(sorted(jumptime))  # set as tuples
    par['jumpsize'] = jumpsize
    par['jumptime'] = jumptime
    #
    x = rSRJD(v0, k, theta, sigma, h, jumpsize, jumptime, 100 * 1000)
    #
    orders = [1, 2, 3, 4, 5]
    mom_comp = comp_moments(orders, par, x)
    print_comp(mom_comp)
    #
    orders = [1, 2, 3, 4, 5]
    cmom_comp = comp_cmoments(orders, par, x)
    print_comp(cmom_comp)

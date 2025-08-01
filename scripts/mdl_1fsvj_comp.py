import numpy as np
import math

from ajdmom.mdl_1fsvj.euler import r1FSVJ
from ajdmom.mdl_1fsvj.mom import m
from ajdmom.mdl_1fsvj.cmom import cm
from ajdmom.mdl_1fsvj.cov import cov


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


def comp_cov(order_pairs, par, y_series):
    mnames = []
    tms = []
    sms = []
    N = len(y_series)
    pre = y_series[0:(N - 1)]
    nxt = y_series[1:N]
    for nm in order_pairs:
        n, m = nm
        tm = cov(n, m, par)  # theoretial cov calculated by the package
        sm = np.cov(pre ** n, nxt ** m)[0][1]  # sample cov
        mnames.append(f'cov({n},{m})')
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


# h should be in par, since it is needed for m(), cm(), cov()
par = {'mu': 0.125, 'k': 0.1, 'theta': 0.25, 'sigma_v': 0.1, 'rho': -0.7,
       'lambda': 0.01, 'mu_j': 0, 'sigma_j': 0.05, 'h': 1}
v_0 = par['theta']
y_series = r1FSVJ(v_0, par, N=4000 * 1000, n_segment=10, h=par['h'])

orders = [1, 2, 3, 4, 5]
mom_comp = comp_moments(orders, par, y_series)
print_comp(mom_comp)

orders = [1, 2, 3, 4, 5]
cmom_comp = comp_cmoments(orders, par, y_series)
print_comp(cmom_comp)

orders = [(1, 1), (2, 1), (1, 2), (3, 1), (2, 2), (1, 3), (4, 1), (3, 2), (2, 3), (1, 4)]
cov_comp = comp_cov(orders, par, y_series)
print_comp(cov_comp)

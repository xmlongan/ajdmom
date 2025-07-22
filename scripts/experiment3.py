import math
import numpy as np

from ajdmom.mdl_svcj.cond_mom import m_y
from ajdmom.mdl_svcj.euler import r_y


def comp_moments(orders, par, x, m):
    x = np.array(x)
    mnames, tms, sms = [], [], []
    for n in orders:
        tm = m(n, par)  # theoretical moment calculated by the package
        sm = np.mean(x ** n)  # sample moment
        mnames.append(f'moment({n})')
        tms.append(tm)
        sms.append(sm)
    return mnames, tms, sms


def print_comp(comp):
    txt = ''.ljust(10)
    txt += 'derived moment'.rjust(15) + 'sample moment'.rjust(15)
    txt += 'diff'.rjust(12) + 'diff(in %)'.rjust(12) + '\n'
    for i in range(len(comp[0])):
        name = comp[0][i]
        tm = comp[1][i]
        sm = comp[2][i]
        diff = abs(tm - sm)
        diff_percent = math.nan if tm == 0 else diff / abs(tm)
        temp = '{:<10}{:>15.7f}{:>15.7f}{:>12.7f}{:>12.0%}\n'
        txt += temp.format(name, tm, sm, diff, diff_percent)
    print(txt[0:-1])


h = 1
v0, k, theta, sigma, rho = 0.007569, 3.46, 0.008, 0.14, -0.82
lmbd, mu_b, sigma_s, mu_v, rhoJ = 0.47, -0.1, 0.0001, 0.05, -0.38
r = 0.0319

mu_s = math.log((1+mu_b)*(1-rhoJ*mu_v)) - sigma_s ** 2 / 2
mu = r - lmbd * mu_b

N = 4000 * 1000

par = {'v0': v0, 'mu': mu, 'k': k, 'theta': theta, 'sigma': sigma, 'rho': rho,
       'lmbd': lmbd, 'mu_v': mu_v, 'rhoJ': rhoJ, 'mu_s': mu_s, 'sigma_s': sigma_s,
       'h': h}

print(f'mu = {mu}, mu_s = {mu_s}')

nsegment = 100
y_series = r_y(v0, mu, k, theta, sigma, rho, lmbd, mu_v, rhoJ, mu_s, sigma_s, h, N, nsegment)
orders = [1, 2, 3, 4, 5]
mom_comp = comp_moments(orders, par, y_series, m_y)
print(f"------------------------------moment_ie------------------------------")
print_comp(mom_comp)

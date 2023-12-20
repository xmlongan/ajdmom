import numpy as np

from ajdmom.mdl_1fsvj.euler import r1FSVJ
from ajdmom.mdl_1fsvj.mom import m

def comp_moments(orders, par, y_series):
  mnames = []; tms = []; sms = []
  for n in orders:
    tm = m(n, par)               # theoretial moment calculated by the package
    sm = np.mean(y_series**n)    # sample moment
    mnames.append(f'moment({n})')
    tms.append(tm)
    sms.append(sm)
  return((mnames,tms,sms))

def print_comp(comp):
  txt = ''.ljust(10) 
  txt += 'derived moment'.rjust(15) + 'sample moment'.rjust(15) 
  txt += 'diff'.rjust(12) + 'diff(in %)'.rjust(12) + '\n'
  # 
  for i in range(len(comp[0])):
    name = comp[0][i]; tm = comp[1][i]; sm = comp[2][i]
    diff = abs(tm-sm)
    if tm == 0:
      diff_percent = math.nan
    else:
      diff_percent = diff/abs(tm)
    # 
    txt += '{:<10}{:>15.4f}{:>15.4f}{:>12.4f}{:>12.0%}\n'.format(name, 
      tm, sm, diff, diff_percent)
  print(txt)

# h should be in par, since it is needed for m(), cm(), cov()
par = {'mu':0.125,'k':0.1,'theta':0.25,'sigma_v':0.1,'rho':-0.7,
    'lambda':0.01, 'mu_j':0, 'sigma_j':0.05, 'h':1}
v_0 = par['theta'] 
y_series = r1FSVJ(v_0, par, N=4000*1000, n_segment=10, h=par['h'])

orders = [1,2,3,4,5]
mom_comp = comp_moments(orders, par, y_series)
print_comp(mom_comp)

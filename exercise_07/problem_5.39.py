#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

import numpy as np
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.size'] = 16
import matplotlib.pyplot as plt

"""
Draw the Phase diagram of Al2SiO5.
Phase boundary lines intersect at:
    K-A: T = 430K
    K-S: T = 535K
    S-A: T = 878K
"""

T0 = 300.
T1 = 1100.
T_KA = 430.
T_KS = 535.
T_SA = 878.
T_rg_KA = np.arange(T0, 690.)
T_rg_KS = np.arange(690., T1)
T_rg_SA = np.arange(690., T1)
# Slope of the phase boundaries, in mPa / K
PT_slope_KA = 1.26
PT_slope_KS = 2.11
PT_slope_SA = -1.77

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(T_rg_KA, PT_slope_KA * (T_rg_KA - T_KA), label='K-A')
ax.plot(T_rg_KS, PT_slope_KS * (T_rg_KS - T_KS), label='K-S')
ax.plot(T_rg_SA, PT_slope_SA * (T_rg_SA - T_SA), label='S-A')

ax.text(600, 100, 'Kyanite', fontsize=16, color='blue')
ax.text(500, 400, 'Andalusite', fontsize=16, color='green')
ax.text(800, 400, 'Sillimanite', fontsize=16, color='red')

ax.legend(loc='upper left')
ax.set_xlabel('T / K')
ax.set_ylabel('P / 10^6 Pa')
ax.set_ylim((0.0, 7.0e2))

plt.show()

#End of file problem_5.39.py

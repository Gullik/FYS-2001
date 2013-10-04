#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

import numpy as np
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.size'] = 16
import matplotlib.pyplot as plt

"""
Plot the difference in Gibbs free energy for
all phases of Al2SiO5
Kyanite
Andalusite
Sillimanite
"""

T0 = 300.
Tf_rg = np.arange(T0, 1000., 1.)

#Entropy at room temperature
S0_K = 83.81
S0_A = 93.22
S0_S = 96.11
# Gibbs free energy at room temperature
G0_K = -2.44388e6
G0_A = -2.44266e6
G0_S = -2.44099e6

deltaG_KA = G0_K - G0_A
deltaG_KS = G0_K - G0_S
deltaG_SA = G0_S - G0_A

deltaS_KA = S0_K - S0_A
deltaS_KS = S0_K - S0_S
deltaS_SA = S0_S - S0_A


deltaG_KA_T = deltaG_KA - deltaS_KA * (Tf_rg - T0)
deltaG_KS_T = deltaG_KS - deltaS_KS * (Tf_rg - T0)
deltaG_SA_T = deltaG_SA - deltaS_SA * (Tf_rg - T0)

# Rest of the code is plotting only
min_G = np.min(np.array([deltaG_KA_T.min(), deltaG_KS_T.min(),
                         deltaG_SA_T.min()]))
max_G = np.max(np.array([deltaG_KA_T.max(), deltaG_KS_T.max(),
                         deltaG_SA_T.max()]))

# Find the point where the slopes intersect zero and the slope of the lines
slope_max = (max_G - min_G) / (Tf_rg[-1] - Tf_rg[0])

min_KA_idx = np.abs(deltaG_KA_T).argmin()
Tf_min_KA = Tf_rg[min_KA_idx]
slope_KA = ((deltaG_KA_T.max() - deltaG_KA_T.min())
            / (Tf_rg[-1] - Tf_rg[0])) / slope_max
ang_KA = np.degrees(np.arctan([slope_KA * 0.3125 * np.pi]))[0]
print 'min_KA = %f, T = %f, slope = %f deg' % (deltaG_KA_T[min_KA_idx],
                                               Tf_min_KA, ang_KA)

min_KS_idx = np.abs(deltaG_KS_T).argmin()
Tf_min_KS = Tf_rg[min_KS_idx]
slope_KS = ((deltaG_KS_T.max() - deltaG_KS_T.min())
            / (Tf_rg[-1] - Tf_rg[0])) / slope_max
ang_KS = np.degrees(np.arctan([slope_KS * 0.3125 * np.pi]))[0]
print 'min_KS = %f, T = %f, slope = %f deg' % (deltaG_KS_T[min_KS_idx],
                                               Tf_min_KS, ang_KS)

min_SA_idx = np.abs(deltaG_SA_T).argmin()
Tf_min_SA = Tf_rg[min_SA_idx]
slope_SA = ((deltaG_SA_T.max() - deltaG_SA_T.min())
            / (Tf_rg[-1] - Tf_rg[0])) / slope_max
ang_SA = np.degrees(np.arctan([slope_SA * 0.3125 * np.pi]))[0]
print 'min_SA = %f, T = %f, slope = %f deg' % (deltaG_SA_T[min_SA_idx],
                                               Tf_min_SA, ang_SA)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
plt.plot(Tf_rg, deltaG_KA_T, label='Kyanite - Andalusite')
plt.plot(Tf_rg, deltaG_KS_T, label='Kyanite - Sillimanite')
plt.plot(Tf_rg, deltaG_SA_T, label='Sillimanite - Andalusite')
plt.plot(Tf_rg, np.zeros_like(Tf_rg), 'k--', label='Phase change')

plt.plot([Tf_min_KA, Tf_min_KA], [min_G, max_G], 'k--')
plt.plot([Tf_min_KS, Tf_min_KS], [min_G, max_G], 'k--')
plt.plot([Tf_min_SA, Tf_min_SA], [min_G, max_G], 'k--')

# Annotate stabel phases
plt.text(300, 3500, 'Kyanite stable')
plt.text(500, 3500, 'Andalusite stable')
plt.text(900, 3000, """Sillimanite\nstable""")

# Find index of Tf_Rg closest to 300.
T_list = [320., 440., 650., 900.]
Tidx_list = [np.argmin(np.abs(Tf_rg - T)) for T in T_list]

#aaaand plot the line labels
plt.text(Tf_rg[Tidx_list[0]], deltaG_KA_T[Tidx_list[0]] + 800.,
         '$\\Delta G_K < \\Delta G_A$', fontsize=16,
         color='blue', rotation=ang_KA)
plt.text(Tf_rg[Tidx_list[0]], deltaG_KS_T[Tidx_list[0]] + 820.,
         '$\\Delta G_K < \\Delta G_S$', fontsize=16,
         color='green', rotation=ang_KS)
plt.text(Tf_rg[Tidx_list[0]], deltaG_SA_T[Tidx_list[0]],
         '$\\Delta G_A < \\Delta G_S$', fontsize=16,
         color='red', rotation=-ang_SA)

plt.text(Tf_rg[Tidx_list[1]], deltaG_KA_T[Tidx_list[1]] + 800.,
         '$\\Delta G_K > \\Delta G_A$', fontsize=16,
         color='blue', rotation=ang_KA)
plt.text(Tf_rg[Tidx_list[1]], deltaG_KS_T[Tidx_list[1]] + 820.,
         '$\\Delta G_K < \\Delta G_S$', fontsize=16,
         color='green', rotation=ang_KS)
plt.text(Tf_rg[Tidx_list[1]], deltaG_SA_T[Tidx_list[1]],
         '$\\Delta G_A < \\Delta G_S$', fontsize=16,
         color='red', rotation=-ang_SA)

plt.text(Tf_rg[Tidx_list[2]], deltaG_KA_T[Tidx_list[2]] + 700.,
         '$\\Delta G_K > \\Delta G_A$', fontsize=16,
         color='blue', rotation=ang_KA)
plt.text(Tf_rg[Tidx_list[2]], deltaG_KS_T[Tidx_list[2]] + 820.,
         '$\\Delta G_K > \\Delta G_S$', fontsize=16,
         color='green', rotation=ang_KS)
plt.text(Tf_rg[Tidx_list[2]], deltaG_SA_T[Tidx_list[2]],
         '$\\Delta G_A < \\Delta G_S$', fontsize=16,
         color='red', rotation=-ang_SA)

plt.text(Tf_rg[Tidx_list[3]], deltaG_KA_T[Tidx_list[3]] + 150.,
         '$\\Delta G_K > \\Delta G_A$', fontsize=16,
         color='blue', rotation=ang_KA)
plt.text(Tf_rg[Tidx_list[3]], deltaG_KS_T[Tidx_list[3]] + 820.,
         '$\\Delta G_K > \\Delta G_S$', fontsize=16,
         color='green', rotation=ang_KS)
plt.text(Tf_rg[Tidx_list[3]], deltaG_SA_T[Tidx_list[3]] - 300.,
         '$\\Delta G_A > \\Delta G_S$', fontsize=16,
         color='red', rotation=-ang_SA)

plt.ylabel('Gibbs free energy / kJ')
plt.xlabel('Temperature / K')
plt.legend(loc='upper left')
plt.show()


#End of file problem_5.29.py

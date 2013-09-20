#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['lines.linewidth'] = 2
import matplotlib.pyplot as plt

"""
Study the entropy, temperature and heat capacity of an Einstein solid
"""

# The multiplicity function for an Einstein solid
S_Einstein = lambda q, N: (q + N) * np.log(q + N) - q * np.log(q) - N * np.log(N)

# Number of oscillators
N_1 = 50
N_2 = 5000
k = 1.
q_rg = np.arange(0., 101., 1.)

# Entropy
S_1 = np.zeros_like(q_rg)
S_2 = np.zeros_like(q_rg)

S_1[0] = 0
S_1[1:] = S_Einstein(q_rg[1:], N_1)
S_2[0] = 0
S_2[1:] = S_Einstein(q_rg[1:], N_2)

# Multiplicity
Omega_1 = np.exp(S_1)
Omega_2 = np.exp(S_2)

# Temperature
T_1 = np.zeros_like(q_rg)
T_2 = np.zeros_like(q_rg)

# Use second order centered difference scheme:
# 1/T = \frac{\partial S}{\partial U}
# use S -> S' = S/k, T -> T' = k T / \epsilon and U = q \epsilon, see problem
# 3.7
# => T_i \simeq \frac{2}{S_{i+1} - S_{i-1}}
T_1[1:-1] = 2. / (S_1[2:] - S_1[:-2])
T_2[1:-1] = 2. / (S_2[2:] - S_2[:-2])

# Heat capacity
C_1 = np.zeros_like(q_rg)
C_2 = np.zeros_like(q_rg)
# Second order centered difference scheme for C_V:
# C_V = \frac{\partial U}{\partial T}
# => C_V = k \frac{q_{i+1} - q_{i-1}}{T'_{i+1} - T'_{i-1}}
C_1[1:-1] = 2. / (T_1[2:] - T_1[:-2])
C_2[1:-1] = 2. / (T_2[2:] - T_2[:-2])
# Normalize to particle number
C_1 = C_1 / N_1
C_2 = C_2 / N_2


# Plot the stuff
fig_all = plt.figure(figsize=(12, 12))
ax_omega = fig_all.add_subplot(221)
ax_omega.semilogy(q_rg, Omega_1, label='N = 50')
ax_omega.semilogy(q_rg, Omega_2, label='N = 5000')
ax_omega.set_ylabel('$\\Omega$')
ax_omega.set_xlabel('$U / \\epsilon$')
ax_omega.legend(loc='upper left')

ax_S = fig_all.add_subplot(222)
ax_S.plot(q_rg[1:-1], S_1[1:-1])
ax_S.plot(q_rg[1:-1], S_2[1:-1])
ax_S.set_ylabel('$S / k$')
ax_S.set_xlabel('$U / \\epsilon$')

ax_T = fig_all.add_subplot(223, sharex=ax_omega)
ax_T.plot(q_rg[1:-1], T_1[1:-1])
ax_T.plot(q_rg[1:-1], T_2[1:-1])
ax_T.set_ylabel('$\\frac{k T}{\\epsilon}$')
ax_T.set_xlabel('$U / \\epsilon$')

ax_C1 = fig_all.add_subplot(224)
ax_C2 = ax_C1.twinx()
ax_C1.plot(T_1[2:-2], C_1[2:-2], 'b-')
ax_C2.plot(T_2[2:-2], C_2[2:-2], 'g-')
ax_C1.set_ylabel('$\\frac{C}{N k}, N = 50$')
ax_C2.set_ylabel('$\\frac{C}{N k}, N = 5000$')
ax_C1.set_xlabel('$\\frac{T}{\epsilon k}$')

fig_all.text(0.5, 0.95, 'Einstein solid', ha='center')

# Qualitatively, the plot C/Nk vs kT/epsilon for N=50
# agrees with the plot for lead from table 1.14 while the plot for N = 5000
# agrees with the plot for diamond from table 1.14


# print the table
print 'N = 50'
print 'q\t Omega\t\t S/k\t\t kT/epsilon\t C/Nk\n'
for idx in np.arange(q_rg.size):
    print '%d\t%e\t%f\t%f\t%f' % (q_rg[idx], Omega_1[idx], S_1[idx],
                                  T_1[idx], C_1[idx])

print 'N = 5000'
print 'q\t Omega\t\t S/k\t\t kT/epsilon\t C/Nk\n'
for idx in np.arange(q_rg.size):
    print '%d\t%e\t%f\t%f\t%f' % (q_rg[idx], Omega_2[idx], S_2[idx],
                                  T_2[idx], C_2[idx])

plt.show()
# End of file problem_3.24.py

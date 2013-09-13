#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

# Import numpy for logarithm function
import numpy as np
# Import matplolib and define nice fontsizes and line widths for plotting
import matplotlib as mpl
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['lines.linewidth'] = 2
# Import practical plot functions
import matplotlib.pyplot as plt


"""
Compute and plot the multiplicity of microstates for
an Einstein solid in the approximation q << N:
    Omega(q,N) = (N/q)^q * exp(q)

"""

# Expression for the logarithm of microstate multiplicity
# This is called like a function, but the lambda definition is shorter
S_over_k = lambda q, N: (q + N) * np.log(q + N) - q * np.log(q) - N * np.log(N)

# Number of oscillators in each system
N_A = 300
N_B = 200
# Define energy ranges for system A and B
q_A = np.arange(1., 101., 1.)
q_B = np.arange(101., 1., -1)

# Compute multiplicity of each microstate for A and B
S_over_k_A = S_over_k(q_A, N_A)
S_over_k_B = S_over_k(q_B, N_B)

# Compute the temperature with a centered difference scheme
T_A = 1. / (S_over_k_A[2:] - S_over_k_A[:-2])
T_B = -1. / (S_over_k_B[2:] - S_over_k_B[:-2])

# Plot the multiplicity of microstates for A, B and A+B
# Create a figure
fig_S = plt.figure()
# Add a canvas to the figure which we plot into
ax_S = fig_S.add_subplot(111)
# Plot into the canvas created in the line above
ax_S.plot(q_A, S_over_k_A, label='System A')
ax_S.plot(q_A, S_over_k_B, label='System B')
ax_S.plot(q_A, S_over_k_A + S_over_k_B, label='Combined')

# Make a title string
ax_S.set_title('Entropy of two interacting Einstein Solids')
# Show a x label in ax_S
ax_S.set_xlabel('$q_A$')
# Show a y label in ax_S
ax_S.set_ylabel('$S / k$')
# And make the legend appear
ax_S.legend(loc='upper left')

# Plot the temperature of A, B, A+B
fig_T = plt.figure()
ax_T = fig_T.add_subplot(111)
ax_T.plot(q_A[1:-1], T_A, label='System A')
ax_T.plot(q_A[1:-1], T_B, label='System B')

ax_T.set_title('Temperature of two interacting Einstein Solids')
ax_T.set_xlabel('$q_A')
ax_T.set_ylabel('$T / \\epsilon / k$')
ax_T.legend(loc='upper left')

plt.show()
# End of file table 3.1

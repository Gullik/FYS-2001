#/usr/bin/env python
#-*- Encoding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt

"""
Problem 1.17
Solve the linear system

B(T) = ((a/RT) + b
C = b

for the coefficients a,b in the Van der Waals equation of state.
Given is a list of B(T), see p.9.

Idea: Introduce T' = 1/T and do a linear fit of (a/R) * T' + b on B(T')

"""
# Tabulated values B(T)
B = np.array([-160., -35., -4.2, 9.0, 16.9, 21.3])
# Original temperatures
T = np.array([100., 200., 300., 400., 500., 600.])
# Inverse temperatures
Tmark = np.array([1. / 100., 1. / 200., 1. / 300., 1. / 400.,
                  1. / 500., 1. / 600.])

# Physical constants
# Ideal gas constants, R = 8.31 Joule / mole K
R = 8.31

# Solve the system B(T') = (a/R) * T' + b for (a/R) and b
# Sample code for linear least-squares fitting in python
# is in the numpy documentation:
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html

A = np.vstack([Tmark, np.ones(Tmark.size)]).T
a_over_R, b = np.linalg.lstsq(A, B)[0]
print 'a/R = %f, b = %f' % (a_over_R, b)

# fine-grained temperature array used for plotting the
# fitted coefficient function
T_fine = np.arange(T.min(), T.max(), 1.)

# Plot tabulated values for B(T') and the linear fit
# Create a figure with an axis object
fig_B = plt.figure()
ax_B = fig_B.add_subplot(111)
# Plot tabulated value and the fit
ax_B.plot(Tmark, B, 'k', lw=3, label='Virial eq. of state')
ax_B.plot(Tmark, a_over_R * Tmark + b, 'ro', label='Fit: VdW eq. of state')
# Add axis labels
ax_B.set_xlabel('Temperature^-1 / K^-1')
ax_B.set_ylabel('First order correction')
ax_B.set_xlim((0.95 * Tmark.min(), 1.01 * Tmark.max()))
# Draw the legend
ax_B.legend(loc='upper right')


fig_orig = plt.figure()
ax_o = fig_orig.add_subplot(111)
ax_o.plot(T, B, label='Virial eq. of state')
ax_o.plot(T, a_over_R*Tmark + b, 'ro', label='Fit: VdW eq. of state')
# Add axis labels
ax_o.set_xlabel('Temperature / K')
ax_o.set_ylabel('First order correction')
ax_o.set_xlim((0.95 * T.min(), 1.01 * T.max()))
# Draw the legend
ax_o.legend(loc='lower right')


plt.show()
# End of file proble_1.17.py

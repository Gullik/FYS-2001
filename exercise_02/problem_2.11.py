#!/usr/bin/env python
#-*- Encoding: UTF-8 -*

import numpy as np
from scipy.misc import comb
import matplotlib.pyplot as plt

"""
Problem 2.11

We are given two systems of two-state dipole moments, A, B, in an
external magnetic field B0.
Both have N_A = N_B = 100 particles.
The energy of each dipole is 0 if it is parallel to B0 and 1 if
antiparallel to B0.
The total energy of the system is 80 energy units, distributed
among both systems.

We are asked to compute the probability to find the system S = A + B
in any given macrostate.
"""


def get_mant_exp(a):
    """
    Given a floating point number a,
    return its mantissa and exponent.
    """
    exponent = np.floor(np.log10(a))
    mantissa = a * (10 ** -exponent)
    return (mantissa, exponent)


N = 100.
q_total = 80.
N_rg = np.ones(q_total + 1.) * N
table_fname = 'table_states.tex'


# Energy units in A
q_A = np.arange(0., q_total + 1., 1.)
# Energy units in B
q_B = np.arange(q_total, -1., -1.)
# q_A + q_B should be a vector with 80 in each entry

# compute the multiplicity of macrostates for each q_A, q_B
Omega_A = comb(N_rg, q_A)
Omega_B = comb(N_rg, q_B)
Omega_AB = Omega_A * Omega_B


# Produce a nice latex table
with open(table_fname, 'w') as tf:

    # Format each result as to fit in the table
    for qa, Oa, qb, Ob, Oab in zip(q_A, Omega_A, q_B, Omega_B, Omega_AB):
        mant_Oa, exp_Oa = get_mant_exp(Oa)
        mant_Ob, exp_Ob = get_mant_exp(Ob)
        mant_Oab, exp_Oab = get_mant_exp(Oab)
        str1 = "$%2d$\t& $%3.1f \\times 10^{%2d}$\t& " % (qa, mant_Oa, exp_Oa)
        str2 = "$%2d$\t& $%3.1f \\times 10^{%2d}$\t&" % (qb, mant_Ob, exp_Ob)
        str3 = "$%3.1f \\times 10^{%2d}$" % (mant_Oab, exp_Oab)
        str4 = "\\\\ \hline\n"
        print str1 + str2 + str3 + str4
        tf.write(str1 + str2 + str3 + str4)
    #[tf.write("%2d & %

plt.figure()
plt.plot(q_A, Omega_A, label='$\\Omega_A$')
plt.plot(q_A, Omega_B, label='$\\Omega_B$')
plt.plot(q_A, Omega_AB, label='$\\Omega_A \\Omega_B$')
plt.xlabel('$q_A$')
plt.legend(loc='upper left')
#plt.show()
# End of file problem_2.11.py

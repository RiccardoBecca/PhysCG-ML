# PhysCG-ML
# Physics-Informed Machine Learning Coarse-Grained Potential

This repository develops a **physics-informed machine learning potential for coarse-grained molecular dynamics (CGMD)**.

The approach combines a physically motivated analytical interaction model with machine-learned parameters. The functional form of the intermolecular potential is fixed according to the relevant physical interactions, while the parameters of this function are learned from data.

## Physics-Informed ML Potential

The intermolecular interaction between two coarse-grained beads is defined as

[
U(r) = U_{\mathrm{TT}}(r) + U_{\mathrm{DH}}(r),
]

with a Tang–Toennies-type van der Waals term

[
U_{\mathrm{TT}}(r)
==================

## \left(\frac{C_{\mathrm{rep}}}{r}\right)^{10}

f_6(br)\left(\frac{C_{\mathrm{att}}}{r}\right)^6,
]

where

[
f_6(x)
======

1-e^{-x}
\left(
1+x+\frac{x^2}{2}
+\frac{x^3}{6}
+\frac{x^4}{24}
+\frac{x^5}{120}
+\frac{x^6}{720}
\right),
]

and a Debye–Hückel electrostatic term

[
U_{\mathrm{DH}}(r)
==================

C_{\mathrm{mix}}
\frac{138.935456,q_1q_2}{\epsilon_r r}
e^{-\kappa r}.
]

The distinguishing feature of the model is that the parameters

[
C_{\mathrm{rep}},\quad C_{\mathrm{att}},\quad b,\quad C_{\mathrm{mix}}
]

are not represented by scalar values. Each is represented by a **multi-component parameter vector**, and the interaction parameters between two beads are obtained through component-wise mixing rules. For two beads (i) and (j),

[
C_{\mathrm{rep}}^{ij}
=====================

\sqrt{
\frac{1}{3}
\sum_k
C_{\mathrm{rep},k}^{(i)}
C_{\mathrm{rep},k}^{(j)}
},
]

[
b^{ij}
======

\sqrt{
\frac{1}{3}
\sum_k
b_k^{(i)}b_k^{(j)}
},
]

[
C_{\mathrm{att}}^{ij}
=====================

\sqrt{
\frac{1}{3}
\sum_k
C_{\mathrm{att},k}^{(i)}
C_{\mathrm{att},k}^{(j)}
},
]

and

[
C_{\mathrm{mix}}^{ij}
=====================

\sqrt{
\frac{1}{3}
\sum_k
C_k^{(i)}C_k^{(j)}
}.
]

Thus, the physical functional form of the potential is explicitly constrained, while machine learning determines the underlying multi-component parameters. This provides a physics-informed alternative to learning an unconstrained black-box energy function.

## Intramolecular Interactions

Intramolecular interactions are derived from structural ensembles using **Boltzmann inversion**. The required distributions are obtained from molecular configurations deposited in the input PDB data and converted into effective coarse-grained intramolecular potentials.

## Coarse-Grained Molecular Dynamics

The resulting intermolecular and intramolecular potentials are used directly for coarse-grained molecular dynamics simulations. The generated trajectories can subsequently be analysed to assess structural stability, folding behaviour, and conformational dynamics.

Detailed information on the implementation, available scripts, and analysis procedures is provided in the `src/` documentation.

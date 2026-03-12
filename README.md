# Molecular Dynamics

This project simulates Molecular Dynamics for a Gd crystal with 900 atoms.
Verlet velocity equations are used to solve for velocity (momentum) and position. 
Langevines Dynamics is used in addition to add a stochastic ''kick'' to the system. 
This creates the 5 step process modelled in this project.
The auto-correlation $C(t)$ is solved for along with the integral $\tau = \int_0^\infty C(t) dt$ 
for the correlation time. Thi is the used in order obtain the sampling frequency for the next task.
That being estimating the Mean-Squared Displacement for a series of different temperatures ranging from 
$100K \to 1000k$. Finally the lattice parameter $a$ was examined. The total energy of the system was recorded for a series of 
lattice parameters ranging from $0.9\sigma \to 1.3\sigma$ 
# Molecular Dynamics

This project simulates Molecular Dynamics for a Gd crystal with 900 atoms.
Verlet velocity equations are used to solve for velocity (momentum) and position. 
Langevines Dynamics is used in addition to add a stochastic ''kick'' to the system. 
This creates the 5 step process modelled in this project.
The auto-correlation $C(t)$ is solved for along with the integral $\tau = \int_0^\infty C(t) dt$ 
for the correlation time. Thi is the used in order obtain the sampling frequency for the next task.
That being estimating the Mean-Squared Displacement for a series of different temperatures ranging from 
$100^\circ K \to 1000^\circ K$. Finally the lattice parameter $a$ was examined. The total energy of the system was recorded for a series of 
lattice parameters ranging from $0.9\sigma \to 1.3\sigma$. This was done at three fixed temperatures.

## Pipeline
To run the code properly, start by installing `requirements.txt` by 

```bash
pip install -r requirements.txt
```
In the your virtual environment.

The files `sim.py` and `sim_fast.py` perform the exact same task. however, `sim_fast.py` utilizes a precomputed 
neighbor list for all atoms, this significantly reduces compute time and yields similar results as `sim.py`, therefore `sim_fast.py` is used in all other files in the project. ´sim_fast.py´ is also the point of entry in the project and performs the first task by calculating the auto correlation and the correlation time. Run it by
```bash
python /src/sim_fast.py --y 0.01 --io
```
`--io` saves the results and ´--y´ specifies the damping parameter $\gamma$. Additionally ´--plot´ can be used to plot the results directly and ´--T´ can be used to specify the temperature which has a default value of $300^\circ K$.

To evaluate the Mean-Squared Displacement run `MSD.py`. Note, this code essentially does the same code as `sim_fast.py` over an array of different temperatures, specified by `--points`, thus scaling up the compute time with numner of temperature points.
```bash
python src/MSD.py --points 7 --plot --io

```
Runs calculates the MSD for 7 temperatures equally divided across the range $100^\circ K \to 1000^\circ K$, showing the plot and savingf the results to JSON.


Expermiments were conducted by running all three major tasks (varying $\gamma$, varying $T$, and varying $a$) a number of times and the averaging the results.
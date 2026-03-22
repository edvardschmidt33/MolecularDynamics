# Molecular Dynamics
 
This project simulates Molecular Dynamics for a Gd crystal with 900 atoms.
Verlet velocity equations are used to solve for velocity (momentum) and position.
Langevin Dynamics is used in addition to add a stochastic "kick" to the system.
This creates the 5-step process modelled in this project.
The auto-correlation $C(t)$ is solved for along with the integral $\tau = \int_0^\infty C(t) dt$
for the correlation time. This is then used in order to obtain the sampling frequency for the next task.
That being estimating the Mean-Squared Displacement for a series of different temperatures ranging from
$100^\circ K \to 1000^\circ K$. Finally the lattice parameter $a$ was examined. The total energy of the system was recorded for a series of
lattice parameters ranging from $0.9\sigma \to 1.3\sigma$. This was done at three fixed temperatures.
 
## Project Structure
 
```
.
├── sim.py            # Original MD simulation (brute-force pairwise forces)
├── sim_fast.py       # Optimized MD simulation with precomputed neighbor lists
├── MSD.py            # Mean-Squared Displacement over a range of temperatures
├── energies.py       # Total energy vs lattice parameter at fixed temperatures
├── fit.py            # Quadratic fitting to find equilibrium lattice parameter
├── plots.py          # Aggregated plotting of all results
├── demo.py           # 3D visualization of the Gd crystal structure
├── requirements.txt  # Python dependencies
├── data/             # JSON output files (auto-created)
└── figs/             # Saved figures (auto-created)
```
 
## Physical Model
 
The simulation models a Gadolinium (Gd) crystal using an HCP-like supercell (9 × 5 × 5 unit cells, 4 atoms per cell = 900 atoms). Interatomic forces are computed via a shifted Lennard-Jones potential with parameters $\sigma = 3.304$ Å and $\epsilon = 0.1136$ eV. Periodic boundary conditions are enforced through the minimum image convention. Temperature control is achieved through Langevin dynamics, where a friction term and stochastic noise are coupled to the Verlet velocity integrator. Random numbers are generated using a Numba-JIT-compiled Box-Muller transform.
 
## Pipeline
 
To run the code properly, start by installing `requirements.txt`:
 
```bash
pip install -r requirements.txt
```
 
### Task 1 — Auto-Correlation and Correlation Time
 
The files `sim.py` and `sim_fast.py` perform the exact same task. However, `sim_fast.py` utilizes a precomputed
neighbor list for all atoms, which significantly reduces compute time and yields similar results as `sim.py`. Therefore `sim_fast.py` is used in all other files in the project. `sim_fast.py` is also the point of entry in the project and performs the first task by calculating the auto-correlation and the correlation time. Run it by:
 
```bash
python src/sim_fast.py --y 0.01 --io
```
 
| Flag | Description |
|------|-------------|
| `--io` | Save thermalization and auto-correlation results to JSON |
| `--y` | Damping parameter $\gamma$ (default: 0.01) |
| `--T` | System temperature in K (default: 300) |
| `--plot` | Show plots after the run |
| `--seed` | Random seed for reproducibility (default: 0) |
 
### Task 2 — Mean-Squared Displacement
 
To evaluate the Mean-Squared Displacement run `MSD.py`. This code essentially runs the same simulation as `sim_fast.py` over an array of different temperatures (specified by `--points`), thus scaling up the compute time with the number of temperature points.
 
```bash
python src/MSD.py --points 7 --plot --io
```
 
This calculates the MSD for 7 temperatures equally divided across the range $100^\circ K \to 1000^\circ K$, shows the plot, and saves the results to JSON.
 
| Flag | Description |
|------|-------------|
| `--points` | Number of temperature points (default: 5) |
| `--plot` | Show MSD vs temperature plot |
| `--io` | Save MSD results to JSON |
| `--seed` | Random seed (default: 0) |
 
### Task 3 — Equilibrium Lattice Parameter
 
Run `energies.py` to sweep the lattice parameter $a$ from $0.95\sigma$ to $1.3\sigma$ at a fixed temperature and record the total energy at each value.
 
```bash
python src/energies.py --T 300 --step 0.1
```
 
| Flag | Description |
|------|-------------|
| `--T` | Fixed temperature in K (default: 300) |
| `--step` | Step size for lattice parameter sweep as fraction of $\sigma$ (default: 0.1) |
| `--plot` | Show energy vs lattice parameter scatter plot |
| `--noio` | Disable JSON output |
| `--seed` | Random seed (default: 0) |
 
Then run `fit.py` to perform a local quadratic fit around the energy minimum and extract the equilibrium lattice parameter for each temperature:
 
```bash
python src/fit.py
```
 
### Aggregated Plots
 
After running the above tasks with multiple seeds, use `plots.py` to load results and produce comparison figures (auto-correlation for different $\gamma$, temperature histories, MSD vs temperature, energy vs lattice parameter):
 
```bash
python src/plots.py
```
 
### Crystal Visualization
 
`demo.py` generates a 3D rendering of the Gd crystal structure:
 
```bash
python src/demo.py
```
 
## Reproducibility
 
Experiments were conducted by running all three major tasks (varying $\gamma$, varying $T$, and varying $a$) a number of times with different random seeds, then averaging the results. The `--seed` flag is available on all simulation scripts for this purpose.
 
## Output
 
All simulation results are saved as JSON files in the `data/` directory. All figures are saved as PNG images in the `figs/` directory. Both directories are created automatically on first run.
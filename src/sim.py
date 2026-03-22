import numpy as np
from numba import njit
import json
import matplotlib.pyplot as plt
import os
import argparse
from tqdm.auto import tqdm
from scipy.integrate import simpson

os.makedirs(f'./data', exist_ok=True)
os.makedirs(f'./figs', exist_ok=True)

k_B = 8.617333262e-5  # eV/K (Global Variable)
### Define Box-Muller ###

@njit
def box_muller_pair():
    u1 = np.random.random()
    u2 = np.random.random()

    # avoid log(0)
    if u1 < 1e-15:
        u1 = 1e-15

    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2

    z0 = r * np.cos(theta)
    z1 = r * np.sin(theta)
    return z0, z1


@njit
def normal_array_box_muller(nrows, ncols):
    out = np.empty((nrows, ncols), dtype=np.float64)
    total = nrows * ncols
    flat = out.ravel()

    i = 0
    while i < total:
        z0, z1 = box_muller_pair()
        flat[i] = z0
        if i + 1 < total:
            flat[i + 1] = z1
        i += 2

    return out



### 1. Initialization ###
def init_supercell(a, nx = 9, ny= 5, nz = 5):
    L_x = a
    L_y = np.sqrt(3)*a
    L_z = np.sqrt(8/3)*a
    basis_frac = np.array([
    [1/4, 5/6, 1/4],
    [1/4, 1/6, 3/4],
    [3/4, 1/3, 1/4],
    [3/4, 2/3, 3/4],
    ], dtype=float)

    #convert to real/cartesian system
    cell_length = np.array([L_x, L_y, L_z])
    basis_cart = basis_frac * cell_length

    # Build full cell
    pos = []

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                shift = np.array([i * L_x, j * L_y, k * L_z])
                for atom in basis_cart:
                    pos.append(atom + shift)

    pos = np.array(pos, dtype=float)
    box = np.array([nx * L_x, ny * L_y, nz * L_z], dtype=float)
    return pos, box


# Ensures periodic boundary conditions 

@njit
def minimum_image(diff, box):
    out = np.empty(3, dtype=np.float64)
    for d in range(3):
        out[d] = diff[d] - box[d] * np.round(diff[d] / box[d])
    return out



def distance_vector(pos, i, j, box):
    diff = pos[i] - pos[j]
    return minimum_image(diff, box)


def distance(pos, i, j, box):
    diff = distance_vector(pos, i, j, box)
    return np.sqrt(np.dot(diff, diff))

def E_kin(m, v):
    return 0.5 * m * np.sum(v**2)

# Check for thermalization
def Temp(m, v):
    n = len(v)
    return 2*E_kin(m, v)/(3 * n * k_B)


def init_velocity(pos, m, temp, seed = None):
    if seed is not None:
        np.random.seed(seed)
    
    n_atoms = len(pos)

    std = np.sqrt(k_B * temp / m)
    v = std * normal_array_box_muller(n_atoms, 3)

    E_kin_b = E_kin(m, v)
    v_m = v.mean(axis=0)
    v -= v_m
    E_kin_a = E_kin(m, v)
    v *= np.sqrt(E_kin_b / E_kin_a)
    return v



# Lennard-Jones potential 
def potential(r, epsilon, sigma, derivative = False):
    if not derivative:
        return 4*epsilon * ((sigma/r)**12 - (sigma/r)**6)
    else:
        return 48 * epsilon * ((sigma / r) ** 12 - 0.5 * (sigma / r) ** 6) / r

@njit
def compute_forces(pos, box, epsilon, sigma, cutoff):
    n = len(pos)
    forces = np.zeros_like(pos)
    epot = 0.0
    cutoff2 = cutoff**2 #compare squared distances isntead
    sigma2 = sigma**2
    for i in range(n-1):
        for j in range(i + 1, n):
            r_ij = minimum_image(pos[i] - pos[j], box)
            r2 = np.dot(r_ij, r_ij)

            if 0.0 < r2 < cutoff2:
                inv_r2 = 1 / r2

                sr2 = sigma2 * inv_r2      # (sigma/r)^2
                sr6 = sr2 * sr2 * sr2      # (sigma/r)^6
                sr12 = sr6 * sr6           # (sigma/r)^12


                epot += 4.0 * epsilon * (sr12 - sr6)
                pref = 24.0 * inv_r2 * epsilon * (2.0 * sr12 - sr6)
                f_ij = pref * r_ij

                # Implementing NIII-law of opposite forces.
                forces[i] += f_ij
                forces[j] -= f_ij

    return forces, epot



def wrap_positions(pos, box):
    return pos % box

### 2. Thermalization ###

def langevin_verlet(init_pos, init_vel, box, m, ts, temp, cutoff, damp, epsilon, sigma, n_therm):
    pos = init_pos.copy()
    vel = init_vel.copy()
    ts_2 = ts*0.5
    n_atoms = len(pos)
    f, _ = compute_forces(pos, box, epsilon, sigma, cutoff)

    temp_hist = []
    epot_hist = []
    ekin_hist = []

    #constants
    c1 = np.exp(-damp*ts_2)
    c2 = np.sqrt(temp*k_B / m * (1 - np.exp(-damp * ts)))

    for _ in tqdm(range(n_therm)):

        xi1 = normal_array_box_muller(n_atoms, 3)
        xi2 = normal_array_box_muller(n_atoms, 3)

        vel = c1*vel + c2*xi1
        a1 = f / m
        vel1 = vel + a1 * ts_2
        # vel1 -= vel1.mean(axis = 0)

        pos = pos + vel1 * ts
        pos = wrap_positions(pos, box)

        f2, epot = compute_forces(pos, box = box, epsilon=epsilon, sigma = sigma, cutoff = cutoff)
        a2 = f2 / m
        vel2 = vel1 + a2*ts_2
        vel = c1 * vel2 + c2 * xi2

        f = f2

        sys_temp = Temp(m, vel)
        epot_hist.append(epot)
        ekin_hist.append(E_kin(m, vel))
        temp_hist.append(sys_temp)

    return pos, vel, f,  np.array(temp_hist), np.array(epot_hist), np.array(ekin_hist)


### 3. Production run ###

def production_run(therm_pos, therm_vel, box, m, ts, temp, cutoff, damp, epsilon, sigma, n_steps, q_st):
    pos = therm_pos.copy()
    vel = therm_vel.copy()
    unwrapped_pos = therm_pos.copy()

    ts_2 = ts*0.5
    n_atoms = len(pos)
    f, _ = compute_forces(pos, box, epsilon, sigma, cutoff)

    disp_traj = np.zeros((n_steps, n_atoms, 3))
    temp_hist = np.zeros(n_steps)
    epot_hist = np.zeros(n_steps)
    ekin_hist = np.zeros(n_steps)

    #constants
    c1 = np.exp(-damp*ts_2)
    c2 = np.sqrt(temp*k_B / m * (1 - np.exp(-damp * ts)))
    for step in tqdm(range(n_steps)):

        xi1 = normal_array_box_muller(n_atoms, 3)
        xi2 = normal_array_box_muller(n_atoms, 3)

        vel = c1*vel + c2*xi1
        a1 = f / m
        vel1 = vel + a1 * ts_2

        # vel1 -= vel1.mean(axis = 0)

        new_pos = pos + vel1 * ts
        unwrapped_pos = unwrapped_pos + vel1 * ts
        pos = wrap_positions(new_pos, box)

        f2, epot = compute_forces(pos, box = box, epsilon=epsilon, sigma = sigma, cutoff = cutoff)
        a2 = f2 / m
        vel2 = vel1 + a2*ts_2
        vel = c1 * vel2 + c2 * xi2

        f = f2

        disp_traj[step] = unwrapped_pos - q_st
        temp_hist[step] = Temp(m, vel)
        epot_hist[step] = epot
        ekin_hist[step] = E_kin(m, vel)

    
    
    return pos, vel, f, disp_traj, temp_hist, epot_hist, ekin_hist


def auto_correlation(disp_traj, t_delay=1000):
    nsteps, natoms, ndim = disp_traj.shape
    t_delay = min(t_delay, nsteps)
    disp_flat = disp_traj.reshape(nsteps, natoms * ndim)  # (nsteps, 3N)

    # Per-coordinate variance <Δq_i²> averaged over all time origins
    denom_per_coord = np.mean(disp_flat ** 2, axis=0)  # shape (3N,)
    valid = denom_per_coord > 0.0

    C = np.zeros(t_delay, dtype=float)
    for delay in tqdm(range(t_delay)):
        x1 = disp_flat[:nsteps - delay]   # (nsteps-delay, 3N)
        x2 = disp_flat[delay:]            # (nsteps-delay, 3N)
        numer_per_coord = np.mean(x1 * x2, axis=0)  # shape (3N,)
        # Average over valid coords: sum(numer/denom) / 3N
        C[delay] = np.mean(numer_per_coord[valid] / denom_per_coord[valid])

    return C

def correlation_time(C, dt):
    return simpson(C, dx=dt)


### 4. Repeat steps 1-3 ###


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Molecular Dynamics Simulation')
    parser.add_argument("--plot", action="store_true",
                        help="Enable plotting")
    parser.add_argument("--io", action="store_true",
                        help="Enable saving JSON output")

    # # Optional numeric arguments
    parser.add_argument("--T", type=float, default=300.0,
                         help="Temperature")
    parser.add_argument("--y", type=float, default=0.01,
                        help="damping value")
    # parser.add_argument("--sweeps", type=int, default=10000,
    #                     help="Number of MC sweeps")
    # parser.add_argument("--eq", type=int, default=1200,
    #                     help="Equilibration sweeps")

    args = parser.parse_args()


    gamma = args.y
    sigma = 3.304        # Å
    epsilon = 0.1136     # eV
    a = 1.1225*sigma     # Lattice parameter (Å)
    ts = 1               # Time step (fs)
    Temp_target = args.T          # Temperature
    cutoff = 10.0          # Å
    
    u = 157.25 
    m = 103.6 * u
    k_B = 8.617333262e-5  # eV/K
    T = 10000
    t_delay = 2000

    if gamma == 0.001:
        thermalization = 2500
    elif gamma == 0.01:
        thermalization = 2000
    else:
        thermalization = 2000

    pos, box = init_supercell(a)
    q_st = pos.copy()

    vel = init_velocity(pos, m, Temp_target, seed=1)
    
    

    pos, vel, force, T_hist, U_hist, K_hist = langevin_verlet(pos, vel, box, m, 
                                                              ts, Temp_target, cutoff, 
                                                              gamma, epsilon, sigma, 
                                                              thermalization)

    

    results_therm = {
        'pos':pos.copy().tolist(),
        'vel':vel.copy().tolist(),
        'force': force.tolist(),
        'T_hist': T_hist.tolist(),
        'U_hist': U_hist.tolist(),
        'K_hist': K_hist.tolist(),
        'gamma' : gamma,
        'thermalization' : thermalization
    }

    print("Final temperature:", T_hist[-1])
    print("Mean temperature last 500 steps:", np.mean(T_hist[-500:]))
    print("Std temperature last 500 steps:", np.std(T_hist[-500:]))



    pos, vel, f, disp_traj, temp_hist, epot_hist, ekin_hist = production_run(pos, vel, box,
                                                             m, ts, Temp_target,
                                                             cutoff, gamma, epsilon,
                                                             sigma, n_steps=T, q_st=q_st)


    com_disp = disp_traj.mean(axis=1, keepdims=True)   # shape (nsteps, 1, 3)
    disp_traj -= com_disp

    C = auto_correlation(disp_traj, t_delay=t_delay)


    tau = correlation_time(C, ts)
    print(f'Correlation time: {tau}')
    t = np.arange(t_delay) * ts

    results = {
        'temp_hist' : temp_hist.tolist(),
        'epot_hist': epot_hist.tolist(),
        'ekin_hist': ekin_hist.tolist(),
        'C': C.tolist(),
        't': t.tolist(),
        'tau': tau,
        't_delay':t_delay,
        'T': T

    }


    if args.io: 
        with open(f'./data/therm_gamma{gamma}.json', 'w') as f:
            json.dump(results_therm, f, indent=4)
            print('Thermalization results saved in JSON')
        
        with open(f'./data/autocor_gamma{gamma}.json', 'w') as f:
            json.dump(results, f, indent=4)
            print('Auto Correlation Results saved in JSON')

    if args.plot:
        plt.figure(figsize=(8, 4))
        plt.plot(T_hist)
        plt.axhline(Temp_target, linestyle='--')
        plt.xlabel("Step")
        plt.ylabel("Temperature (K)")
        plt.title("Thermalization check")
        plt.tight_layout()
        plt.savefig(f'./figs/thermalization_gamma{gamma}.png')
        plt.show()

        plt.figure(figsize=(8, 4))
        plt.plot(t, C)
        plt.xlabel("Delay (t)")
        plt.ylabel("Auto-Correlation (C(t))")
        plt.title(fr"Auto correlation with $\gamma = {gamma}$")
        plt.tight_layout()
        plt.savefig(f'./figs/autocor_gamma{gamma}.png')
        plt.show()




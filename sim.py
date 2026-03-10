import numpy as np
import json
import matplotlib.pyplot as plt

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
def minimum_image(diff, box):
    return diff - box * np.round(diff / box)


def distance_vector(pos, i, j, box):
    diff = pos[i] - pos[j]
    return minimum_image(diff, box)


def distance(pos, i, j, box):
    diff = distance_vector(pos, i, j, box)
    return np.sqrt(np.dot(diff, diff))

def E_kin(m, v):
    return 0.5 * m * np.sum(v**2)

# Check for thermalization
def Temp (m, v):
    n = len(v)
    return E_kin(m, v)/(3/2 * n * k_B)


def init_velocity(pos, m, temp, seed = None):
    if seed is not None:
        np.random.seed(seed)
    
    n_atoms = len(pos)

    std = np.sqrt(k_B * temp / m)
    v = np.random.normal(0, std, size= (n_atoms, 3))

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


def compute_forces(pos, box, epsilon, sigma, cutoff):
    n = len(pos)
    forces = np.zeros_like(pos)
    epot = 0.0
    cutoff2 = cutoff**2 #compare squared distances isntead

    for i in range(n-1):
        for j in range(i + 1, n):
            r_ij = minimum_image(pos[i] - pos[j], box)
            r2 = np.dot(r_ij, r_ij)

            if 0.0 < r2 < cutoff2:
                inv_r2 = 1 / r2

                term1 = (sigma**2*inv_r2)**6 # (sigma/r)**12
                term2 = (sigma**2*inv_r2)**3 # (sigma/r)**6

                epot += 4 * epsilon * (term1 - term2)
                pref = 24 * inv_r2 * epsilon * (2.0 * term1 - term2)
                f_ij = pref * r_ij

                # Implementing NIII-law of opposite forces.
                forces[i] += f_ij
                forces[j] -= f_ij

    return forces, epot



def wrap_positions(pos, box):
    return pos % box

### 2. Thermalization ###


### 3. Production run ###



### 4. Repeat steps 1-3 ###


if __name__ == '__main__':
    sigma = 3.304        # Å
    epsilon = 0.1136     # eV
    a = 1.1225*sigma     # Lattice parameter (Å)
    ts = 1               # Time step (fs)
    Temp = 300.0          # Temperature
    cutoff = 10.0          # Å
    y_damp = [0.001, 0.01, 0.1]
    thermalization = 1000
    u = 157.25 
    m = 103.6 * u
    k_B = 8.617333262e-5  # eV/K



    pos, box = init_supercell(a)

    print(box)
    q_st = pos.copy()
    forces, epot = compute_forces(pos, box, epsilon, sigma, cutoff)
    # should be close to zero

    print(np.sum(forces, axis = 0))
    vel = init_velocity(pos, m, Temp, 0)

    print(vel.shape)


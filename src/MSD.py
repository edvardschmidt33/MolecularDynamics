from sim_fast import *
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from plots import file_ret
from pathlib import Path
from tqdm.auto import tqdm


def MSD(disp_traj, sample_stride) -> float:
    samples = disp_traj[::sample_stride]

    sq_sum= np.sum(samples**2, axis = 2)
    MSD = sq_sum.mean()
    
    return MSD



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Molecular Dynamics Simulation')
    parser.add_argument("--plot", action="store_true",
                        help="Enable plotting")
    parser.add_argument("--io", action="store_true",
                        help="Enable saving JSON output")

    # # Optional numeric arguments
    parser.add_argument("--points", type=int, default=5,
                         help="Number of temperatures to run")


    args = parser.parse_args()
    temp001, C001, t001, tau = file_ret(0.01)
    Temps = np.linspace(100, 1000, args.points)
    gamma = 0.01
    sigma = 3.304        # Å
    epsilon = 0.1136     # eV
    a = sigma     # Lattice parameter (Å)
    ts = 1               # Time step (fs)
    cutoff = 10.0          # Å
    
    u = 157.25 
    m = 103.6 * u
    k_B = 8.617333262e-5  # eV/K
    T_steps = 10000
    t_delay = 2000
    thermalization = 1500

    MSD_list = np.zeros_like(Temps, dtype = float)

    for i, T in enumerate(tqdm(Temps, desc='Processing temperatures')):
        pos, box = init_supercell(a)
        q_st = pos.copy()

        
        
        neighbor_cutoff = 12.0   # Å
        print("Building neighbor list...")
        pairs_i, pairs_j = build_neighbor_list(pos, box, neighbor_cutoff)
        n_atoms_check = len(pos)
        neighbor_count = np.zeros(n_atoms_check, dtype=int)
        np.add.at(neighbor_count, pairs_i, 1)
        np.add.at(neighbor_count, pairs_j, 1)
        print(f"Neighbors per atom: min={neighbor_count.min()}, max={neighbor_count.max()}, "
            f"mean={neighbor_count.mean():.1f}  (should all be equal)")
        assert neighbor_count.min() == neighbor_count.max(), \
            "Neighbor counts differ — check PBC in build_neighbor_list!"
        print(f"Total neighbor pairs: {len(pairs_i)}  (vs {n_atoms_check*(n_atoms_check-1)//2} full pairs)\n")
        


        vel = init_velocity(pos, m, T, seed=1)
        pos, vel, force, T_hist, U_hist, K_hist = langevin_verlet(pos, vel, box, m,
                                                              ts, T, cutoff,
                                                              gamma, epsilon, sigma,
                                                              thermalization,
                                                              pairs_i, pairs_j)
        
        print("Final temperature:", T_hist[-1])
        print("Mean temperature last 500 steps:", np.mean(T_hist[-500:]))
        print("Std temperature last 500 steps:", np.std(T_hist[-500:]))




        pos, vel, f, disp_traj, temp_hist, epot_hist, ekin_hist = production_run(pos, vel, box,
                                                                m, ts, T,
                                                                cutoff, gamma, epsilon,
                                                                sigma, n_steps=T_steps, q_st=q_st,
                                                                pairs_i=pairs_i, pairs_j=pairs_j)


        com_disp = disp_traj.mean(axis=1, keepdims=True)   # shape (nsteps, 1, 3)
        disp_traj -= com_disp

        C = auto_correlation(disp_traj, t_delay=t_delay)

        tau = correlation_time(C, ts)
        print(f'Correlation time: {tau}')
        sample_stride = int(tau / ts)

        MSD_t = MSD(disp_traj, sample_stride)
        MSD_list[i] = MSD_t
    
    if args.io: 
        res = {'MSD': MSD_list.tolist(),
               'Temps': Temps.tolist()}
        with open(f'./data/MSD_list.json', 'w') as f:
            json.dump(res, f, indent=4)
            print('MSD results saved in JSON')
    
    if args.plot:
        plt.scatter(Temps, MSD_list, color = 'IndianRed')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Mean-Squared Displacement (MSD)')
        plt.title(f'MSD over {len(Temps)} temperatures form 100 - 1000 K')
        plt.show()



            
            

        
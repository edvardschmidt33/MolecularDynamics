from sim_fast import *
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from tqdm.auto import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Molecular Dynamics Simulation')
    parser.add_argument("--plot", action="store_true",
                        help="Enable plotting")
    parser.add_argument("--noio", action="store_true",
                        help="Disable saving JSON output")

    # # Optional numeric arguments
    parser.add_argument("--step", type=float, default=0.1,
                         help="Stepsize for lattice parameter")
    parser.add_argument("--T", type=float, default=300.0,
                         help="System temperature")
    parser.add_argument("--seed", type=int, default = 0,
                        help= 'Random-seed for simulation')

    args = parser.parse_args()    
    seed = args.seed
    np.random.seed(seed)

    sigma = 3.304  
    h = args.step * sigma
    epsilon = 0.1136     # eV
    start = 0.95*sigma
    stop = 1.3*sigma
    N = int((stop - start)/h) + 1
    a_list = np.linspace(start, stop, N)
    gamma = 0.01
    temp = args.T
    ts = 1               # Time step (fs)
    cutoff = 10.0          # Å
    
    u = 157.25 
    m = 103.6 * u
    k_B = 8.617333262e-5  # eV/K
    T_steps = 10000
    t_delay = 2000
    thermalization = 1500

    energies = np.zeros_like(a_list)


    for i, a in enumerate(tqdm(a_list, desc='Lattice parameters')):
        ### Do thermalization ### 
           
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
        

        vel = init_velocity(pos, m, temp, seed=seed)

        pos, vel, force, T_hist, U_hist, K_hist = langevin_verlet(pos, vel, box, m,
                                                              ts, temp, cutoff,
                                                              gamma, epsilon, sigma,
                                                              thermalization,
                                                              pairs_i, pairs_j)
        
        print("Final temperature:", T_hist[-1])
        print("Mean temperature last 500 steps:", np.mean(T_hist[-500:]))
        print("Std temperature last 500 steps:", np.std(T_hist[-500:]))


        pos, vel, f, disp_traj, temp_hist, epot_hist, ekin_hist = production_run(pos, vel, box,
                                                                m, ts, temp,
                                                                cutoff, gamma, epsilon,
                                                                sigma, n_steps=T_steps, q_st=q_st,
                                                                pairs_i=pairs_i, pairs_j=pairs_j)


        E_kin_m = np.mean(ekin_hist)
        E_pot_m = np.mean(epot_hist)

        E_tot = E_kin_m + E_pot_m

        energies[i] = E_tot

    if args.plot:

        plt.scatter(a_list, energies)
        plt.xlabel('Lattice parameter')
        plt.ylabel('Mean Energy')
        plt.show()

    if not args.noio:
        res = {'Energies': energies.tolist(),
                'a_list': a_list.tolist(),
                'temp':temp}
        
        with open(f'./data/energies_temp{temp}_{seed}.json', 'w') as f:
            json.dump(res, f, indent=4)
        print('Energy results saved in JSON')

import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.integrate import simpson
from fit import file_ret_energy

def file_ret(gamma, n_seeds, fast=False):
    temp_hists = []
    C_values = []
    tau_values = []
    t = None
 
    for seed in range(n_seeds):
        if fast:
            filename = f'./data/autocor_gamma{gamma}_fast{seed}.json'
        else:
            filename = f'./data/autocor_gamma{gamma}{seed}.json'
 
        try:
            with open(filename, "r") as f:
                data = json.load(f)
 
            temp_hists.append(np.array(data["temp_hist"]))
            C_values.append(np.array(data["C"]))
            if t is None:
                t = np.array(data["t"])
            tau_values.append(np.array(data["tau"]))
 
        except FileNotFoundError:
            print(f"Warning: File {filename} not found. Skipping seed {seed}.")
            continue
 
    # Average over all seeds
    temp_hist_avg = np.mean(temp_hists, axis=0)
    C_avg = np.mean(C_values, axis=0)
    tau_values = np.array(tau_values)
    tau = np.mean(tau_values)
 
    return temp_hist_avg, C_avg, t, tau

def file_ret_MSD(n_seeds):
    MSD_list = []
    Temps = None
 
    for seed in range(n_seeds):
        filename = f'./data/MSD_list{seed}.json'
 
        try:
            with open(filename, "r") as f:
                data = json.load(f)
 
            MSD_list.append(np.array(data["MSD"]))
            if Temps is None:
                Temps = np.array(data["Temps"])
 
        except FileNotFoundError:
            print(f"Warning: File {filename} not found. Skipping seed {seed}.")
            continue
 
    # Average and compute std over all seeds
    MSD_array = np.array(MSD_list)
    MSD_avg = np.mean(MSD_array, axis=0)
    MSD_std = np.std(MSD_array, axis=0)
 
    return MSD_avg, MSD_std, Temps


if __name__ == '__main__':

    temp01, C01, t01, tau01 = file_ret(0.1, 3, fast=True)
    temp001, C001, t001, tau001 = file_ret(0.01, 3, fast= True)
    temp0001, C0001, t0001, tau0001 = file_ret(0.001, 3, fast = True)

    MSD_list, _,  Temps = file_ret_MSD(3)

    plt.plot(t01, C01, color = 'RoyalBlue', label = fr'$\gamma = 0.1$,  $\tau = {tau01:.3f}$')
    plt.plot(t001, C001, color = 'IndianRed', label = fr'$\gamma = 0.01$,  $\tau = {tau001:.3f}$')
    plt.plot(t0001, C0001, color = 'ForestGreen', label = fr'$\gamma = 0.001$,  $\tau = {tau0001:.3f}$')
    plt.hlines(0, xmin = t01[0]-100, xmax=t01[-1]+100, linestyles='--', colors='black', alpha = 0.7)
    plt.legend()
    plt.xlabel(f'Delay-time (t) (fs)')
    plt.ylabel(f'Auto Correlation (C(t))')
    plt.title(f'Auto Correlation comparison for $\gamma = 0.1, 0.01, 0.001$')
    plt.savefig('./figs/autocor_aggr.png')
    plt.show()

    plt.plot(temp01, color = 'RoyalBlue', label = fr'$\gamma = 0.1$, mean T  = {temp01.mean():.3f}')
    plt.plot(temp001, color = 'IndianRed', label = fr'$\gamma = 0.01$, mean T  = {temp001.mean():.3f}')
    plt.plot(temp0001, color = 'ForestGreen', label = fr'$\gamma = 0.001$, mean T  = {temp0001.mean():.3f}')
    plt.hlines(300, 0, 10000, linestyles='--', colors='black', alpha = 0.7)
    plt.legend()
    plt.xlabel(f'Timesteps')
    plt.ylabel(f'Temperature T')
    plt.title(fr'System Temperature comparison for $ \gamma $ = 0.1, 0.01, 0.001')
    plt.savefig('./figs/temp_aggr.png')
    plt.show()


    plt.plot(Temps, MSD_list, color = 'IndianRed', marker = 'o')
    plt.xlabel('Temperature (K)')
    plt.ylabel(fr'MSD ($Å^2$)')
    plt.title('MSD as a function of temperature')
    plt.savefig('./figs/MSD_plot.png')
    plt.show()
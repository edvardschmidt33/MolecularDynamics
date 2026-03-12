import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.integrate import simpson

def file_ret(gamma, fast = False):
    if fast:
        filename = f'./data/autocor_gamma{gamma}_fast.json'
    else:
        filename = f'./data/autocor_gamma{gamma}.json'

    with open(filename, "r") as f:
        data = json.load(f)
    temp_hist = np.array(data["temp_hist"])
    C = np.array(data["C"])
    t = np.array(data["t"])
    tau = np.array(data["tau"])

    return temp_hist, C, t, tau

def correlation_time(C, dt):
    # Find first zero crossing
    zero_crossing = np.where(C <= 0)[0]
    if len(zero_crossing) > 0:
        cutoff = zero_crossing[0]
    else:
        cutoff = len(C)  # no zero crossing found, use full array
    return simpson(C[:cutoff], dx=dt)

if __name__ == '__main__':

    temp01, C01, t01, tau01 = file_ret(0.1, fast=True)
    temp001, C001, t001, tau001 = file_ret(0.01, True)
    temp0001, C0001, t0001, tau0001 = file_ret(0.001,True)


    with open('./data/MSD_list.json', "r") as f:
        data = json.load(f)
    MSD_list = np.array(data["MSD"])
    Temps = np.array(data["Temps"])


    # tau01 = correlation_time(C01,1)
    # tau001 = correlation_time(C001,1)
    # tau0001 = correlation_time(C0001,1)

    plt.plot(t01, C01, color = 'RoyalBlue', label = fr'$\gamma = 0.1$,  $\tau = {tau01:.3f}$')
    plt.plot(t001, C001, color = 'IndianRed', label = fr'$\gamma = 0.01$,  $\tau = {tau001:.3f}$')
    plt.plot(t0001, C0001, color = 'ForestGreen', label = fr'$\gamma = 0.001$,  $\tau = {tau0001:.3f}$')
    plt.hlines(0, xmin = t01[0]-100, xmax=t01[-1]+100, linestyles='--', colors='black', alpha = 0.7)
    plt.legend()
    plt.xlabel(f'Delay (t)')
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
    plt.ylabel('MSD')
    plt.title('MSD as a function of temperature')
    plt.savefig('./figs/MSD_plot.png')
    plt.show()
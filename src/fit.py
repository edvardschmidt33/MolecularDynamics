import numpy as np
import matplotlib.pyplot as plt
import json


def file_ret_energy(temp=300.0, seed=0):
    filename = f'./data/energies_temp{temp}_{seed}.json'

    with open(filename, "r") as f:
        data = json.load(f)

    energies = np.array(data["Energies"])
    temp = np.array(data["temp"])
    a_list = np.array(data["a_list"])

    return energies, a_list, temp


def local_curve_fit(temp=300.0, seed=0, n_neighbors=2):
    """
    Fit a parabola locally around the discrete minimum.

    n_neighbors=2 means:
    use 2 points on each side of the minimum if available,
    i.e. typically 5 points total.
    """
    energies, a_list, temp = file_ret_energy(temp, seed)

    # 1. find discrete minimum
    i_min = np.argmin(energies)

    # 2. choose local fitting window
    i1 = max(0, i_min - n_neighbors)
    i2 = min(len(a_list), i_min + n_neighbors + 1)

    a_fit = a_list[i1:i2]
    E_fit = energies[i1:i2]

    # Safety check: need at least 3 points for quadratic fit
    if len(a_fit) < 3:
        raise ValueError("Not enough points for quadratic fit. Increase number of sampled a-values.")

    # 3. quadratic fit only near minimum
    coeffs = np.polyfit(a_fit, E_fit, 2)

    # coeffs = [c2, c1, c0]
    c2, c1, c0 = coeffs

    if c2 <= 0:
        print(f"Warning: fitted parabola opens downward for T={temp}. Fit may be unreliable.")

    # 4. vertex of parabola = fitted equilibrium lattice parameter
    a_eq = -c1 / (2 * c2)

    # evaluate smooth fitted curve for plotting
    a_plot = np.linspace(a_fit[0], a_fit[-1], 200)
    E_plot = np.polyval(coeffs, a_plot)

    return {
        "energies": energies,
        "a_list": a_list,
        "temp": temp,
        "i_min": i_min,
        "a_fit": a_fit,
        "E_fit": E_fit,
        "coeffs": coeffs,
        "a_eq": a_eq,
        "a_plot": a_plot,
        "E_plot": E_plot,
    }


if __name__ == '__main__':

    sigma = 3.304

    fit50 = local_curve_fit(50.0)
    fit300 = local_curve_fit(300.0)
    fit1000 = local_curve_fit(1000.0)
    fit150 = local_curve_fit(150.0)
    fit600 = local_curve_fit(600.0)

    # raw data
    energies50, a_list50 = fit50["energies"], fit50["a_list"]
    energies300, a_list300 = fit300["energies"], fit300["a_list"]
    energies1000, a_list1000 = fit1000["energies"], fit1000["a_list"]
    energies150, a_list150 = fit150["energies"], fit150["a_list"]
    energies600, a_list600 = fit600["energies"], fit600["a_list"]

    # fitted curves
    a_plot50, E_plot50 = fit50["a_plot"], fit50["E_plot"]
    a_plot300, E_plot300 = fit300["a_plot"], fit300["E_plot"]
    a_plot1000, E_plot1000 = fit1000["a_plot"], fit1000["E_plot"]
    a_plot150, E_plot150 = fit150["a_plot"], fit150["E_plot"]
    a_plot600, E_plot600 = fit600["a_plot"], fit600["E_plot"]

    # fitted equilibrium lattice parameters
    equil50 = fit50["a_eq"]
    equil300 = fit300["a_eq"]
    equil1000 = fit1000["a_eq"]
    equil150 = fit150["a_eq"]
    equil600 = fit600["a_eq"]
    

    plt.scatter(a_list1000, energies1000, label='T = 1000 K', color='IndianRed')
    plt.scatter(a_list300, energies300, label='T = 300 K', color='RoyalBlue')
    plt.scatter(a_list50, energies50, label='T = 50 K', color='ForestGreen')
    plt.scatter(a_list150, energies150, label='T = 150 K', color='Orange')
    plt.scatter(a_list600, energies600, label='T = 600 K', color='Violet')

    plt.plot(a_plot1000, E_plot1000, color='IndianRed', label='local fit T = 1000 K')
    plt.plot(a_plot300, E_plot300, color='RoyalBlue', label='local fit T = 300 K')
    plt.plot(a_plot50, E_plot50, color='ForestGreen', label='local fit T = 50 K')
    plt.plot(a_plot150, E_plot150, color='Orange', label='local fit T = 50 K')
    plt.plot(a_plot600, E_plot600, color='Violet', label='local fit T = 50 K')

    # mark fitted minima
    plt.axvline(equil50, color='ForestGreen', linestyle='--', alpha=0.5)
    plt.axvline(equil300, color='RoyalBlue', linestyle='--', alpha=0.5)
    plt.axvline(equil1000, color='IndianRed', linestyle='--', alpha=0.5)
    plt.axvline(equil150, color='Orange', linestyle='--', alpha=0.5)
    plt.axvline(equil600, color='Violet', linestyle='--', alpha=0.5)

    plt.xlabel('Lattice parameter (Å)')
    plt.ylabel('Total Energy (eV)')
    plt.title(r'$E_{\mathrm{tot}}$ as a function of lattice parameter $a$')
    plt.legend()
    plt.savefig('./figs/en_vs_latticeparam.png')
    plt.show()

    print(60 * '-')
    print('Calculated Equilibrium Lattice parameter for T = 50 K')
    print(20 * ' ' + f'a = {equil50:.6f} Å')
    print(20 * ' ' + f'a / sigma = {equil50 / sigma:.6f}')
    print(60 * '-')

    print('Calculated Equilibrium Lattice parameter for T = 150 K')
    print(20 * ' ' + f'a = {equil150:.6f} Å')
    print(20 * ' ' + f'a / sigma = {equil150 / sigma:.6f}')
    print(60 * '-')

    print('Calculated Equilibrium Lattice parameter for T = 300 K')
    print(20 * ' ' + f'a = {equil300:.6f} Å')
    print(20 * ' ' + f'a / sigma = {equil300 / sigma:.6f}')
    print(60 * '-')

    print('Calculated Equilibrium Lattice parameter for T = 600 K')
    print(20 * ' ' + f'a = {equil600:.6f} Å')
    print(20 * ' ' + f'a / sigma = {equil600 / sigma:.6f}')
    print(60 * '-')

    print('Calculated Equilibrium Lattice parameter for T = 1000 K')
    print(20 * ' ' + f'a = {equil1000:.6f} Å')
    print(20 * ' ' + f'a / sigma = {equil1000 / sigma:.6f}')
    print(60 * '-')

    plt.plot([50, 150, 300, 600, 1000], [equil50, equil150, equil300, equil600, equil1000], color = 'IndianRed')
    plt.xlabel(fr'Temperatures ($^\circ K $)')
    plt.ylabel(fr'Equilibrium Lattice parameter ($Å$)')
    plt.title('Equilibrium Lattice parameter as a function of temperature')
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
import json


def file_ret_energy(temp=300.0):
    filename = f'./data/energies_temp{temp}.json'

    with open(filename, "r") as f:
       data = json.load(f)
    
    energies = np.array(data["Energies"])
    temp = np.array(data["temp"])
    a_list = np.array(data["a_list"])

    return energies, a_list, temp


def curve_fit(temp = 300.0):
    energies, a_list, temp = file_ret_energy(temp)  
    coeffs =  np.polyfit(a_list[0:], energies[0:], 2)
    a_eq = -coeffs[1] / (2 * coeffs[0])           # vertex of parabola
    parabola = np.polyval(coeffs, a_list)
    
    return parabola


if __name__ == '__main__':

    energies1000, a_list1000, temp1000 = file_ret_energy(1000.0)
    energies300, a_list300, temp300 = file_ret_energy(300.0)
    energies50, a_list50, temp50 = file_ret_energy(50.0)
    
    parabola1000 = curve_fit(1000.0)
    parabola300 = curve_fit(300.0)
    parabola50 = curve_fit(50.0)
    plt.scatter(a_list1000[0:], energies1000[0:], label = 'T = 1000', color = 'IndianRed')
    plt.scatter(a_list300[0:], energies300[0:], label = 'T = 300', color = 'RoyalBlue')
    plt.scatter(a_list50[0:], energies50[0:], label = 'T = 50', color = 'ForestGreen')
    plt.plot(a_list1000,parabola1000, color = 'IndianRed', label = 'fitted T = 1000')
    plt.plot(a_list300,parabola300, color = 'RoyalBlue', label = 'fitted T = 300')
    plt.plot(a_list50,parabola50, color = 'ForestGreen', label = 'fitted T = 50')
    plt.xlabel('Lattice parameter')
    plt.ylabel('Total Energy')
    plt.title(fr'$E_{{tot}}$ as a function of lattice parameter for $T = 50^\circ, 300^\circ, 1000^\circ K$')
    plt.legend()
    plt.show()


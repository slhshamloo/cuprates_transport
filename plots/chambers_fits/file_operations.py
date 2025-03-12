import numpy as np
import os, re
from copy import deepcopy

from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.conductivity import Conductivity
import plot_chambers_fits
import defaults
import lmfit

### From plot_chambers_fits.py

def load_data(paper, sample, field):
    omega_dict = {}
    sigma_dict = {}
    for real_imag in ('1', '2'):
        for side in ('l', 'r'):
            omega_dict[real_imag+side], sigma_dict[real_imag+side] = (
                np.loadtxt(f"data/{paper}/{sample}_sigma"
                           f"{real_imag}{side}_{field}T.csv",
                           skiprows=1, delimiter=',', unpack=True))
    omega = np.concatenate((omega_dict['1l'], omega_dict['1r']))
    sigma1 = np.concatenate((sigma_dict['1l'], sigma_dict['1r']))
    sigma2 = np.concatenate((
        np.interp(omega_dict['1l'], omega_dict['2l'], sigma_dict['2l']),
        np.interp(omega_dict['1r'], omega_dict['2r'], sigma_dict['2r'])))
    sigma = sigma1 + 1j * sigma2
    return omega, sigma

def save_fit(fit_result, sample, fields, extra_info):
    fieldString = '-'.join([str(f) for f in fields])
    with open(os.path.dirname(os.path.relpath(__file__))
              + f"/params/{sample}_{fieldString}T{extra_info}.dat", 'w') as f:
        f.write(lmfit.fit_report(fit_result))


def load_fit(sample, fields, extra_info=""):
    parameter_values = {}
    parameter_errors = {}
    fieldString = '-'.join([str(f) for f in fields])
    with open(os.path.dirname(os.path.relpath(__file__))
              + f"/params/{sample}_{fieldString}T{extra_info}.dat", 'r') as f:
        while True:
            line = f.readline()
            if not line:
                return parameter_values, parameter_errors
            if "[[Variables]]" in line:
                break
        line = f.readline()
        while line and not re.match(r"\[\[[\w\s]*\]\]", line):
            if line:
                value_name = line.strip().split()[0][:-1]
                parameter_values[value_name] = float(line.split()[1])
                parameter_errors[value_name] = line.split()[3]
            line = f.readline()
    return parameter_values, parameter_errors



def save_fit_output_data(sample, field, omegas):
    sigma = plot_chambers_fits.generate_chambers_fit(sample, field, 2*np.pi*omegas)
    np.savetxt(os.path.dirname(os.path.relpath(__file__))
               + f"/outputs/{sample}_{field}T.csv",
               np.column_stack((omegas, sigma.real, sigma.imag)),
               delimiter=",", header="omega,sigma_real,sigma_imag")


def load_fit_output_data(sample, field):
    omega, sigma_real, sigma_imag = np.loadtxt(
        os.path.dirname(os.path.relpath(__file__))
        + f"/outputs/{sample}_{field}T.csv",
        delimiter=",", skiprows=1, unpack=True)
    return omega, sigma_real + 1j * sigma_imag
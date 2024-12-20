import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import os
import sys
from lmfit import minimize, Parameters, report_fit
import pickle
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

## Free parameters to vary
vary = {
    "omega_c": True,
    "omega_pn": True,
    "omega_ps": True,
    "gamma": True,
    "epsilon_inf": False
}

## Field and Temperature values to probe
temperature = [
    35, 40, 45, 50  
]

field = [
    0, 9, 20, 31
]



## Fit ///////////////////////////////////////////////////////////////////////////
def fit_model(omega, gamma, omega_pn, omega_c, omega_ps):
    """Compute optical conductivity in the Drude model with given parameters
    
    All parameters must be given in THz and the result is in (mOhm.cm)^-1"""
    e0 = 8.854e-12
    return 1j*1e7*e0*(2*np.pi*omega_pn)**2/(2*np.pi*omega - omega_c + 1j*gamma) + 1j*(2*np.pi*omega_ps)**2/omega


def compute_diff(pars, reals, imags, d_real, d_imag):
    gamma   = pars["gamma"].value
    omega_pn  = pars["omega_pn"].value
    omega_c = pars["omega_c"].value
    omega_ps = pars["omega_ps"].value
    sim_r = np.real(fit_model(reals, gamma, omega_pn, omega_c, omega_ps))
    sim_imag = np.imag(fit_model(imags, gamma, omega_pn, omega_c, omega_ps))
    return np.concatenate([sim_r - d_real, sim_imag - d_imag])


def fitOneCase(T,B):
    ## Load Data /////////////////////////////////////////////////////////////////////
    fullpath = os.path.relpath(__file__)
    dirname, fname = os.path.split(fullpath)
    project_root = dirname + "/../../"
    datapath = project_root+"data/post"

    # Handle the real part
    x_data, y_data = [], []
    i = 1
    for j in ["r", "l"]:
        filename = f"post{T}_sigma{i}{j}_{B}T.csv"
        data = np.loadtxt(f"{datapath}/{filename}", 
                          dtype="float", 
                          comments="#", 
                          delimiter=',', 
                          skiprows=1)
        x_data.append(data[:,0])
        y_data.append(data[:,1])

    x_data_r = np.concatenate(x_data)
    y_data_r = np.concatenate(y_data)

    # Handle the imaginary part
    x_data, y_data = [], []
    i = 2
    for j in ["r", "l"]:
        filename = f"post{T}_sigma{i}{j}_{B}T.csv"
        data = np.loadtxt(f"{datapath}/{filename}", 
                          dtype="float", 
                          comments="#", 
                          delimiter=',', 
                          skiprows=1)
        x_data.append(data[:,0])
        y_data.append(data[:,1])

    x_data_i = np.concatenate(x_data)
    y_data_i = np.concatenate(y_data)

    # Estimate initial value of omega_c
    e = 1.602e-19
    me = 9.109e-31
    omega_c_zero = 1e-12*e*B/(2*me) # Cyclotron frequency in rad.THz

    pars = Parameters()
    pars.add("gamma", value = 1e0, vary=vary["gamma"])
    pars.add("omega_pn", value = 1e4, vary=vary["omega_pn"])
    pars.add("omega_ps", value = 1e3, vary=vary["omega_ps"])
    pars.add("omega_c", value = omega_c_zero,  vary=vary["omega_c"])


    out = minimize(compute_diff, pars, args=(x_data_r, x_data_i, y_data_r, y_data_i))

    gamma = out.params["gamma"].value # Gamma in rad.THz
    omega_pn = out.params["omega_pn"].value # OmegaPN in rad**2.THz**2
    omega_ps = out.params["omega_ps"].value
    omega_c = out.params["omega_c"].value # Omegac in rad.THz

    print(f"B:{B} T; T:{T} K; Params are:\nGamma:{gamma}\nomega_pn:{omega_pn}\nomega_ps:{omega_ps}\nomega_c:{omega_c}\n#################################")

    return out

# Run the case and create the results
results = dict()
for T in temperature:
    if not T in results:
        results[T] = dict()
    for B in field:
        out = fitOneCase(T, B)
        results[T][B] = out

## Save the fitting result
fullpath = os.path.relpath(__file__)
dirname, fname = os.path.split(fullpath)
project_root = dirname + "/../../"
datapath = project_root+"data_processing/post"

with open(f'{datapath}/all_fits.pickle', 'wb') as f:
    pickle.dump(results, f)
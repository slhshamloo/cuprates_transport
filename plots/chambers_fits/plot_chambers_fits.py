import os, re
import numpy as np
import matplotlib as mpl
from copy import deepcopy
from matplotlib import pyplot as plt
from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.conductivity import Conductivity


mpl.rcdefaults()
# mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.size'] = 24
# mpl.rcParams['font.family'] = "serif"
# mpl.rcParams['font.serif'] = ["cm"]
mpl.rcParams['axes.labelsize'] = 24
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['xtick.direction'] = "in"
mpl.rcParams['ytick.direction'] = "in"
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['ytick.major.width'] = 1.0
mpl.rcParams['axes.linewidth'] = 1.0
# Output Type 3 (Type3) or Type 42 (TrueType)
# TrueType allows editing the text in illustrator
mpl.rcParams['pdf.fonttype'] = 3


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


def get_init_params():
    return {
        "band_name": "Nd-LSCO",
        "a": 3.75,
        "b": 3.75,
        "c": 13.2,
        "energy_scale": 160,
        "band_params":{"mu":-0.758, "t": 1, "tp":-0.12,
                       "tpp":0.06, "tz": 0.07},
        "res_xy": 20,
        "res_z": 5,
        "N_time": 1000,
        "Bamp": 45,
        "gamma_0": 8.0, # 12.595,
        "gamma_k": 3e3, # 63.823,
        "power": 6.2,
        # "gamma_dos_max": 50,
        "march_square": True
    }


def load_fit(sample, field):
    parameter_values = {}
    parameter_errors = {}
    with open(os.path.dirname(os.path.relpath(__file__))
              + f"/params/{sample}_{field}T.dat", 'r') as f:
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


def generate_chambers_fit(sample, field, omegas, bypass_fit=False,
                          init_params=get_init_params()):
    parameter_values, _ = load_fit(sample, field)
    params = deepcopy(init_params)
    params['Bamp'] = field
    if not bypass_fit:
        for parmeter_key, parameter_value in parameter_values.items():
            if parmeter_key in params:
                params[parmeter_key] = parameter_value
            else:
                params['band_params'][parmeter_key] = parameter_value
    band_obj = BandStructure(**params)
    band_obj.runBandStructure()
    cond_obj = Conductivity(band_obj, **params)
    cond_obj.runTransport()
    sigma = np.empty_like(omegas, dtype=complex)
    for i, omega in enumerate(omegas):
        setattr(cond_obj, "omega", omega)
        cond_obj.chambers_func()
        sigma[i] = (cond_obj.sigma[0, 0] + 1j*cond_obj.sigma[0, 1]
                    ).conjugate() * 1e-5
    return sigma


def save_fit_output_data(sample, field, omegas):
    sigma = generate_chambers_fit(sample, field, 2*np.pi*omegas)
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


def generate_figure(figsize=(9.2, 5.6)):
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    for ax in axs:
        ax.tick_params(axis='x', which='major', pad=7)
        ax.tick_params(axis='y', which='major', pad=8)
    axs[0].set_ylabel(r"$\mathrm{Re}(\sigma_{ll//rr})$",
                      labelpad=8)
    axs[1].set_ylabel(r"$\mathrm{Im}(\sigma_{ll//rr})$",
                      labelpad=8)
    axs[1].set_xlabel(r"Frequency (THz)", labelpad=8)
    fig.align_ylabels()
    return fig, axs


def plot_chambers_fit(paper, sample, field, bypass_fit=False, save_fig=False):
    fig, axs = generate_figure()
    omega_data, sigma_data = load_data(paper, sample, field)
    if bypass_fit:
        omega_fit = np.linspace(omega_data.min(), omega_data.max(), 50)
        sigma_fit = generate_chambers_fit(sample, field, 2*np.pi*omega_fit,
                                          bypass_fit=True)
    else:
        omega_fit, sigma_fit = load_fit_output_data(sample, field)

    axs[0].plot(omega_fit, sigma_fit.real, label="Fit",
                lw=2, color='black', ls='--')
    axs[1].plot(omega_fit, sigma_fit.imag, label="Fit",
                lw=2, color='black', ls='--')
    axs[0].plot(omega_data[omega_data<0], sigma_data.real[omega_data<0],
                label='Data', lw=3, color='red', ls='-')
    axs[0].plot(omega_data[omega_data>0], sigma_data.real[omega_data>0],
                lw=3, color='red', ls="-")
    axs[1].plot(omega_data[omega_data<0], sigma_data.imag[omega_data<0],
                label='Data', lw=3, color='red', ls='-')
    axs[1].plot(omega_data[omega_data>0], sigma_data.imag[omega_data>0],
                lw=3, color='red', ls='-')

    axs[1].legend(loc='upper left', frameon=False, fontsize=20)
    axs[0].text(0.05, 0.9, f"{sample}\n{field}T", transform=axs[0].transAxes,
                ha='left', va='top', fontsize=20)
    if bypass_fit:
        params = get_init_params()
    else:
        params = load_fit(sample, field)[0]
    axs[1].text(0.9, 0.05,
                fr"$\Gamma_0 = {float(params["gamma_0"]):.3}$"
                + "\n" + fr"$\Gamma_k = {float(params["gamma_k"]):.3}$"
                + "\n" + fr"$\nu = {float(params["power"]):.3}$",
                transform=axs[1].transAxes,
                ha="right", va="bottom", fontsize=20)

    fig.tight_layout()
    if save_fig:
        fig.savefig(os.path.dirname(os.path.relpath(__file__))
                    + f"/figures/Chambers_{sample}_{field}T.pdf",
                    bbox_inches='tight')
    plt.show()


def output_all_fits():
    for sample in ["OD17K"]:
        for field in [0, 4, 9, 14, 20, 31]:
            omega_data, _ = load_data("legros", sample, field)
            omega_fit = np.linspace(omega_data.min(), omega_data.max(), 50)
            save_fit_output_data(sample, field, omega_fit)


def plot_all_fits():
    for sample in ["OD17K"]:
        for field in [0, 4, 9, 14, 20, 31]:
            plot_chambers_fit("legros", sample, field, save_fig=True)


if __name__ == "__main__":
    output_all_fits()
    plot_all_fits()
    # plot_chambers_fit("legros", "OD17K", 9, bypass_fit=True)

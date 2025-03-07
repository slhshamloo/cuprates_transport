import os, re
import numpy as np
import matplotlib as mpl
from copy import deepcopy
from matplotlib import pyplot as plt
from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.conductivity import Conductivity

import file_operations
import defaults


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

load_data = file_operations.load_data
load_fit = file_operations.load_fit
generate_chambers_fit = file_operations.generate_chambers_fit
load_fit_output_data = file_operations.load_fit_output_data
save_fit_output_data = file_operations.save_fit_output_data
get_init_params = defaults.get_init_params



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

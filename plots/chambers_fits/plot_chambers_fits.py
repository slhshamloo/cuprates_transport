import os, re
import numpy as np
import matplotlib as mpl
from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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

generate_chambers_fit = file_operations.generate_chambers_fit
load_data = file_operations.load_data
load_fit = file_operations.load_fit
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


def plot_chambers_fit(paper, sample, field, bypass_fit=False,
                      save_fig=False, extra_info=""):
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
        params = load_fit(sample, [field])[0]
    axs[1].text(0.98, 0.05,
                fr"$\Gamma_0 = {float(params['gamma_0']):.3g}$, "
                + fr"$\Gamma_k = {float(params['gamma_k']):.3g}$"
                + "\n" + fr"$\nu = {float(params['power']):.3g}$, "   
                + fr"$t = {float(params['energy_scale']):.3g}$",      
                transform=axs[1].transAxes,
                ha="right", va="bottom", fontsize=20)

    fig.tight_layout()
    if save_fig:
        fig.savefig(os.path.dirname(os.path.relpath(__file__))
                    + f"/figures/Chambers_{sample}_{field}T{extra_info}.pdf",
                    bbox_inches='tight')
    plt.show()

def sigma_from_parameters(field, omegas, parameter_values):
    """Generate sigma from the specified parameters on the omega range, at a given field"""

    params = deepcopy(parameter_values)
    params['Bamp'] = -field

    band_obj = BandStructure(**params)
    band_obj.runBandStructure()
    cond_obj = Conductivity(band_obj, **params)
    cond_obj.runTransport()

    sigma = np.empty_like(omegas, dtype=complex)
    for i, omega in enumerate(omegas):
        setattr(cond_obj, "omega", omega)
        cond_obj.chambers_func()
        sigma[i] = (cond_obj.sigma[0, 0] + 1j*cond_obj.sigma[0, 1]
                    ) * 1e-5

    return sigma


def plotting(omega_data, sigma_data, omega_fit, sigma_fit, sample, field, params):
    """Generate the figure object from all the relevant values already provided"""

    fig, axs = generate_figure()

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
    axs[1].text(0.98, 0.05,
                fr"$\Gamma_0 = {float(params['gamma_0']):.3g}$, "
                + fr"$\Gamma_k = {float(params['gamma_k']):.3g}$"
                + "\n" + fr"$\nu = {float(params['power']):.3g}$, "   
                + fr"$t = {float(params['energy_scale']):.3g}$",      
                transform=axs[1].transAxes,
                ha="right", va="bottom", fontsize=20)

    fig.tight_layout()

    return fig, axs

def from_parameters(paper, sample, fields, extra_info, fixed_non_override, parameter_values=None):
    """Generate a multipage pdf for the plots of a multifield case"""

    # Save the plots as a multipage pdf
    fieldString = '-'.join([str(f) for f in fields])
    with PdfPages(os.path.dirname(os.path.relpath(__file__))
                    + f"/figures/Chambers_{sample}_{fieldString}T{extra_info}.pdf") as pdf:
        ## Load fit parameter values
        if parameter_values is None:
            # Generate values from fit params
            parameter_values, _ = load_fit(sample, fields, extra_info)
        
        # Create params object for Boltzmann transport
        params = defaults.get_init_params()
        for parameter_key, parameter_value in parameter_values.items():
            if parameter_key in params:
                params[parameter_key] = parameter_value
            else:
                params['band_params'][parameter_key] = parameter_value

        # Handle plots with some fixed parameters during fitting
        for parameter_key, parameter_value in fixed_non_override.items():
            if parameter_key in params:
                params[parameter_key] = parameter_value



        for field in fields:
            ## Load and generate plotted data
            # Load the data for the given field
            omega_data, sigma_data = load_data(paper, sample, field)

            # Generate the fitting results
            omega_fit = np.linspace(omega_data.min(), omega_data.max(), 50)
            field_params = deepcopy(params)
            field_params['Bamp'] = field
            sigma_fit = sigma_from_parameters(field, 2*np.pi*omega_fit, field_params)

            ## Plot the loaded and generated data
            fig, _ = plotting(omega_data, sigma_data, omega_fit, sigma_fit, sample, field, field_params)

            ## Save the figure as a new page in the pdf
            pdf.savefig(fig, bbox_inches='tight')

def output_all_fits():
    for sample in ["OD17K"]:
        for field in [0, 4, 9, 14, 20, 31]:
            omega_data, _ = load_data("legros", sample, field)
            omega_fit = np.linspace(omega_data.min(), omega_data.max(), 50)
            save_fit_output_data(sample, field, [0, 4, 9, 14, 20, 31], omega_fit)


def plot_all_fits():
    for sample in ["OD17K"]:
        for field in [0, 4, 9, 14, 20, 31]:
            plot_chambers_fit("legros", sample, field, save_fig=True)


if __name__ == "__main__":
    output_all_fits()
    plot_all_fits()
    # plot_chambers_fit("legros", "OD17K", 9, bypass_fit=True)

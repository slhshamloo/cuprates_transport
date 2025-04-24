import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tqdm
from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.conductivity import Conductivity
from scipy.constants import Boltzmann, hbar, elementary_charge, physical_constants, electron_mass
meV = physical_constants["electron volt"][0] * 1e-3 # 1 meV in Joule


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


def get_lsco_params():
    return {
        "band_name": "Nd-LSCO",
        "a": 3.75,
        "b": 3.75,
        "c": 13.2,
        "energy_scale": 80,
        "band_params":{"mu":-0.758, "t": 1, "tp":-0.12,
                       "tpp":0.06, "tz": 0.07},
        "res_xy": 20,
        "res_z": 5,
        "N_time": 1000,
        "Bamp": 30,
        "gamma_0": 10,
        "gamma_k": 50,
        "power": 6,
        "march_square": True
    }


def get_free_params():
    return {
        "band_name": "Free electrons",
        "a": 3.75,
        "b": 3.75,
        "c": 13.2,
        "energy_scale": 160,
        "band_params":{"mu": -0.5, "t": 1},
        "tight_binding": "mu + t*((kx)**2+(ky)**2)",
        "res_xy": 20,
        "res_z": 5,
        "N_time": 1000,
        "Bamp": 30,
        "gamma_0": 7.0,
        "gamma_k": 1e3,
        "power": 4,
        "omega": 1.0,
    }


def compute_sigmas(omegas, params, bandObject):
    sigma_xx = np.empty_like(omegas, dtype=complex)
    sigma_xy = np.empty_like(omegas, dtype=complex)
    condObject = Conductivity(bandObject, **params)
    condObject.runTransport()
    for i, omega in tqdm.tqdm(enumerate(omegas)):
        condObject.omega = omega
        condObject.chambers_func()
        sigma_xx[i] = condObject.sigma[0,0]
        sigma_xy[i] = condObject.sigma[0,1]
    return sigma_xx, sigma_xy


def save_data(omegas, sigma_xx, sigma_xy, filename):
    data = np.column_stack((omegas, sigma_xx, sigma_xy))
    np.savetxt(filename, data, delimiter=",", header="omega,sigma_xx,sigma_xy")


def vary_value(band, value_name, value_label, values,
               omegas, params, bandObject, extra_label=""):
    for value in tqdm.tqdm(values):
        params[value_name] = value
        sigma_xx, sigma_xy = compute_sigmas(omegas, params, bandObject)
        save_data(omegas, sigma_xx, sigma_xy, "user_data/vary/"
                  + f"{band}_{value_label}_{value}{extra_label}.csv")


def calculate_range(ranges_dict, band, params=get_lsco_params(), label_dict={},
                    extra_labels_dict={}, omegas=np.linspace(-20, 20, 100)):
    bandObject = BandStructure(**params)
    bandObject.runBandStructure()
    for vary_param in ranges_dict:
        params_copy = params.copy()
        value_label = (label_dict[vary_param] if vary_param in label_dict
                       else vary_param)
        vary_value(band, vary_param, value_label, ranges_dict[vary_param],
                   omegas, params_copy, bandObject,
                   extra_labels_dict.get(vary_param, ""))


def generate_figure_single(sigma_substript="xx", figsize=(9.2, 5.6)):
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    for ax in axs:
        ax.tick_params(axis='x', which='major', pad=7)
        ax.tick_params(axis='y', which='major', pad=8)
    axs[0].set_ylabel(fr"$\mathrm{{Re}}(\sigma_{{{sigma_substript}}})$",
                      labelpad=8)
    axs[1].set_ylabel(fr"$\mathrm{{Im}}(\sigma_{{{sigma_substript}}})$",
                      labelpad=8)
    axs[1].set_xlabel(fr"$\omega\ (\mathrm{{THz}})$", labelpad=8)
    fig.align_ylabels()
    return fig, axs


def generate_figure_double(sigma_substript="xx", figsize=(12, 8)):
    fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey="row")
    for ax in axs.flat:
        ax.tick_params(axis='x', which='major', pad=7)
        ax.tick_params(axis='y', which='major', pad=8)
    axs[0, 0].set_ylabel(fr"$\mathrm{{Re}}(\sigma_{{{sigma_substript}}})$",
                         labelpad=8)
    axs[1, 0].set_ylabel(fr"$\mathrm{{Im}}(\sigma_{{{sigma_substript}}})$",
                         labelpad=8)
    fig.supxlabel(fr"$\omega\ (\mathrm{{THz}})$", y=0.08)
    fig.align_ylabels()
    return fig, axs


def load_data(band, value_label, value, extra_label=""):
    return np.loadtxt(
        f"user_data/vary/{band}_{value_label}_{value}{extra_label}.csv",
        delimiter=",", skiprows=1, unpack=True, dtype=complex)


def plot_data(axs, band, value_label, value_latex, values,
              convert_to_lr=True, extra_label="", **kwargs):
    for i, value in enumerate(values):
        omega, sigma_xx, sigma_xy = load_data(band, value_label, value,
                                              extra_label)
        if convert_to_lr:
            sigma = sigma_xx + 1j * sigma_xy
        else:
            sigma = sigma_xx
        axs[0].plot(omega.real, sigma.real * 1e-5,
                    c=mpl.cm.viridis(i/(len(values) - 1)),
                    label=f"${value_latex} = {value}$", **kwargs)
        axs[1].plot(omega.real, -sigma.imag * 1e-5,
                    c=mpl.cm.viridis(i/(len(values) - 1)),
                    label=f"${value_latex} = {value}$", **kwargs)


def plot_free_vary_gamma_k():
    fig, axs = generate_figure_single(figsize=(10, 8))
    fig.suptitle("Free electrons, No Magnetic Field", y=0.94)
    plot_data(axs, "free", "gamma_k", "\\Gamma_k", [0, 10, 25, 50, 100],
              convert_to_lr=False, lw=2)
    axs[0].legend(loc="upper right", fontsize=20, frameon=False,
                  ncols=2, columnspacing=1.0)
    axs[1].text(0.03, 0.9, fr"$\Gamma_0 = 10$" + "\n" + fr"$\nu = 6$",
                transform=axs[1].transAxes, ha="left", va="top")
    plt.tight_layout()
    fig.savefig("user_plots/comparisons/free_vary_gamma_k.pdf")
    plt.show()


def plot_free_vary_power():
    fig, axs = generate_figure_single(figsize=(10, 8))
    fig.suptitle("Free electrons, No Magnetic Field", y=0.94)
    plot_data(axs, "free", "power", "\\nu", [2, 6, 12, 18],
              convert_to_lr=False, lw=2)
    axs[0].legend(loc="upper right", fontsize=20, frameon=False,
                  ncols=2, columnspacing=1.0)
    axs[1].text(0.03, 0.9, fr"$\Gamma_0 = 10$" + "\n" + fr"$\Gamma_k = 20$",
                transform=axs[1].transAxes, ha="left", va="top")
    plt.tight_layout()
    fig.savefig("user_plots/comparisons/free_vary_power.pdf")
    plt.show()


def plot_lsco_vary_gamma_k(gamma_k_vals=[0, 10, 25, 50, 100]):
    fig, axs = generate_figure_single(figsize=(10, 8))
    fig.suptitle("LSCO, No Magnetic Field", y=0.94)
    plot_data(axs, "NdLSCO", "gamma_k", "\\Gamma_k", gamma_k_vals,
              convert_to_lr=False, lw=2)
    axs[0].legend(loc="upper right", fontsize=20, frameon=False,
                  ncols=2, columnspacing=1.0)
    axs[1].text(0.03, 0.9, fr"$\Gamma_0 = 10$" + "\n" + fr"$\nu = 6$",
                transform=axs[1].transAxes, ha="left", va="top")
    plt.tight_layout()
    fig.savefig("user_plots/comparisons/NdLSCO_vary_gamma_k.pdf")
    plt.show()


def plot_lsco_vary_power(nu_vals=[2, 6, 12, 18]):
    fig, axs = generate_figure_single(figsize=(10, 8))
    fig.suptitle("LSCO, No Magnetic Field", y=0.94)
    plot_data(axs, "NdLSCO", "power", "\\nu", nu_vals,
              convert_to_lr=False, lw=2)
    axs[0].legend(loc="upper right", fontsize=20, frameon=False,
                  ncols=2, columnspacing=1.0)
    axs[1].text(0.03, 0.9, fr"$\Gamma_0 = 10$" + "\n" + fr"$\Gamma_k = 20$",
                transform=axs[1].transAxes, ha="left", va="top")
    plt.tight_layout()
    fig.savefig("user_plots/comparisons/NdLSCO_vary_power.pdf")
    plt.show()


def plot_free_iso_vs_aniso_vary_B(B_vals=[0, 500, 1000, 1500, 2000]):
    fig, axs = generate_figure_double(figsize=(12, 8), sigma_substript="ll/rr")
    axs[0, 0].set_title("Free electrons, Isotropic", pad=10)
    axs[0, 1].set_title("Free electrons, Anisotropic", pad=10)

    plot_data(axs[:, 0], "free_iso", "B", "B", B_vals, lw=2)
    plot_data(axs[:, 1], "free_aniso", "B", "B", B_vals, lw=2)

    axs[1, 0].legend(loc="lower right", fontsize=20, frameon=False,
                     labelspacing=0.2, borderpad=0.2, borderaxespad=0.2)
    axs[0, 1].text(0.97, 0.95, fr"$\Gamma_0 = 10$" + "\n"
                   + fr"$\Gamma_k = 20$" + "\n" + fr"$\nu = 6$",
                   transform=axs[0, 1].transAxes, ha="right", va="top",
                   fontsize=20)

    plt.tight_layout()
    fig.savefig("user_plots/comparisons/free_iso_vs_aniso_vary_field.pdf")
    plt.show()


def plot_free_iso_vs_aniso_vary_gamma_0(gamma_0_vals=[10, 15, 20, 25]):
    fig, axs = generate_figure_double(figsize=(12, 8))
    axs[0, 0].set_title("Free electrons, Isotropic", pad=10)
    axs[0, 1].set_title("Free electrons, Anisotropic", pad=10)

    plot_data(axs[:, 0], "free_iso", "gamma_0", "\\Gamma_0",
              gamma_0_vals, lw=2)
    plot_data(axs[:, 1], "free_aniso", "gamma_0", "\\Gamma_0",
              gamma_0_vals, lw=2)
    
    axs[0, 0].legend(loc="upper right", fontsize=20, frameon=False, 
                     borderpad=0.2, borderaxespad=0.3)
    axs[0, 1].text(0.97, 0.95, fr"$B = 0$" + "\n"
                   + fr"$\Gamma_k = 20$" + "\n" + fr"$\nu = 6$",
                   transform=axs[0, 1].transAxes, ha="right", va="top",
                   fontsize=20)

    plt.tight_layout()
    fig.savefig("user_plots/comparisons/free_iso_vs_aniso_vary_gamma_0.pdf")
    plt.show()


def plot_lsco_iso_vs_aniso_vary_B(gamma_k=20, nu=6,
                                  B_vals=[0, 50, 100, 150, 200]):
    fig, axs = generate_figure_double(figsize=(12, 8), sigma_substript="ll/rr")
    axs[0, 0].set_title("Nd-LSCO, Isotropic", pad=10)
    axs[0, 1].set_title("Nd-LSCO, Anisotropic", pad=10)

    extra_label = f"_gamma_k_{gamma_k}_nu_{nu}"
    plot_data(axs[:, 0], "NdLSCO_iso", "B", "B", B_vals, lw=2)
    plot_data(axs[:, 1], "NdLSCO_aniso", "B", "B", B_vals,
              extra_label=extra_label, lw=2)

    axs[1, 0].legend(loc="lower right", fontsize=20, frameon=False,
                     labelspacing=0.1, borderpad=0.2, borderaxespad=0.2)
    axs[0, 1].text(0.97, 0.95, fr"$\Gamma_0 = 10$" + "\n"
                   + fr"$\Gamma_k = {gamma_k}$" + "\n" + fr"$\nu = {nu}$",
                   transform=axs[0, 1].transAxes, ha="right", va="top",
                   fontsize=20)

    plt.tight_layout()
    fig.savefig("user_plots/comparisons/NdLSCO_iso_vs_aniso_vary_field"
                + extra_label + ".pdf")
    plt.show()


def plot_lsco_iso_vs_aniso_vary_gamma_0(gamma_0_vals=[10, 15, 20, 25]):
    fig, axs = generate_figure_double(figsize=(12, 8))
    axs[0, 0].set_title("Nd-LSCO, Isotropic", pad=10)
    axs[0, 1].set_title("Nd-LSCO, Anisotropic", pad=10)

    plot_data(axs[:, 0], "NdLSCO_iso", "gamma_0", "\\Gamma_0",
              gamma_0_vals, lw=2)
    plot_data(axs[:, 1], "NdLSCO_aniso", "gamma_0", "\\Gamma_0",
              gamma_0_vals, lw=2)
    
    axs[0, 0].legend(loc="upper right", fontsize=20, frameon=False, 
                     borderpad=0.2, borderaxespad=0.3)
    axs[0, 1].text(0.97, 0.95, fr"$B = 0$" + "\n"
                   + fr"$\Gamma_k = 20$" + "\n" + fr"$\nu = 6$",
                   transform=axs[0, 1].transAxes, ha="right", va="top",
                   fontsize=20)

    plt.tight_layout()
    fig.savefig("user_plots/comparisons/NdLSCO_iso_vs_aniso_vary_gamma_0.pdf")
    plt.show()


def plot_vary_general(
        value_label, value_latex, values, band, title=None,
        extra_label_data="", extra_label_plot="", figsize=(10, 8),
        non_varying_params=[], non_varying_latex=[], **kwargs):
    fig, axs = generate_figure_single(figsize=figsize, sigma_substript="ll/rr")
    title = title if title else band
    axs[0].set_title(title)
    plot_data(axs, band, value_label, value_latex, values,
              extra_label=extra_label_data, **kwargs)
    axs[1].legend(loc="upper left", fontsize=20, frameon=False, ncols=2,
                  labelspacing=0.3, borderpad=0.3, borderaxespad=0.3,
                  columnspacing=0.5)
    non_varying_params_text = "\n".join(
        [f"${non_varying_latex[i]} = {non_varying_params[i]}$"
         for i in range(len(non_varying_params))])
    axs[1].text(0.97, 0.05, non_varying_params_text, fontsize=20,
                transform=axs[1].transAxes, ha="right", va="bottom")
    fig.tight_layout()
    fig.savefig("user_plots/comparisons/"
                f"{band}{extra_label_plot}_vary_{value_label}.pdf")
    plt.show()


def plot_vary_general_all(ranges, band, title, params=get_lsco_params(),
                          extra_label_data="", extra_label_plot="",
                          figsize=(10, 8), **kwargs):
    for param_name in ranges:
        non_varying_params = [params[param] for param in ranges
                              if param != param_name]
        non_varying_latex = [latex_dict[param] for param in ranges
                             if param != param_name]
        plot_vary_general(
            param_name, latex_dict[param_name], ranges[param_name], band,
            title=title, extra_label_data=extra_label_data,
            extra_label_plot=extra_label_plot, figsize=figsize,
            non_varying_params=non_varying_params,
            non_varying_latex=non_varying_latex, **kwargs)


ranges_dict = {
    # "Bamp": [0, 50, 100, 150, 200],
    "gamma_0": [5, 10, 15, 20, 25],
    "gamma_k": [0, 10, 25, 50, 100],
    "power": [2, 6, 10, 14, 18],
}

labels_dict = {
    "Bamp": "B"
}

latex_dict = {
    "Bamp": "B",
    "gamma_0": "\\Gamma_0",
    "gamma_k": "\\Gamma_k",
    "power": "\\nu"
}


def main():
    calculate_range(ranges_dict, "LSCO", label_dict=labels_dict,
                    extra_labels_dict={param_name: "_B_30T"
                                       for param_name in ranges_dict})
    plot_vary_general_all(ranges_dict, "LSCO", "LSCO, B = 30 T", lw=2,
                          extra_label_data="_B_30T", extra_label_plot="_B_30T")


if __name__ == "__main__":
    main()

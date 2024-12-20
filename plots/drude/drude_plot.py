import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0

epsilon_0 = epsilon_0 * 1e7

## Defaults ########################################\

mpl.rcdefaults()
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.size'] = 16
mpl.rcParams['font.family'] = "serif"
mpl.rcParams['font.serif'] = ["cm"]
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
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


## Load Data ######################################
def load_data(paper, sample, mag_field):
    omega_l, sigma1l = np.loadtxt(
        f"data/{paper}/{sample}_sigma1l_{mag_field}T.csv",
        skiprows=1, delimiter=',', unpack=True)
    omega_r, sigma1r = np.loadtxt(
        f"data/{paper}/{sample}_sigma1r_{mag_field}T.csv",
        skiprows=1, delimiter=',', unpack=True)
    omega_l_2, sigma2l = np.loadtxt(
        f"data/{paper}/{sample}_sigma2l_{mag_field}T.csv",
        skiprows=1, delimiter=',', unpack=True)
    omega_r_2, sigma2r = np.loadtxt(
        f"data/{paper}/{sample}_sigma2r_{mag_field}T.csv",
        skiprows=1, delimiter=',', unpack=True)
    sigma2l = np.interp(omega_l, omega_l_2, sigma2l)
    sigma2r = np.interp(omega_r, omega_r_2, sigma2r)
    omega = np.concatenate((omega_l, omega_r))
    sigma1 = np.concatenate((sigma1l, sigma1r))
    sigma2 = np.concatenate((sigma2l, sigma2r))
    sigma = sigma1 + 1j * sigma2
    return omega, sigma


## Generate Figure ################################
def generate_figure(omega_min, omega_l_max, sigma1_min, size = (8, 8)):
    fig, axs = plt.subplots(2, 1, figsize=size, sharex=True)
    for i, ax in enumerate(axs):
        # ax.axvline(x=0, ls =':', c ="k", lw=1.0)
        ax.tick_params(axis='x', which='major', pad=7)
        ax.tick_params(axis='y', which='major', pad=8)
        ax.set_ylabel(
            fr"$\sigma_{i + 1} (\mathrm{{m}}\Omega^-1\mathrm{{cm}}^-1)$",
            labelpad=8)
    sigma_l_pos = omega_l_max - (omega_l_max - omega_min) * 0.2
    axs[0].text(sigma_l_pos, sigma1_min, r"$\sigma_l$", ha="center")
    axs[0].text(-sigma_l_pos, sigma1_min, r"$\sigma_r$", ha="center")
    axs[1].set_xlabel("Frequency (THz)", labelpad=8)
    fig.align_ylabels()
    return fig, axs


## Plot Data ######################################
def fitting_model(omega, omega_pn_sq, omega_ps_sq, omega_c, gamma):
    return 1j * epsilon_0 * (omega_pn_sq / (omega - omega_c + 1j * gamma)
                             + omega_ps_sq / omega)


def get_fit_omega_sigma(omega, pars):
    omega_l_max = np.max(omega[omega < 0])
    omega_r_min = np.min(omega[omega > 0])
    omega1_fit = np.linspace(omega.min(), omega.max(), 1000)
    omega2_fit = np.concatenate((np.linspace(omega.min(), omega_l_max, 1000),
                                 np.linspace(omega_r_min, omega.max(), 1000)))
    sigma1_fit = np.real(fitting_model(
        omega1_fit, pars["omega_pn_sq"].value, pars["omega_ps_sq"].value,
        pars["omega_c"].value, pars["gamma"].value))
    sigma2_fit = np.imag(fitting_model(
        omega2_fit, pars["omega_pn_sq"].value, pars["omega_ps_sq"].value,
        pars["omega_c"].value, pars["gamma"].value))
    return omega1_fit, sigma1_fit, omega2_fit, sigma2_fit


def plot_data(axs, omega, sigma, fit, field, color):
    omega1_fit, sigma1_fit, omega2_fit, sigma2_fit = \
        get_fit_omega_sigma(omega, fit.params)
    data_lines = []
    fit_lines = []
    for ax, omega_fit, sigma_fit, sigma_plot in zip(
            axs, [omega1_fit, omega2_fit], [sigma1_fit, sigma2_fit],
            [np.real(sigma), np.imag(sigma)]):
        for omega_data, sigma_data in zip(
                [omega[omega < 0], omega[omega > 0]],
                [sigma_plot[omega < 0], sigma_plot[omega > 0]]):
            data_lines.append(*ax.plot(
                omega_data, sigma_data, c=color, lw=4,
                label=r"$\omega_{pn}^2="
                    fr"{fit.params['omega_pn_sq'].value:.2}, "
                    r"\omega_{ps}^2="
                    f"{fit.params['omega_ps_sq'].value:.2},$\n"
                    fr"$\omega_c={fit.params['omega_c'].value:.2}, "
                    fr"\Gamma={fit.params['gamma'].value:.2}$"))
        fit_lines.append(*ax.plot(omega_fit, sigma_fit, c=color,
                                  lw=2, ls="--", label=f"{field} T"))
    return data_lines, fit_lines


## Plot Post ####################################
def plot_post(temperature):
    omegas = []
    sigmas = []
    for field in [0, 9, 20, 31]:
        omega, sigma = load_data("post", f"post{temperature}", field)
        omegas.append(omega)
        sigmas.append(sigma)
    sigma1_min = min(sigma.real.min() for sigma in sigmas)
    sigma2_min = min(sigma.imag.min() for sigma in sigmas)
    sigma2_max = max(sigma.imag.max() for sigma in sigmas)
    omega_min = min(omega.min() for omega in omegas)
    omega_max = max(omega.max() for omega in omegas)
    omega_l_max = np.max(omega[omega < 0])
    fig, axs = generate_figure(omega_min, omega_l_max, sigma1_min)

    fields = [0, 9, 20, 31]
    colors = ['r', 'g', 'b', 'm']
    with open("user_data/post_fits.pkl", "rb") as f:
        fits = pickle.load(f)[f"post_{temperature}K"]
        fits = [fits[field] for field in fields]
    lines = []
    artists = []
    for fit, omega, sigma, field, color, in zip(
            fits, omegas, sigmas, fields, colors):
        data_lines, fit_lines = plot_data(axs, omega, sigma, fit, field, color)
        lines.append(fit_lines[0])
        artists.append(data_lines[2])

    axs[0].legend(handles=lines, loc="lower center")
    for artist, color, pos_scale, ha, xpos in zip(
            artists, colors, [0.85, 0.6, 0.25, 0.0],
            ["left", "left", "right", "right"],
            [omega_min * 1.2, omega_min * 1.2,
             omega_max * 1.2, omega_max * 1.2]):
        axs[1].text(
            xpos, sigma2_min + (sigma2_max - sigma2_min) * pos_scale,
            artist.get_label(), color=color, ha=ha)

    axs[1].set_xlim(omega_min * 1.25, omega_max * 1.25)
    fig.tight_layout()
    return fig, axs


def main():
    for temperature in [35, 40, 45, 50]:
        fig, axs = plot_post(temperature)
        fig.savefig(f"user_plots/post_fit_{temperature}K.pdf",
                    bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()


# plot_post(50)
# plt.show()

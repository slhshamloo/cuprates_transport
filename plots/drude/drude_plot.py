import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0

epsilon_0 = epsilon_0 * 1e7

## Defaults ########################################\

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
def generate_figure():
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ax in axs:
        # ax.axvline(x=0, ls =':', c ="k", lw=1.0)
        ax.tick_params(axis='x', which='major', pad=7)
        ax.tick_params(axis='y', which='major', pad=8)
    axs[0].set_ylabel(r"$\mathrm{Re}(\sigma)$", labelpad=8)
    axs[1].set_ylabel(r"$\mathrm{Im}(\sigma)$", labelpad=8)
    axs[1].set_xlabel(r"$\omega\ (\mathrm{{THz}})$", labelpad=8)
    axs[0].set_xlim(-2, 2)
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
    omega2_fit = np.concatenate(
        (np.linspace(omega.min(), omega_l_max, 1000),
         np.linspace(omega_r_min, omega.max(), 1000)))
    sigma1_fit = np.real(fitting_model(
        omega1_fit, pars["omega_pn_sq"].value, pars["omega_ps_sq"].value,
        pars["omega_c"].value, pars["gamma"].value))
    sigma2_fit = np.imag(fitting_model(
        omega2_fit, pars["omega_pn_sq"].value, pars["omega_ps_sq"].value,
        pars["omega_c"].value, pars["gamma"].value))
    return omega1_fit, sigma1_fit, omega2_fit, sigma2_fit


def plot_data(axs, omega, sigma, fit):
    omega1_fit, sigma1_fit, omega2_fit, sigma2_fit = \
        get_fit_omega_sigma(omega, fit.params)
    data_lines = []
    fit_lines = []
    for ax, omega_fit, sigma_fit, sigma_plot in zip(
            axs, [omega1_fit, omega2_fit], [sigma1_fit, sigma2_fit],
            [sigma.real, sigma.imag]):
        fit_lines.append(*ax.plot(omega_fit, sigma_fit, c='k',
                                  lw=3, ls='--', label="Fit"))
        for omega_data, sigma_data in zip(
                [omega[omega < 0], omega[omega > 0]],
                [sigma_plot[omega < 0], sigma_plot[omega > 0]]):
            data_lines.append(*ax.plot(omega_data, sigma_data, c='r',
                                       lw=3, ls='-', label="Data"))
    return sigma1_fit, data_lines, fit_lines

## Plot Post ####################################
def plot_post(temperature, field):
    omega, sigma = load_data("post", f"post{temperature}", field)
    fig, axs = generate_figure()

    with open("user_data/post_fits.pkl", "rb") as f:
        fit = pickle.load(f)[f"post_{temperature}K"][field]

    sigma1_fit, data_lines, fit_lines = plot_data(
        axs, omega, sigma, fit)
    axs[1].legend(handles=[fit_lines[1], data_lines[2]], loc="upper left",
                  fontsize=18, frameon=False, handletextpad=0.5)

    axs[0].text(-1.9, sigma1_fit.max(), "LSCO\n" + r"$p=0.16$",
                ha="left", va="top")
    axs[0].text(1.9, sigma1_fit.max(),
                fr"$T={temperature}\ K$" + '\n' + fr"$B={field}\ T$",
                ha="right", va="top")
    params_text = (fr"$\Gamma={fit.params['gamma'].value:.2}$" + '\n'
         + fr"$\omega_c={fit.params['omega_c'].value:.2}$" + '\n'
         + fr"$\omega_{{pn}}^2 = {fit.params['omega_pn_sq'].value:.2}$" + '\n'
         + fr"$\omega_{{ps}}^2 = {fit.params['omega_ps_sq'].value:.2}$")
    axs[1].text(1.8, sigma.imag.min(), params_text, fontsize=18,
                ha="right", va="bottom")
    fig.tight_layout()
    return fig, axs


# def main():
#     for temperature in [35, 40, 45, 50]:
#         for field in [0, 9, 20, 31]:
#             fig, axs = plot_post(temperature, field)
#             fig.savefig(f"user_plots/post_fit_{temperature}K_{field}T.pdf",
#                         bbox_inches="tight")
#             plt.close(fig)


# if __name__ == "__main__":
#     main()


plot_post(50, 0)
plt.show()

import pickle, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0
epsilon_0 = epsilon_0 * 1e7


# matplotlib settings
# reset defaults
mpl.rcdefaults()
# font
mpl.rcParams['font.size'] = 24
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['ytick.major.width'] = 1.0
mpl.rcParams['pdf.fonttype'] = 3
# plotting
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 20


# LaTeX labels
name_to_latex = {
    "omega_pn_sq": r"(\omega_{p,n}/2\pi)^2",
    "omega_ps_sq": r"(\omega_{p,s}/2\pi)^2",
    "omega_c": r"\omega_c/2\pi",
    "gamma": r"\Gamma/2\pi",
    "epsilon_inf": r"\epsilon_\infty"
}

# dopings
sample_to_doping = {
    "UD39K": 0.131,
    "OD36K": 0.202,
    "OD35K": 0.205,
    "OD17K": 0.244,
    "OD13K": 0.251,
    "NSC": 0.263
}


def generate_figure():
    fig = plt.figure(figsize=(10, 6))
    gs = mpl.gridspec.GridSpec(2, 2, width_ratios=[1, 0.2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[:, 1])
    ax3.axis('off')
    for ax in [ax1, ax2]:
        # ax.axvline(x=0, ls =':', c ="k", lw=1.0)
        ax.tick_params(axis='x', which='major', pad=7)
        ax.tick_params(axis='y', which='major', pad=8)
    ax1.set_ylabel(r"$\mathrm{Re}(\sigma)$")
    ax2.set_ylabel(r"$\mathrm{Im}(\sigma)$")
    ax2.set_xlabel(r"$\omega\ (\mathrm{{THz}})$")
    fig.align_ylabels()
    return fig, ax1, ax2, ax3


def load_data(paper, sample, field):
    omega_l, sigma1l = np.loadtxt(
        f"data/{paper}/{sample}_sigma1l_{field}T.csv",
        skiprows=1, delimiter=',', unpack=True)
    omega_r, sigma1r = np.loadtxt(
        f"data/{paper}/{sample}_sigma1r_{field}T.csv",
        skiprows=1, delimiter=',', unpack=True)
    omega_l_2, sigma2l = np.loadtxt(
        f"data/{paper}/{sample}_sigma2l_{field}T.csv",
        skiprows=1, delimiter=',', unpack=True)
    omega_r_2, sigma2r = np.loadtxt(
        f"data/{paper}/{sample}_sigma2r_{field}T.csv",
        skiprows=1, delimiter=',', unpack=True)
    sigma2l = np.interp(omega_l, omega_l_2, sigma2l)
    sigma2r = np.interp(omega_r, omega_r_2, sigma2r)
    omega = np.concatenate((omega_l, omega_r))
    sigma1 = np.concatenate((sigma1l, sigma1r))
    sigma2 = np.concatenate((sigma2l, sigma2r))
    sigma = sigma1 + 1j * sigma2
    return omega, sigma


def load_fit(fit_path, sample, field):
    with open(fit_path, 'rb') as f:
        fit = pickle.load(f)[f"legros_{sample}"][field]
    return fit


def fitting_model(omega, omega_pn_sq, omega_ps_sq, omega_c,
                  gamma, epsilon_inf=1):
    return 2j * np.pi * epsilon_0 * (
        omega_pn_sq / (omega - omega_c + 1j * gamma) + omega_ps_sq / omega
        - omega * (epsilon_inf - 1))


def get_fit_omega_sigma(omega_min, omega_max, pars):
    omega1_fit = np.linspace(-omega_max, omega_max, 1000)
    omega2_fit = np.concatenate((
        np.linspace(-omega_max, -omega_min, 500),
        np.linspace(omega_min, omega_max, 500)))
    sigma1_fit = fitting_model(
        omega1_fit, pars[f"omega_pn_sq"].value, pars[f"omega_ps_sq"].value,
        pars[f"omega_c"].value, pars[f"gamma"].value, pars["epsilon_inf"]).real
    sigma2_fit = fitting_model(
        omega2_fit, pars[f"omega_pn_sq"].value, pars[f"omega_ps_sq"].value,
        pars[f"omega_c"].value, pars[f"gamma"].value, pars["epsilon_inf"]).imag
    return omega1_fit, sigma1_fit, omega2_fit, sigma2_fit


def plot_data(ax1, ax2, omega, sigma, fit):
    omega1_fit, sigma1_fit, omega2_fit, sigma2_fit = \
        get_fit_omega_sigma(0.2, 2.0, fit.params)
    for ax, omega_fit, sigma_fit, sigma_data in zip(
            [ax1, ax2], [omega1_fit, omega2_fit], [sigma1_fit, sigma2_fit],
            [sigma.real, sigma.imag]):
        ax.plot(omega_fit[omega_fit < 0], sigma_fit[omega_fit < 0],
                c='k', lw=3, ls='--', label="Fit")
        ax.plot(omega_fit[omega_fit > 0], sigma_fit[omega_fit > 0],
                c='k', lw=3, ls='--')
        ax.plot(omega[omega < 0], sigma_data[omega < 0],
                c='r', lw=3, ls='-', label="Data")
        ax.plot(omega[omega > 0], sigma_data[omega > 0],
                c='r', lw=3, ls='-')


def plot_parameters_text(ax3, renderer, fit):
    ypos = 1.0
    for param in fit.params.items():
        color = 'r' if param[1].vary else 'k'
        t = ax3.text(
            0, ypos, f"${name_to_latex[param[0]]} = {param[1].value:.3g}$",
            transform=ax3.transAxes, fontsize='small', color=color,
            va='top', ha='left')
        bbox = t.get_window_extent(renderer=renderer)
        ypos -= ax3.transAxes.inverted().transform_bbox(bbox).height + 0.03


def plot_optical_drude(fit, omega, sigma):
    fig, ax1, ax2, ax3 = generate_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    plot_data(ax1, ax2, omega, sigma, fit)
    plot_parameters_text(ax3, renderer, fit)
    fig.tight_layout(pad=0.5)
    return fig, ax1, ax2, ax3


def load_and_plot(paper, sample, field):
    omega, sigma = load_data(paper, sample, field)
    fit = load_fit(os.path.dirname(os.path.relpath(__file__))
                   + f"/fits/{paper}_fits.pkl", sample, field)
    fig, ax1, ax2, ax3 = plot_optical_drude(fit, omega, sigma)
    ax1.text(0.03, 0.92, f"LSCO\n$p={sample_to_doping[sample]}$",
             transform=ax1.transAxes, ha='left', va='top', fontsize='medium')
    ax2.legend(loc='upper left', frameon=False, fontsize='medium')
    return fig


def main():
    sample = "OD35K"
    field = 31
    fig = load_and_plot("legros", sample, field)
    fig.savefig(os.path.dirname(os.path.relpath(__file__))
                + f"/figs/legros_{sample}_B={field}_epsinf=free.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()

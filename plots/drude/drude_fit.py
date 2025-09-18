import pickle, os
import numpy as np
from lmfit import minimize, Parameters, fit_report
from tqdm import tqdm
from scipy.constants import epsilon_0
epsilon_0 = epsilon_0 * 1e7


def load_data(paper, sample, field):
    omega_dict = {}
    sigma_dict = {}
    for real_imag in ('1', '2'):
        for side in ('l', 'r'):
            omega_dict[real_imag+side], sigma_dict[real_imag+side] = (
                np.loadtxt(f"data/{paper}/{sample}_sigma"
                           f"{real_imag}{side}_{field}T.csv",
                           skiprows=1, delimiter=',', unpack=True))
    omega = np.concatenate((omega_dict["1l"], omega_dict["1r"]))
    sigma1 = np.concatenate((sigma_dict["1l"], sigma_dict["1r"]))
    sigma2 = np.concatenate((
        np.interp(omega_dict["1l"], omega_dict["2l"], sigma_dict["2l"]),
        np.interp(omega_dict["1r"], omega_dict["2r"], sigma_dict["2r"])))
    sigma = sigma1 + 1j * sigma2
    return omega, sigma


## Fit ///////////////////////////////////////////////////////////////////////
def fitting_model(omega, omega_pn_sq, omega_ps_sq, omega_c,
                  gamma, epsilon_inf=1):
    return 2j * np.pi * epsilon_0 * (
        omega_pn_sq / (omega - omega_c + 1j * gamma) + omega_ps_sq / omega
        - omega * (epsilon_inf - 1))


def compute_diff(pars, omega, data):
    sim = fitting_model(
        omega, pars["omega_pn_sq"].value, pars["omega_ps_sq"].value,
        pars["omega_c"].value, pars["gamma"].value, pars["epsilon_inf"].value)
    diff = sim - data
    return diff.view(float)


def compute_diff_multi(pars, omegas, data_sets, fields):
    diffs = []
    for field, omega, data in zip(fields, omegas, data_sets):
        sim = fitting_model(
            omega, pars[f"omega_pn_sq_{field}"].value,
            pars[f"omega_ps_sq_{field}"].value, pars[f"omega_c_{field}"].value,
            pars[f"gamma_{field}"].value, pars["epsilon_inf"].value)
        diffs.append(np.abs(sim - data))
    return np.concatenate(diffs)


def generate_pars(vary_epsilon_inf=False):
    pars = Parameters()
    pars.add("epsilon_inf", value=1, min=1, max=1e6, vary=vary_epsilon_inf)
    pars.add("omega_pn_sq", value=4e4, min=1e3, max=1e6, vary=True)
    pars.add("omega_ps_sq", value=0, min=0, max=1e6, vary=True)
    pars.add("omega_c", value=0.1, min=0, max=1, vary=True)
    pars.add("gamma", value=5, min=0.9, max=10, vary=True)
    return pars


def generate_pars_multi(fields):
    pars = Parameters()
    pars.add("epsilon_inf", value=1, min=1, max=1e6)
    for field in fields:
        pars.add(f"omega_pn_sq_{field}", value=4e4, min=1e3, max=1e6)
        pars.add(f"omega_ps_sq_{field}", value=0, min=0, max=1e6)
        pars.add(f"omega_c_{field}", value=0.1, min=0, max=1)
        pars.add(f"gamma_{field}", value=2, min=0.5, max=10)
    return pars


## Batch fitting ////////////////////////////////////////////////////////////////
def generate_fits(paper, sample, fields, vary_epsilon_inf=False):
    fits = {}
    for field in tqdm(fields):
        omega, sigma = load_data(paper, sample, field)
        pars = generate_pars(vary_epsilon_inf=vary_epsilon_inf)
        out = minimize(compute_diff, pars, args=(omega, sigma))
        fits[field] = out
    return fits


def generate_multi_fit(paper, sample, fields):
    omegas = []
    sigmas = []
    for field in fields:
        omega, sigma = load_data(paper, sample, field)
        omegas.append(omega)
        sigmas.append(sigma)
    pars = generate_pars_multi(fields)
    return minimize(compute_diff_multi, pars, args=(omegas, sigmas, fields),
                    method='differential_evolution')


def generate_post_fits():
    fit_sets = {}
    for temperature in tqdm(range(35, 55, 5)):
        fit_sets[f"post_{temperature}K"] = generate_fits(
            "post", f"post{temperature}", [0, 9, 20, 31])
    return fit_sets


def generate_legros_fits(vary_epsilon_inf=True):
    fit_sets = {}
    for sample in tqdm(["NSC", "OD13K", "OD17K", "OD35K", "OD36K", "UD39K"]):
        fit_sets[f"legros_{sample}"] = generate_fits(
            "legros", sample, [0, 4, 9, 14, 20, 31],
            vary_epsilon_inf=vary_epsilon_inf)
    return fit_sets


def generate_legros_multi_fits():
    fit_sets = {}
    for sample in tqdm(["NSC", "OD13K", "OD17K", "OD35K", "OD36K", "UD39K"]):
        fit_sets[f"legros_{sample}"] = generate_multi_fit(
            "legros", sample, [0, 4, 9, 14, 20, 31])
        with open(os.path.dirname(os.path.relpath(__file__))
                  + f"/fits/legros_{sample}_multi_fit.dat", "w") as f:
            f.write(fit_report(fit_sets[f"legros_{sample}"]))
    return fit_sets


def generate_and_save_fits():
    with open(os.path.dirname(os.path.relpath(__file__))
              + "/fits/post_fits.pkl", "wb") as f:
        pickle.dump(generate_post_fits(), f)
    with open(os.path.dirname(os.path.relpath(__file__))
              + "/fits/legros_fits.pkl", "wb") as f:
        pickle.dump(generate_legros_fits(), f)


def generate_fit_csv():
    with open(os.path.dirname(os.path.relpath(__file__))
              + "/fits/post_fits.pkl", "rb") as f:
        post_fits = pickle.load(f)
    with open(os.path.dirname(os.path.relpath(__file__))
              + "/fits/legros_fits.pkl", "rb") as f:
        legros_fits = pickle.load(f)
    with open(os.path.dirname(os.path.relpath(__file__))
              + "/fits/fit_data.csv", "w") as f:
        f.write("paper,sample,field,omega_pn_sq,"
                "omega_ps_sq,omega_c,gamma\n")
        for paper, fits in [("post", post_fits), ("legros", legros_fits)]:
            for sample, fit_set in fits.items():
                for field, fit in fit_set.items():
                    omega_pn_sq, omega_ps_sq, omega_c, gamma = \
                        get_fit_params(fit)
                    f.write(f"{paper},{sample},{field},{omega_pn_sq},"
                            f"{omega_ps_sq},{omega_c},{gamma}\n")


def get_fit_params(fit):
    omega_pn_sq = fit.params["omega_pn_sq"].value
    omega_ps_sq = fit.params["omega_ps_sq"].value
    omega_c = fit.params["omega_c"].value
    gamma = fit.params["gamma"].value
    return omega_pn_sq, omega_ps_sq, omega_c, gamma


def main():
    generate_and_save_fits()
    generate_fit_csv()
    # fits = generate_legros_multi_fits()
    # with open(os.path.dirname(os.path.relpath(__file__))
    #           + "/legros_multi_fits.pkl", "wb") as f:
    #     pickle.dump(fits, f)


if __name__ == "__main__":
    main()

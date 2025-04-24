import pickle
import numpy as np
from lmfit import minimize, Parameters
from tqdm import tqdm
from scipy.constants import epsilon_0

epsilon_0 = epsilon_0 * 1e7


def load_data(paper, sample, mag_field):
    omega_dict = {}
    sigma_dict = {}
    for real_imag in ('1', '2'):
        for side in ('l', 'r'):
            omega_dict[real_imag+side], sigma_dict[real_imag+side] = (
                np.loadtxt(f"data/{paper}/{sample}_sigma"
                           f"{real_imag}{side}_{mag_field}T.csv",
                           skiprows=1, delimiter=',', unpack=True))
    omega = np.concatenate((omega_dict["1l"], omega_dict["1r"]))
    sigma1 = np.concatenate((sigma_dict["1l"], sigma_dict["1r"]))
    sigma2 = np.concatenate((
        np.interp(omega_dict["1l"], omega_dict["2l"], sigma_dict["2l"]),
        np.interp(omega_dict["1r"], omega_dict["2r"], sigma_dict["2r"])))
    sigma = sigma1 + 1j * sigma2
    return omega, sigma


## Fit ///////////////////////////////////////////////////////////////////////
def fitting_model(omega, omega_pn_sq, omega_ps_sq, omega_c, gamma):
    return 2j * np.pi * epsilon_0 * (
        omega_pn_sq / (omega - omega_c + 1j * gamma) + omega_ps_sq / omega)


def compute_diff(pars, omega, data):
    sim = fitting_model(
        omega, pars["omega_pn_sq"].value, pars["omega_ps_sq"].value,
        pars["omega_c"].value, pars["gamma"].value)
    return np.abs(sim - data)**2


def generate_pars():
    pars = Parameters()
    pars.add("omega_pn_sq", value=4e4, min=1e4, max=1e5)
    pars.add("omega_ps_sq", value=1e3, min=0, max=1e5)
    pars.add("omega_c", value=0.1, min=0, max=1)
    pars.add("gamma", value=5, min=0.9, max=10)
    return pars


def fit_data(pars, omega, sigma):
    return minimize(compute_diff, pars, args=(omega, sigma))


## Batch fitting ////////////////////////////////////////////////////////////////
def generate_fits(paper, sample, fields):
    fits = {}
    for field in tqdm(fields):
        omega, sigma = load_data(paper, sample, field)
        pars = generate_pars()
        out = fit_data(pars, omega, sigma)
        fits[field] = out
    return fits


def generate_post_fits():
    fit_sets = {}
    for temperature in tqdm(range(35, 55, 5)):
        fit_sets[f"post_{temperature}K"] = generate_fits(
            "post", f"post{temperature}", [0, 9, 20, 31])
    return fit_sets


def generate_legros_fits():
    fit_sets = {}
    for sample in tqdm(["NSC", "OD13K", "OD17K", "OD35K", "OD36K", "UD39K"]):
        fit_sets[f"legros_{sample}"] = generate_fits(
            "legros", sample, [0, 4, 9, 14, 20, 31])
    return fit_sets


def generate_and_save_fits():
    with open("user_data/post_fits.pkl", "wb") as f:
        pickle.dump(generate_post_fits(), f)
    with open("user_data/legros_fits.pkl", "wb") as f:
        pickle.dump(generate_legros_fits(), f)


def generate_fit_csv():
    with open("user_data/post_fits.pkl", "rb") as f:
        post_fits = pickle.load(f)
    with open("user_data/legros_fits.pkl", "rb") as f:
        legros_fits = pickle.load(f)
    with open("user_data/fit_data.csv", "w") as f:
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


if __name__ == "__main__":
    main()

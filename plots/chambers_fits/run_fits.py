import os, re
import numpy as np
import lmfit
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import sys
sys.path.append(os.path.dirname(os.path.relpath(__file__)))
from chambers_fit_omega import run_fit
from chambers_fit_omega_parallel import run_fit_multi_field_parallel


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
    omega *= 2 * np.pi
    return omega, sigma


def get_init_params():
    return {
        "band_name": "LSCO",
        "a": 3.76,
        "b": 3.76,
        "c": 13.22,
        "energy_scale": 80,
        "band_params":{"mu":-0.758, "t": 1, "tp":-0.12,
                       "tpp":0.06, "tz": 0.07},
        "res_xy": 20,
        "res_z": 5,
        "N_time": 1000,
        "Bamp": 45,
        "gamma_0": 7.5, # 12.595,
        "gamma_k": 1000, # 63.823,
        "power": 4.5,
        # "gamma_dos_max": 50,
        "march_square": True,
        # "parallel": False
    }


ranges_dict = {
    "energy_scale": [50, 100],
    # "tp": [-0.2,-0.1],
    # "tpp": [-0.10,0.04],
    # "tppp": [-0.1,0.1],
    # "tpppp": [-0.05,0.05],
    # "tz": [0.001,0.08],
    # "tz2": [0.001,0.08],
    # "tz3": [0.001,0.08],
    # "tz4": [0.001,0.08],
    # "mu": [-0.7,-0.9],
    "gamma_0": [1.0, 50.0],
    "gamma_k": [1.0, 500],
    "power":[1.0, 50.0],
    # "gamma_dos_max": [0.1, 200],
    # "factor_arcs" : [1, 300],
}


def save_fit(fit_result, sample, fields, extra_info=""):
    fieldString = '-'.join([str(f) for f in fields])
    with open(os.path.dirname(os.path.relpath(__file__))
              + f"/params/{sample}_{fieldString}T{extra_info}.dat", 'w') as f:
        f.write(lmfit.fit_report(fit_result))


def load_fit(sample, fields):
    parameter_values = {}
    parameter_errors = {}
    fieldString = '-'.join([str(f) for f in fields])
    with open(os.path.dirname(os.path.relpath(__file__))
              + f"/params/{sample}_{fieldString}T.dat", 'r') as f:
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
                parameter_values[value_name] = line.split()[1]
                parameter_errors[value_name] = line.split()[3]
            line = f.readline()
    return parameter_values, parameter_errors


def load_fits(samples, fields):
    parameter_values = {}
    parameter_errors = {}
    parameter_samples = {}
    parameter_fields = {}
    for sample in samples:
        for field in fields:
            if os.path.exists(os.path.dirname(os.path.relpath(__file__))
                              + f"/params/{sample}_{field}T.dat"):
                parameter_values_file, parameter_errors_file = (
                    load_fit(sample, field))
                for value_name in parameter_values_file:
                    if value_name not in parameter_values:
                        parameter_values[value_name] = []
                        parameter_errors[value_name] = []
                        parameter_samples[value_name] = []
                        parameter_fields[value_name] = []
                    parameter_values[value_name].append(
                        parameter_values_file[value_name])
                    parameter_errors[value_name].append(
                        parameter_errors_file[value_name])
                    parameter_samples[value_name].append(sample)
                    parameter_fields[value_name].append(field)
    return (parameter_values, parameter_errors,
            parameter_samples, parameter_fields)


def export_fits_to_csv(samples, fields):
    parameter_values, parameter_errors, parameter_samples, parameter_fields = (
        load_fits(samples, fields))
    with open(os.path.dirname(os.path.relpath(__file__))
              + "/fit_params.csv", 'w') as f:
        f.write("sample,field,")
        for i, key in enumerate(sorted(parameter_values.keys())):
            if i != len(parameter_values) - 1:
                f.write(f"{key},{key}_error,")
            else:
                f.write(f"{key},{key}_error\n")
        for i in range(max(len(v) for v in parameter_values.values())):
            f.write(f"{parameter_samples[key][i]},{parameter_fields[key][i]},")
            for j, key in enumerate(sorted(parameter_values.keys())):
                if i > len(parameter_values[key]):
                    f.write(",")
                else:
                    if j != len(parameter_values) - 1:
                        f.write(f"{parameter_values[key][i]},"
                                f"{parameter_errors[key][i]},")
                    else:
                        f.write(f"{parameter_values[key][i]},"
                                f"{parameter_errors[key][i]}\n")


def run_single_fit(paper, sample, field, ranges=ranges_dict,
                   init_params=get_init_params()):
    omega, sigma = load_data(paper, sample, field)
    init_params['Bamp'] = field
    return run_fit(omega, sigma, init_params, ranges)


def run_fits(paper, samples, fields):
    with ProcessPoolExecutor() as executor:
        papers = [paper] * len(samples) * len(fields)
        samples = [[sample] * len(fields) for sample in samples]
        samples = [sample for sublist in samples for sample in sublist]
        fields = [[field] for field in fields] * len(samples)
        fields = [field for sublist in fields for field in sublist]
        results = list(tqdm(
            executor.map(run_single_fit, papers, samples, fields),
            total=len(samples)))
        for i, fit_result in enumerate(results):
            save_fit(fit_result, samples[i], fields[i])


def load_interp_multi_field(paper, sample, fields, nsample_polarity=20):
    omegas = np.empty((len(fields), 2*nsample_polarity))
    sigmas = np.empty((len(fields), 2*nsample_polarity), dtype=complex)
    for i, field in enumerate(fields):
        omega, sigma = load_data(paper, sample, field)
        omega_l = omega[omega < 0]
        omega_l = np.linspace(omega_l[0], omega_l[-1], nsample_polarity)
        omega_r = omega[omega > 0]
        omega_r = np.linspace(omega_r[0], omega_r[-1], nsample_polarity)
        omegas[i] = np.concatenate((omega_l, omega_r))
        sigma_l = np.interp(omega_l, omega[omega < 0], sigma[omega < 0])
        sigma_r = np.interp(omega_r, omega[omega > 0], sigma[omega > 0])
        sigmas[i] = np.concatenate((sigma_l, sigma_r))
    return omegas, sigmas


def run_fits_multi_field(paper, samples, fields, nsample_polarity=20):
    for sample in samples:
        omegas, sigmas = load_interp_multi_field(
            paper, sample, fields, nsample_polarity)
        init_params = get_init_params()
        fit_result = run_fit_multi_field_parallel(
            fields, omegas, sigmas, init_params, ranges_dict)
        with open(os.path.dirname(os.path.relpath(__file__))
                  + f"/params/{sample}_all.dat", 'w') as f:
            f.write(lmfit.fit_report(fit_result))


def main():
    # run_fits("post", ["post50", "post45", "post40", "post35"],
    #          [0, 9, 20, 31])
    # run_fits("legros", ["NSC", "OD13K", "OD17K", "OD35K", "OD36K", "UD39K"],
    #          [0, 4, 9, 14, 20, 31])
    run_fits("legros", ["OD17K"], [0, 4, 9, 14, 20, 31])
    # run_fits_single_field("legros", "OD17K", 14)
    # export_fits_to_csv(
    #     ["NSC", "OD13K", "OD17K", "OD35K", "OD36K", "UD39K",
    #      "post35", "post40", "post45", "post50"], [0, 4, 9, 14, 20, 31])


if __name__ == "__main__":
    main()

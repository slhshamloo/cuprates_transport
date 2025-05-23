import os
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.conductivity import Conductivity
from matplotlib import pyplot as plt
from matplotlib import colors as clr


def chambers_residual(omegas, sigmas, band_obj, params):
    band_obj = deepcopy(band_obj)
    cond_obj = Conductivity(band_obj, **params)
    cond_obj.runTransport()

    sigma_fit = np.empty_like(omegas, dtype=complex)
    for (i, omega) in enumerate(omegas):
        setattr(cond_obj, "omega", omega)
        cond_obj.chambers_func()
        sigma_fit[i] = (cond_obj.sigma[0, 0] + 1j*cond_obj.sigma[0, 1]
                        ) * 1e-5

    return np.abs(sigmas-sigma_fit) ** 2


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
        "a": 3.76,
        "b": 3.76,
        "c": 13.22,
        "energy_scale": 160,
        "band_params":{"mu":-0.758, "t": 1, "tp":-0.12,
                       "tpp":0.06, "tz": 0.07},
        "res_xy": 20,
        "res_z": 5,
        "N_time": 1000,
        "Bamp": 9,
        "gamma_0": 10.0,
        "gamma_k": 100,
        "power": 6,
        # "gamma_dos_max": 50,
        "march_square": True,
        "parallel": False
    }


def calculate_residual(x_label, y_label, x_val, y_val,
                       fields, band_obj, omega_sigma_pairs):
    params = get_init_params()
    params[x_label] = x_val
    params[y_label] = y_val

    res = []
    for k in range(len(fields)):
        params['Bamp'] = fields[k]
        omegas, sigmas = omega_sigma_pairs[k]
        res.append(chambers_residual(omegas, sigmas, band_obj, params))

    return np.mean(res)


def calculate_residual_plane(
        x_label, y_label, x_bounds, y_bounds, x_count, y_count,
        paper, sample, fields, extra_label=""):
    
    if len(fields)==1: fieldDescr = fields[0]
    else: fieldDescr = '-'.join([str(field) for field in fields])
    xy = np.mgrid[x_bounds[0]:x_bounds[1]:1j*x_count,
                  y_bounds[0]:y_bounds[1]:1j*y_count]
    x_vals = xy[0].flatten()
    y_vals = xy[1].flatten()
    omega_sigma_pairs = [load_data(paper, sample, field) for field in fields]
    band_obj = BandStructure(**get_init_params())
    band_obj.runBandStructure()
    residuals = np.empty_like(x_vals)
    with ProcessPoolExecutor(max_workers=8) as executor:
        for (i, res) in enumerate(tqdm(executor.map(
                calculate_residual, [x_label]*x_vals.size,
                [y_label]*y_vals.size, x_vals, y_vals,
                [fields]*x_vals.size, [band_obj]*x_vals.size,
                [omega_sigma_pairs]*x_vals.size), total=x_count*y_count)):
            residuals[i] = res
    residuals = residuals.reshape(x_count, y_count)
    np.save(os.path.dirname(os.path.relpath(__file__))
            + f"/explore/xy_{sample}_{fieldDescr}T{extra_label}.npy", xy)
    np.save(os.path.dirname(os.path.relpath(__file__))
            + f"/explore/residuals_{sample}_{fieldDescr}T{extra_label}.npy",
            residuals)
    return xy, residuals


def plot_residual_plane(xy, residuals, vary_x_label, vary_y_label,
                        title, filename=""):
    fig, ax = plt.subplots(figsize=(10, 8))
    c = ax.imshow(
        residuals.swapaxes(0, 1), norm=clr.LogNorm(), cmap='viridis', origin='lower',
        extent=[xy[0].min(), xy[0].max(), xy[1].min(), xy[1].max()],
        aspect='auto')
    fig.colorbar(c, ax=ax)
    ax.set_xlabel(vary_x_label)
    ax.set_ylabel(vary_y_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.dirname(os.path.relpath(__file__))
                + f"/explore/{filename}", bbox_inches='tight')
    #plt.show()


def main():
    extra_label = "_energy_scale_160_gamma_k_100"
    fields = [0,4,9, 14, 20, 31]
    sample = "OD17K"

    if len(fields)==1: fieldDescr = fields[0]
    else: fieldDescr = '-'.join([str(field) for field in fields])

    xy, residuals = calculate_residual_plane(
        "gamma_0", "power", [1, 30], [1, 5], 10, 10,
       "legros", sample, fields, extra_label=extra_label)
    # xy = np.load(os.path.dirname(os.path.relpath(__file__))
    #              + f"/explore/xy_{sample}_{fieldDescr}T{extra_label}.npy")
    # residuals = np.load(os.path.dirname(os.path.relpath(__file__))
    #                     + f"/explore/residuals_{sample}_{fieldDescr}T{extra_label}.npy")
    plot_residual_plane(xy, residuals, r"$\nu_0$", r"$\Gamma_k$",
        rf"Residuals for {sample} at {fieldDescr}T, $\Gamma_0=10$",
        filename=f"residuals_{sample}_{fieldDescr}T{extra_label}_power_gamma_k.pdf")


if __name__ == "__main__":
    main()

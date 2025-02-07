import lmfit
import numpy as np
from copy import deepcopy
from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.conductivity import Conductivity


def get_lmfit_pars(params, ranges):
    lmfit_pars = lmfit.Parameters()
    for value_label in ranges:
        if value_label in params:
            value = params[value_label]
        else:
            value = params["band_params"][value_label]
        lmfit_pars.add(value_label, value=value, min=ranges[value_label][0],
                       max=ranges[value_label][1])
    return lmfit_pars


def chambers_residual(lmfit_pars, omegas, sigmas,
                      band_obj, params, ranges):
    func_params = params.copy()
    band_obj = deepcopy(band_obj)
    rerun_band = False
    for value_label in ranges:
        if value_label == "energy_scale":
            func_params[value_label] = lmfit_pars[value_label].value
            band_obj.energy_scale = lmfit_pars[value_label].value
            rerun_band = True
        elif value_label in func_params:
            func_params[value_label] = lmfit_pars[value_label].value
        else:
            func_params["band_params"][value_label] = (
                lmfit_pars[value_label].value)
            band_obj[value_label] = lmfit_pars[value_label].value
            rerun_band = True
    if rerun_band:
        band_obj.runBandStructure()
    cond_obj = Conductivity(band_obj, **func_params)
    cond_obj.runTransport()

    sigma_fit = np.empty_like(sigmas)
    for (i, omega) in enumerate(omegas):
        setattr(cond_obj, "omega", omega)
        cond_obj.chambers_func()
        sigma_fit[i] = (cond_obj.sigma[0, 0] + 1j * cond_obj.sigma[0, 1]
                        ).conjugate() * 1e-5

    return (sigmas - sigma_fit).view(float)


def run_fit(omega, sigma, init_params, ranges_dict):
    band_obj = BandStructure(**init_params)
    band_obj.runBandStructure()
    pars = get_lmfit_pars(init_params, ranges_dict)
    return lmfit.minimize(
        chambers_residual, pars,
        args=(omega, sigma, band_obj, init_params, ranges_dict))
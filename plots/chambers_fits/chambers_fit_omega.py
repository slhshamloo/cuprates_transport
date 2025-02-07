import lmfit
import numpy as np
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


def chambers_residual(lmfit_pars, omegas, sigmas, params, ranges):
    func_params = params.copy()
    for value_label in ranges:
        if value_label in func_params:
            func_params[value_label] = lmfit_pars[value_label].value
        else:
            func_params["band_params"][value_label] = lmfit_pars[value_label].value
    band_obj = BandStructure(**func_params)
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
    pars = get_lmfit_pars(init_params, ranges_dict)
    return lmfit.minimize(chambers_residual, pars,
                          args=(omega, sigma, init_params, ranges_dict))
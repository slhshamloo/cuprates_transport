import numpy as np
from scipy.optimize import differential_evolution, least_squares
from lmfit import Parameters
from lmfit.minimizer import MinimizerResult
from copy import deepcopy
from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.conductivity import Conductivity


def chambers_residual(param_values, param_keys, omegas, sigmas,
                      band_obj, init_params, reduce_error=True):
    func_params = init_params.copy()
    band_obj = deepcopy(band_obj)
    rerun_band = False
    for (i, param_key) in enumerate(param_keys):
        if param_key == 'energy_scale':
            func_params[param_key] = param_values[i]
            band_obj.energy_scale = param_values[i]
            rerun_band = True
        elif param_key in func_params:
            func_params[param_key] = param_values[i]
        else:
            func_params['band_params'][param_key] = param_values[i]
            band_obj[param_key] = param_values[i]
            rerun_band = True
    if rerun_band:
        band_obj.runBandStructure()
    cond_obj = Conductivity(band_obj, **func_params)
    cond_obj.runTransport()

    sigma_fit = np.empty_like(sigmas)
    for (i, omega) in enumerate(omegas):
        setattr(cond_obj, "omega", omega)
        cond_obj.chambers_func()
        sigma_fit[i] = (cond_obj.sigma[0, 0] + 1j*cond_obj.sigma[0, 1]
                        ).conjugate() * 1e-5
    if reduce_error:
        return np.sum(np.abs(sigmas-sigma_fit)**2) / (
            len(omegas)-len(param_keys))
    else:
        return np.abs(sigmas-sigma_fit)


def get_lmfit_pars(params, ranges):
    lmfit_pars = Parameters()
    for value_label in ranges:
        lmfit_pars.add(value_label, value=params[value_label],
                       min=ranges[value_label][0],
                       max=ranges[value_label][1])
    return lmfit_pars


def convert_scipy_result_to_lmfit(result, polish, param_keys, ranges, ndata):
    chi_sq = np.sum(polish.fun**2)
    if chi_sq < result.fun * (ndata-len(param_keys)):
        calc_err = True
        params = dict(zip(param_keys, polish.x))
        lmfit_pars = get_lmfit_pars(params, ranges)

        if hasattr(polish, 'jac'):
            cov_matrix = np.linalg.inv(polish.jac.T@polish.jac) * (
                np.sum(polish.fun**2) / (len(polish.fun)-len(param_keys)))
        else:
            cov_matrix = np.zeros((len(param_keys), len(param_keys)))
        for (i, param_key) in enumerate(param_keys):
            lmfit_pars[param_key].stderr = np.sqrt(cov_matrix[i, i])
            lmfit_pars[param_key].correl = {}
            for (j, other_param) in enumerate(param_keys):
                if i != j:
                    lmfit_pars[param_key].correl[other_param] = (
                        cov_matrix[i, j] / np.sqrt(
                            cov_matrix[i, i]*cov_matrix[j, j]))
    else:
        calc_err = False
        params = dict(zip(param_keys, result.x))
        lmfit_pars = get_lmfit_pars(params, ranges)

    return MinimizerResult(
        params = lmfit_pars, success=result.success, message=result.message,
        method="differential_evolution", errorbars=calc_err, nfev=result.nfev,
        nvarys=len(param_keys), ndata=ndata, nfree=ndata-len(param_keys),
        chisqr=chi_sq, redchi=chi_sq/(ndata-len(param_keys)),
        aic=ndata*np.log(chi_sq/ndata) + 2*len(param_keys),
        bic=ndata*np.log(chi_sq/ndata) + len(param_keys)*np.log(ndata))


def run_fit(omegas, sigmas, init_params, ranges_dict):
    band_obj = BandStructure(**init_params)
    band_obj.runBandStructure()
    param_keys = list(ranges_dict.keys())
    bounds = [ranges_dict[key] for key in param_keys]
    fit_result = differential_evolution(
        chambers_residual, bounds, workers=-1, polish=False,
        args=(param_keys, omegas, sigmas, band_obj, init_params, True))
    bounds = [
        [max(0.0, fit_result.x[i] - 0.05 * (ranges_dict[param_keys[i]][1]
                                            -ranges_dict[param_keys[i]][0]))
         for i in range(len(param_keys))],
        [fit_result.x[i] + 0.05 * (ranges_dict[param_keys[i]][1]
                                   -ranges_dict[param_keys[i]][0])
         for i in range(len(param_keys))]]
    polished_fit = least_squares(
        chambers_residual, fit_result.x, bounds=bounds,
        args=(param_keys, omegas, sigmas, band_obj, init_params, False))
    return convert_scipy_result_to_lmfit(fit_result, polished_fit, param_keys,
                                         ranges_dict, len(omegas))

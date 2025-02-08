import numpy as np
from scipy.optimize import differential_evolution
from lmfit import Parameters
from lmfit.minimizer import MinimizerResult
from copy import deepcopy
from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.conductivity import Conductivity


def chambers_residual(param_values, param_keys, omegas, sigmas,
                      band_obj, init_params):
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

    return np.sum(np.abs(sigmas-sigma_fit)**2) / (
        len(omegas)-len(param_keys))


def get_lmfit_pars(params, ranges):
    lmfit_pars = Parameters()
    for value_label in ranges:
        lmfit_pars.add(value_label, value=params[value_label],
                       min=ranges[value_label][0],
                       max=ranges[value_label][1])
    return lmfit_pars


def convert_scipy_result_to_lmfit(result, param_keys, ranges,
                                  init_params, ndata):
    params = dict(zip(param_keys, result.x))
    if hasattr(result, 'jac'):
        cov_matrix = np.linalg.inv(result.jac.T @ result.jac) * result.fun
    else:
        cov_matrix = np.zeros((len(param_keys), len(param_keys)))
    param_errors = dict(zip(param_keys, np.sqrt(np.diag(cov_matrix))))
    return MinimizerResult(
        params = get_lmfit_pars(params, ranges),
        uvars=param_errors, covar=cov_matrix, var_names=param_keys,
        init_vals = [init_params[param_key] for param_key in param_keys],
        success=result.success, message=result.message, nfev=result.nfev,
        nvarys=len(param_keys), ndata=ndata, nfree = ndata - len(param_keys),
        redchi=result.fun, chisqr = result.fun * (ndata-len(param_keys)))


def run_fit(omegas, sigmas, init_params, ranges_dict):
    band_obj = BandStructure(**init_params)
    band_obj.runBandStructure()
    param_keys = list(ranges_dict.keys())
    bounds = [ranges_dict[key] for key in param_keys]
    fit_result = differential_evolution(
        chambers_residual, bounds, workers=-1, polish=True,
        args=(param_keys, omegas, sigmas, band_obj, init_params))
    return convert_scipy_result_to_lmfit(fit_result, param_keys, ranges_dict,
                                         init_params, len(omegas))

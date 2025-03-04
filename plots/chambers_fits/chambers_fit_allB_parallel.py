import numpy as np
from scipy.optimize import differential_evolution, least_squares
from lmfit import Parameters
from lmfit.minimizer import MinimizerResult
from copy import deepcopy
from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.conductivity import Conductivity
import time
import sys
from functools import reduce

def chambers_residual_allB(param_values, param_keys, omegaSigmaPairs, fields,
                      band_obj, init_params, reduce_error=True):
    
    lengths = [len(e[0]) for e in omegaSigmaPairs]
    len_omegas = reduce(lambda x,y: x+y, lengths)
    init_params = deepcopy(init_params)

    func_params = deepcopy(init_params)
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

    def iteratorFunction(field, pairs):
        sigmas, omegas = pairs
        func_params['Bamp'] = field
        cond_obj = Conductivity(band_obj, **func_params)
        cond_obj.runTransport()

        sigma_fit = np.empty_like(sigmas, dtype='complex')
        for (i, omega) in enumerate(omegas):
            setattr(cond_obj, "omega", omega)
            cond_obj.chambers_func()
            sigma_fit[i] = (cond_obj.sigma[0, 0] + 1j*cond_obj.sigma[0, 1]
                            ).conjugate() * 1e-5
        if reduce_error:
            return np.sum(np.abs(sigmas-sigma_fit)**2)
        else:
            return np.abs(sigmas-sigma_fit)
    
    residuals = map(iteratorFunction, fields, omegaSigmaPairs)
    residual = reduce(lambda x,y: x+y, residuals)

    if reduce_error:
        residual /= (len_omegas-len(param_keys))

    return residual


def get_lmfit_pars(params, ranges):
    lmfit_pars = Parameters()
    for value_label in ranges:
        lmfit_pars.add(value_label, value=params[value_label],
                       min=ranges[value_label][0],
                       max=ranges[value_label][1])
    return lmfit_pars


def convert_scipy_result_to_lmfit(
        result, lmfit_pars, param_keys, ndata, calc_err=True):
    chi_sq = np.sum(result.fun**2)
    if calc_err:
        if hasattr(result, 'jac'):
            cov_matrix = np.linalg.inv(result.jac.T@result.jac) * (
                np.sum(result.fun**2) / (ndata-len(param_keys)))
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

    return MinimizerResult(
        params = lmfit_pars, success=result.success, message=result.message,
        method="differential_evolution", errorbars=calc_err, nfev=result.nfev,
        nvarys=len(param_keys), ndata=ndata, nfree=ndata-len(param_keys),
        chisqr=chi_sq, redchi=chi_sq/(ndata-len(param_keys)),
        aic=ndata*np.log(chi_sq/ndata) + 2*len(param_keys),
        bic=ndata*np.log(chi_sq/ndata) + len(param_keys)*np.log(ndata))


def run_fit_allB_parallel(omega_sigma_pairs, fields, init_params, ranges_dict):

    ## Handle the reporting at the end of each iteration
    globals()['iteration'] = 0
    globals()['time_iter'] = time.time()
    globals()['best_x'] = init_params
    def fn(xk, convergence):
        globals()['iteration'] += 1
        text = "Iteration: %d\titer time: %.3f\tconvergence: %.3e" % (globals()['iteration'], (time.time() - globals()['time_iter']), convergence)
        globals()['time_iter'] = time.time()
        if (xk != globals()['best_x']).all():
            globals()['best_x'] = xk
            # obj_val = fit_object.compute_diff2(xk, verbose=False)
            sys.stdout.flush()
            # text += "\tNew best:" + str([round(x, 10) for x in xk]) + "\tchi^2: %.3e" % obj_val
            text += "\tNew best:" + str([round(x, 10) for x in xk])
        print(text)
        sys.stdout.flush()
    
    ## Band objects for all fields
    band_params = deepcopy(init_params)
    band_obj = BandStructure(**band_params)

    ## Bounds (are all the same)
    param_keys = list(ranges_dict.keys())
    bounds = [ranges_dict[key] for key in param_keys]
    global_fit = differential_evolution(
        chambers_residual_allB, bounds, workers=-1, polish=False,
        x0=[init_params[key] for key in param_keys],
        args=(param_keys, omega_sigma_pairs, fields, band_obj, init_params, True), callback=fn)
    params = dict(zip(param_keys, global_fit.x))
    lmfit_pars = get_lmfit_pars(params, ranges_dict)

    ## Polishing phase
    bounds = [[ranges_dict[key][0] for key in param_keys],
              [ranges_dict[key][1] for key in param_keys]]
    polished_fit = least_squares(
        chambers_residual_allB, global_fit.x, bounds=bounds,
        args=(param_keys, omega_sigma_pairs, fields, band_obj, init_params, False))
    for i, param_key in enumerate(param_keys):
        lmfit_pars[param_key].value = polished_fit.x[i]
    total_len = reduce(lambda x,y: x+y, [len(e[0]) for e in omega_sigma_pairs])
    local_fit_lmfit = convert_scipy_result_to_lmfit(
        polished_fit, lmfit_pars, param_keys, total_len, True)
    
    return local_fit_lmfit    
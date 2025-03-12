### From plot_chambers_fits.py

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
        "gamma_0": 5,
        "gamma_k": 500,
        "power": 4,
        # "gamma_dos_max": 50,
        "march_square": True,
        "parallel": False
    }

#def get_init_params():
#    return {
#        "Bamp": 45,
#        "gamma_0": 7.5, # 12.595,
#        "gamma_k": 50, # 63.823,
#        "power": 4.5,
#    }

def get_ranges():
    return {
        "energy_scale": [50, 250],
        # "tp": [-0.2,-0.1],
        # "tpp": [-0.10,0.04],
        # "tppp": [-0.1,0.1],
        # "tpppp": [-0.05,0.05],
        # "tz": [0.001,0.08],
        # "tz2": [0.001,0.08],
        # "tz3": [0.001,0.08],
        # "tz4": [0.001,0.08],
        # "mu": [-0.7,-0.9],
        "gamma_0": [1.0, 50.0],#"gamma_0": [1.0, 50.0],
        "gamma_k": [1.0, 2000],#"gamma_k": [1.0, 500],
        "power":[1.0, 20.0]#"power":[1.0, 50.0],
        # "gamma_dos_max": [0.1, 200],
        # "factor_arcs" : [1, 300],
    }
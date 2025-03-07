### From plot_chambers_fits.py

def get_init_params():
    return {
        "band_name": "Nd-LSCO",
        "a": 3.75,
        "b": 3.75,
        "c": 13.2,
        "energy_scale": 80,
        "band_params":{"mu":-0.758, "t": 1, "tp":-0.12,
                       "tpp":0.06, "tz": 0.07},
        "res_xy": 20,
        "res_z": 5,
        "N_time": 1000,
        "Bamp": 9,
        "gamma_0": 10.0,
        "gamma_k": 5,
        "power": 6,
        # "gamma_dos_max": 50,
        "march_square": True,
        "parallel": False
    }
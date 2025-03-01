from numpy import pi, deg2rad
import numpy as np
import os
from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.conductivity import Conductivity
from scipy.constants import Boltzmann, hbar, elementary_charge, physical_constants, electron_mass, epsilon_0
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#


##!!!!!!! WARNING !!!!!!!!##
# You need to set kz_max = pi / c, instead of 2pi / c for it work. #####
input("Did you set kz_max = pi / c in Bandstructure ?")
input("Did you rebuild cuprates_transport ?")


params = {
    "band_name": "Free electrons",
    "a": 3.75,
    "b": 3.75,
    "c": 13.2,
    "energy_scale": 1000,
    "band_params":{"mu":-0.5, "t": 1},
    "tight_binding": "mu + t*((kx)**2+(ky)**2)", #+ t*((kx+pi/a)**2+(ky-pi/b)**2) + t*((kx-pi/a)**2+(ky-pi/b)**2)",
    "res_xy": 300,
    "res_z": 3,
    "dfdE_cut_percent": 0.001,
    "N_epsilon": 30,
    "N_time": 1000,
    "T" : 0,
    "Bamp": 0,
    "gamma_0": 40.25,
    "omega": 1.0,
}


## Units ////////
meV = physical_constants["electron volt"][0] * 1e-3 # 1 meV in Joule

a = params["a"]* 1e-10
c = params["c"]* 1e-10
## Volume between two planes in the unit cell
V = a**2 * c

## Create Bandstructure object
bandObject = BandStructure(**params)

## Discretize Fermi surface
bandObject.runBandStructure()
# bandObject.figMultipleFS2D()
# bandObject.figDiscretizeFS3D()


kf = np.sqrt(bandObject.kf[0,0]**2+bandObject.kf[1,0]**2)/1e-10 # m
# vf = np.sqrt(bandObject.vf[0,0]**2+bandObject.vf[1,0]**2)*1e-10/hbar

E_F = params["band_params"]["mu"]*params["energy_scale"]*meV
m_star =hbar**2*kf**2/(2*np.abs(E_F)) # comes from E_F = p_F**2 / (2 * m_star)
print("m* = " + str(m_star/electron_mass))
# bandObject.mass_func()
# print("m* = " + str(bandObject.mass))
# m_star = bandObject.mass * electron_mass

carrier_density = bandObject.n / V # in m^-3


## Compute rhoxx and RH >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
N = 250
wMax = 150
fieldValues = [0, 250, 500, 1000, 3000]
omegaValues = np.linspace(0,wMax, N)
for B in fieldValues:
  # Initialize results array
  results = np.zeros((N,3), dtype="complex")
  results[:,0] = omegaValues

  # Get Chambers' formula results
  params["Bamp"] = B
  for i in range(N):
    # TODO: Vectorize the computation over omega, requires attention to Conductivity (JIT?)
    params["omega"] = omegaValues[i]
    condObject = Conductivity(bandObject, **params)
    condObject.runTransport()
    results[i,1] = condObject.sigma[0,0]

  # Get Drude model
  m = m_star #kg
  e = elementary_charge # C
  n = carrier_density # m^-3
  w =  omegaValues * 1e12 # rad.s-1
  Ga = params["gamma_0"] * 1e12 # rad.s-1
  wc_sq = e**2*B**2/m**2 # (rad.s-1)2
  results[:,2] = 1j*n*e**2/m * (w + 1j*Ga) / ((w + 1j*Ga)**2 - wc_sq)

  path = os.path.relpath(__file__)
  dirname, fname = os.path.split(path)
  newfname = f"sigmaxx__w_0_{wMax}__B_{B}.dat"
  print(newfname)
  np.savetxt(f"{dirname}/{newfname}", results, header="omega (THz)   sigma_xx Chambers ()   sigma_xx Drude ()", comments="#", delimiter=",")
  print(f"Saved {newfname}")

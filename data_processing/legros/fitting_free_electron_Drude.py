import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import os
import sys
from lmfit import minimize, Parameters, report_fit
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

x_min = -2
x_max = 2

## Load Data /////////////////////////////////////////////////////////////////////
fullpath = os.path.relpath(__file__)
dirname, fname = os.path.split(fullpath)
project_root = dirname + "/../../"
datapath = project_root+"data/legros"

k = "NSC"
B = 31

# Handling the real part
i = 1
# Load positive values part
j = "r"
filename = f"{k}_sigma{i}{j}_{B}T.csv"
data = np.loadtxt(f"{datapath}/{filename}", dtype="float", comments="#", delimiter=',', skiprows=1)
x_data_r = data[:,0]
y_data_r = data[:,1]
# Load negative values
j = "l"
filename = f"{k}_sigma{i}{j}_{B}T.csv"
data = np.loadtxt(f"{datapath}/{filename}", dtype="float", comments="#", delimiter=',', skiprows=1)
x_data_r = np.concatenate([data[:,0], x_data_r])
y_data_r = np.concatenate([data[:,1], y_data_r])

# Handling the imaginary part
i = 2
# Load positive values part
j = "r"
filename = f"{k}_sigma{i}{j}_{B}T.csv"
data = np.loadtxt(f"{datapath}/{filename}", dtype="float", comments="#", delimiter=',', skiprows=1)
x_data_i = data[:,0]
y_data_i = data[:,1]
# Load negative values
j = "l"
filename = f"{k}_sigma{i}{j}_{B}T.csv"
data = np.loadtxt(f"{datapath}/{filename}", dtype="float", comments="#", delimiter=',', skiprows=1)
x_data_i = np.concatenate([data[:,0], x_data_i])
y_data_i = np.concatenate([data[:,1], y_data_i])

## Fit ///////////////////////////////////////////////////////////////////////////
def fit_model(omega, gamma, omega_pn_sq, omega_c, omega_ps_sq):
    """ Omega and gamma in rad.THz, omega_pn_sq in THz^2"""
    e = 1.602e-19
    me = 9.109e-31
    e0 = 8.854e-12
    # omega_c = 1e-12*e*B/(2*me) # Cyclotron frequency in rad.THz
    return 1j*1e7*e0*omega_pn_sq/(2*np.pi*omega - omega_c + 1j*gamma) + 1j*omega_ps_sq/omega


def compute_diff(pars, reals, imags, d_real, d_imag):
    gamma   = pars["gamma"].value
    omega_pn_sq   = pars["omega_pn_sq"].value
    omega_c = pars["omega_c"].value
    omega_ps_sq = pars["omega_ps_sq"].value
    sim_r = np.real(fit_model(reals, gamma, omega_pn_sq, omega_c, omega_ps_sq))
    sim_imag = np.imag(fit_model(imags, gamma, omega_pn_sq, omega_c, omega_ps_sq))
    return np.concatenate([sim_r - d_real, sim_imag - d_imag])

e = 1.602e-19
me = 9.109e-31
omega_c_zero = 1e-12*e*B/(2*me) # Cyclotron frequency in rad.THz


pars = Parameters()
pars.add("gamma", value = 1e0)
pars.add("omega_pn_sq", value = 40*1e3)
pars.add("omega_ps_sq", value = 0, vary=False)
pars.add("omega_c", value = omega_c_zero)


out = minimize(compute_diff, pars, args=(x_data_r, x_data_i, y_data_r, y_data_i))

gamma = out.params["gamma"].value # Gamma in rad.THz
omega_pn_sq = out.params["omega_pn_sq"].value # OmegaPN in rad**2.THz**2
omega_ps_sq = out.params["omega_ps_sq"].value
omega_c = out.params["omega_c"].value # Omegac in rad.THz

fullpath = os.path.relpath(__file__)
dataname = f"{fullpath[0:-3]}__T_{k}_B_{B}.dat"

with open(dataname, 'w') as sys.stdout:
    report_fit(out)


x_fit = np.linspace(x_min, x_max, 1000)
y_fit = fit_model(x_fit, gamma, omega_pn_sq, omega_c, omega_ps_sq)



##############################################################################
### Plotting #################################################################
##############################################################################

mpl.rcdefaults()
mpl.rcParams['font.size'] = 24. # change the size of the font in every figure
mpl.rcParams['font.family'] = 'Arial' # font Arial in every figure
mpl.rcParams['axes.labelsize'] = 24.
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['xtick.direction'] = "in"
mpl.rcParams['ytick.direction'] = "in"
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.major.width'] = 0.6
mpl.rcParams['ytick.major.width'] = 0.6
mpl.rcParams['axes.linewidth'] = 0.6 # thickness of the axes lines
mpl.rcParams['pdf.fonttype'] = 3  # Output Type 3 (Type3) or Type 42 (TrueType), TrueType allows
                                    # editing the text in illustrator


####################################################
## Plot Parameters #################################

fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure

fig.subplots_adjust(left = 0.18, right = 0.82, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

# axes.axhline(y=0, ls ="--", c ="k", linewidth=0.6)

#############################################
# fig.text(0.22,0.23, "")
#############################################

#############################################
axes.set_xlim(1.2*x_min, 1.2*x_max)   # limit for xaxis
# axes.set_ylim(-3.5,1.2) # leave the ymax auto, but fix ymin
axes.tick_params(axis='x', which='major', pad=7)
axes.tick_params(axis='y', which='major', pad=8)

axes.set_xlabel(r"x", labelpad = 8)
axes.set_ylabel(r"y", labelpad = 8)
#############################################


######################################################
color = 'r'
line = axes.plot(x_data_r, y_data_r, label = r"data")
plt.setp(line, ls ="", c = color, lw = 2, marker = "o", mfc = color, ms = 7, mec = color, mew= 2)
######################################################

######################################################
color = 'k'
line = axes.plot(x_fit, np.real(y_fit), label = r"fit")
plt.setp(line, ls ="-", c = color, lw = 3, marker = "", mfc = color, ms = 7, mec = color, mew= 2)
######################################################

######################################################
plt.legend(loc = 0, fontsize = 20, frameon = False, numpoints=1, markerscale = 1, handletextpad=0.5)
# plt.legend(bbox_to_anchor=(0.5, 0.5), loc = 1, fontsize = 12, frameon = False, numpoints=1, markerscale = 1, handletextpad=0.5)
######################################################


## Set ticks space and minor ticks space ############
#####################################################.

# xtics = 30 # space between two ticks
# mxtics = xtics / 2.  # space between two minor ticks
# ytics = 1
# mytics = ytics / 2. # or "AutoMinorLocator(2)" if ytics is not fixed, just put 1 minor tick per interval
# majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

# axes.xaxis.set_major_locator(MultipleLocator(xtics))
# axes.xaxis.set_major_formatter(majorFormatter)
# axes.xaxis.set_minor_locator(MultipleLocator(mxtics))

# axes.yaxis.set_major_locator(MultipleLocator(ytics))
# axes.yaxis.set_major_formatter(majorFormatter)
# axes.yaxis.set_minor_locator(MultipleLocator(mytics))

######################################################


plt.show()

fullpath = os.path.relpath(__file__)
dirname, fname = os.path.split(fullpath)
project_root = dirname + "/../../"
datapath = project_root+"plots/legros"

figurename = f"{datapath}/fitting_Drude_T_{k}_B_{B}.pdf"

fig.savefig(figurename, bbox_inches = "tight")
plt.close()
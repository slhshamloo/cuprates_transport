import numpy as np
import scipy as sp
from scipy import interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import os
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

## Plotting >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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

# cmap = mpl.cm.get_cmap("viridis", len(files))
# colors = cmap(np.arange(len(files)))
# cmap = mpl.colormaps["viridis"]
# colors[-1] = (1, 0, 0, 1)
color = "#0000FF"

####################################################
## Plot Parameters #################################

fig, axis = plt.subplots(1, 1, figsize = (9.2, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
fig.subplots_adjust(left = 0.18, right = 0.82, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

# axs.axhline(y=0, ls ="--", c ="k", linewidth=0.6)

#############################################
# fig.text(0.79,0.23, r"label", ha = "right")
#############################################


##############################################################################
### Load and Plot Data #######################################################
##############################################################################

######################################################
Bvalues = [1000]
wmax = 150
fullpath = os.path.relpath(__file__)
dirname, fname = os.path.split(fullpath)
project_root = dirname + "/../../"
datapath = project_root+"data/free_electron"


for i, B in enumerate(Bvalues):
  filename = fname[:-3]
  data = np.loadtxt(f"{datapath}/{filename}.dat", dtype="complex", comments="#", delimiter=',')

  x_omega = np.real(data[:,0])

  data[:,1:] *= 1e-5 # Unit conversion
  Re_sigmaxx_Chambers = np.real(data[:,1])
  Re_sigmaxx_Drude = np.real(data[:,2])

  # color = cmap(i/len(Bvalues))
  chambers = axis.plot(x_omega, Re_sigmaxx_Chambers, label = r"Chambers")
  drude = axis.plot(x_omega, Re_sigmaxx_Drude, label = r"Drude")


  plt.setp(chambers, ls ="-", c = color, lw = 2, marker = "", mfc = color, ms = 7, mec = color, mew= 2)
  plt.setp(drude, ls =":", c = color, lw = 2, marker = "", mfc = color, ms = 7, mec = color, mew= 2)

  ######################################################

  fig.text(0.79,0.79, f"Free electrons\n$B$ = {B} T", ha = "right")


  ######################################################

axis.legend(loc = 3, fontsize = 16, frameon = False, numpoints=1, markerscale = 1, handletextpad=0.5)
# axs.legend(bbox_to_anchor=(0.5, 0.5), loc = 1, fontsize = 12, frameon = False, numpoints=1, markerscale = 1, handletextpad=0.5)
######################################################
#############################################
axis.set_xlim(0)   # limit for xaxis
axis.set_ylim(0)   # limit for xaxis
# axs.set_xscale('log')
# axs.set_yscale('log')

axis.set_ylabel(r"Re($\sigma$) ( m$\Omega^{-1}$ cm$^{-1} )$", labelpad = 8)
axis.set_xlabel(r"$\omega$ ( THz )", labelpad = 8)
axis.tick_params(axis='x', which='major', pad=7, length=8)
axis.tick_params(axis='y', which='major', pad=8, length=8)
axis.tick_params(axis='x', which='minor', pad=7, length=4)
axis.tick_params(axis='y', which='minor', pad=8, length=4)
#############################################

## Set ticks space and minor ticks space ############
#####################################################.

# xtics = 30 # space between two ticks
# mxtics = xtics / 2.  # space between two minor ticks
# ytics = 1
# mytics = ytics / 2. # or "AutoMinorLocator(2)" if ytics is not fixed, just put 1 minor tick per interval
# majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

# axs.xaxis.set_major_locator(MultipleLocator(xtics))
# axs.xaxis.set_major_formatter(majorFormatter)
# axs.xaxis.set_minor_locator(MultipleLocator(mxtics))

# axs.yaxis.set_major_locator(MultipleLocator(ytics))
# axs.yaxis.set_major_formatter(majorFormatter)
# axs.yaxis.set_minor_locator(MultipleLocator(mytics))

######################################################
"""
figurename = "optical_B_1000_Re_150THz"
plt.show()
fig.savefig(f"user_plots/optical_conductivity/{figurename}.pdf", bbox_inches = "tight")
plt.close()
"""
path = os.path.relpath(__file__)
ex_filename = f"{path[:-3]}.pdf"

plt.show()
fig.savefig(ex_filename, bbox_inches = "tight")
print(f"Saved {ex_filename}")
plt.close()
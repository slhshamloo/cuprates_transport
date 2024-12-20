import numpy as np
import scipy as sp
from scipy import interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import os
import pickle
from matplotlib.lines import Line2D
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
cmap = mpl.colormaps["viridis"]
# colors[-1] = (1, 0, 0, 1)
# color = "#0000FF"

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
Bvalues = [0]
wmax = 150
fullpath = os.path.relpath(__file__)
dirname, fname = os.path.split(fullpath)
project_root = dirname + "/../../"
datapath = project_root+"data_processing/post"

with open(f"{datapath}/all_fits.pickle", 'rb') as f:
  results = pickle.load(f)

plots = {}

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
for T, resOverB in results.items():
  # Extract Gamma over B for given T
  bVals = []
  omegacVals = []
  for b in resOverB:
    bVals.append(b)
    omegacVals.append(resOverB[b].params['omega_c'])

  bVals = np.array(bVals)
  omegacVals = np.array(omegacVals)/(2*np.pi)

  plots[T] = axis.plot(bVals, omegacVals, label=f"T = {T} K")
  plt.setp(plots[T], ls ="", lw = 2, marker = "o", ms=10, mew= 2)

######################################################

fig.text(0.2,0.8, f"From Post et al. data\nFit extraction", ha = "left")


######################################################
axis.legend(loc = 4, fontsize = 16, frameon = True, numpoints=1, markerscale = 1, handletextpad=0.5)

######################################################
#############################################
axis.set_xlim((-1.5,32))   # limit for xaxis
axis.set_ylim((-0.05,0.23))   # limit for xaxis
# axs.set_xscale('log')
# axs.set_yscale('log')

axis.set_ylabel(r"$\omega_c/2\pi$ ( THz )", labelpad = 8)
axis.set_xlabel(r"$B$ ( T )", labelpad = 8)
axis.tick_params(axis='x', which='major', pad=7, length=8)
axis.tick_params(axis='y', which='major', pad=8, length=8)
axis.tick_params(axis='x', which='minor', pad=7, length=4)
axis.tick_params(axis='y', which='minor', pad=8, length=4)
#############################################

## Set ticks space and minor ticks space ############
#####################################################.

# xtics = 30 # space between two ticks
# mxtics = xtics / 2.  # space between two minor ticks
ytics = 0.1
#mytics = ytics / 2. # or "AutoMinorLocator(2)" if ytics is not fixed, just put 1 minor tick per interval
majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

# axs.xaxis.set_major_locator(MultipleLocator(xtics))
# axs.xaxis.set_major_formatter(majorFormatter)
# axs.xaxis.set_minor_locator(MultipleLocator(mxtics))

axis.yaxis.set_major_locator(MultipleLocator(ytics))
# axs.yaxis.set_major_formatter(majorFormatter)
# axs.yaxis.set_minor_locator(MultipleLocator(mytics))
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
######################################################

path = os.path.relpath(__file__)
ex_filename = f"{path[:-3]}.pdf"

plt.show()
fig.savefig(ex_filename, bbox_inches = "tight")
print(f"Saved {ex_filename}")
plt.close()
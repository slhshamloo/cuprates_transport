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

Bvalues = [
  0, 250, 500, 1000, 3000
]

cmap = mpl.cm.get_cmap("jet", len(Bvalues))
# colors = cmap(np.arange(len(files)))
# colors[-1] = (1, 0, 0, 1)

####################################################
## Plot Parameters #################################

fig, axs = plt.subplots(2, 1, figsize = (9.2, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
fig.subplots_adjust(left = 0.18, right = 0.82, bottom = 0.18, top = 0.90, wspace=0.7) # adjust the box of axes regarding the figure size

# axs.axhline(y=0, ls ="--", c ="k", linewidth=0.6)

#############################################
# fig.text(0.79,0.23, r"label", ha = "right")
#############################################


##############################################################################
### Load and Plot Data #######################################################
##############################################################################

######################################################
wmax = 150

fullpath = os.path.relpath(__file__)
dirname, fname = os.path.split(fullpath)
project_root = dirname + "/../../"
datapath = project_root+"data/free_electron"

for i, B in enumerate(Bvalues):
  filename = f"sigmaxx__w_0_{wmax}__B_{B}.dat"
  data = np.loadtxt(f"{datapath}/{filename}", dtype="complex", comments="#", delimiter=',')

  x_omega = np.real(data[:,0])
  data[:,1:] *= 1e-5 # Unit conversion
  Re_sigmaxx_Chambers = np.real(data[:,1])
  Im_sigmaxx_Chambers = np.imag(data[:,1])

  color = cmap(i/len(Bvalues))

  re = axs[0].plot(x_omega, Re_sigmaxx_Chambers, label = fr"{B} T")
  im = axs[1].plot(x_omega, Im_sigmaxx_Chambers, label = fr"{B} T")

  plt.setp(re, ls ="-", c = color, lw = 2, marker = "", mfc = color, ms = 7, mec = color, mew= 2)
  plt.setp(im, ls ="-", c = color, lw = 2, marker = "", mfc = color, ms = 7, mec = color, mew= 2)


  ######################################################
# hline for imaginary part

  ######################################################
axs[0].legend(loc = 1, fontsize = 16, ncols=3, frameon = False, numpoints=1, markerscale = 1, handletextpad=0.5)
# axs[1].legend(loc = 3, fontsize = 16, frameon = False, numpoints=1, markerscale = 1, handletextpad=0.5)
# axs.legend(bbox_to_anchor=(0.5, 0.5), loc = 1, fontsize = 12, frameon = False, numpoints=1, markerscale = 1, handletextpad=0.5)
######################################################
#############################################
for ax in axs:
  ax.set_xlim(0)   # limit for xaxis

axs[1].axhline(y=0, ls ="--", c ="k", linewidth=0.6)

axs[0].set_ylim(0) # leave the ymax auto, but fix ymin
# axs.set_xscale('log')
# axs.set_yscale('log')

axs[0].set_ylabel(r"Re($\sigma$)", labelpad = 8)
axs[1].set_ylabel(r"Im($\sigma$)", labelpad = 8)
axs[1].set_xlabel(r"$\omega$ ( THz )", labelpad = 8)

for ax in axs:
  ax.tick_params(axis='x', which='major', pad=7, length=8)
  ax.tick_params(axis='y', which='major', pad=8, length=8)
  ax.tick_params(axis='x', which='minor', pad=7, length=4)
  ax.tick_params(axis='y', which='minor', pad=8, length=4)

axs[0].get_xaxis().set_ticklabels([])
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
fig.text(0.79,0.77, f"Free electrons\n$B$ = {B} T", ha = "right")
fig.align_ylabels()
#fig.suptitle(f"Free electrons\t$B$ = {B} T", y=0.98)
figurename = filename + ".pdf"
plt.show()
fig.savefig(f"user_plots/optical_conductivity/{figurename}", bbox_inches = "tight")
plt.close()
"""
path = os.path.relpath(__file__)
ex_filename = f"{path[:-3]}.pdf"

plt.show()
fig.savefig(ex_filename, bbox_inches = "tight")
print(f"Saved {ex_filename}")
plt.close()
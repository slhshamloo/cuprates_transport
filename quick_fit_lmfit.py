import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import os
from lmfit import minimize, Parameters, report_fit
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

x_min = 0
x_max = 6

## Load Data /////////////////////////////////////////////////////////////////////
data = np.loadtxt("filenameT.dat", dtype="float", comments="#")
x_data = data[:,0]
y_data = data[:,1]

## Fit ///////////////////////////////////////////////////////////////////////////
def fit_model(x, a, b):
    return a + b * x**2

def compute_diff(pars, x, data):
    a   = pars["a"].value
    b   = pars["b"].value
    sim = fit_model(x, a, b)
    return sim - data


pars = Parameters()
pars.add("a", value = 0)
pars.add("b", value = 3)

out = minimize(compute_diff, pars, args=(x_data, y_data))

report_fit(out)
a = out.params["a"].value
b = out.params["b"].value

x_fit = np.linspace(x_min, x_max, 1000)
y_fit = fit_model(x_fit, a, b)



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
axes.set_xlim(0,5)   # limit for xaxis
# axes.set_ylim(-3.5,1.2) # leave the ymax auto, but fix ymin
axes.tick_params(axis='x', which='major', pad=7)
axes.tick_params(axis='y', which='major', pad=8)
axes.set_xlabel(r"x", labelpad = 8)
axes.set_ylabel(r"y", labelpad = 8)
#############################################


######################################################
color = 'r'
line = axes.plot(x_data, y_data, label = r"data")
plt.setp(line, ls ="", c = color, lw = 2, marker = "o", mfc = color, ms = 7, mec = color, mew= 2)
######################################################

######################################################
color = 'k'
line = axes.plot(x_fit, y_fit, label = r"fit")
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

script_name = os.path.basename(__file__)
figurename = script_name[0:-3] + ".pdf"

plt.show()
fig.savefig(figurename, bbox_inches = "tight")
plt.close()

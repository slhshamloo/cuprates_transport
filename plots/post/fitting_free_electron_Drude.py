import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import os
import sys
from lmfit import minimize, Parameters, report_fit
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

## Free parameters to vary
vary = {
    "omega_c": True,
    "omega_pn": True,
    "omega_ps": True,
    "gamma": True,
    "epsilon_inf": False
}


def plotOnce(folder, label, doping, T, B):
    ## Load Data /////////////////////////////////////////////////////////////////////
    fullpath = os.path.relpath(__file__)
    dirname, fname = os.path.split(fullpath)
    project_root = dirname + "/../../"
    datapath = project_root + f"data/{folder}"

    # Handle the real part
    x_data, y_data = [], []
    i = 1
    for j in ["r", "l"]:
        filename = f"{label}_sigma{i}{j}_{B}T.csv"
        data = np.loadtxt(f"{datapath}/{filename}", 
                          dtype="float", 
                          comments="#", 
                          delimiter=',', 
                          skiprows=1)
        x_data.append(data[:,0])
        y_data.append(data[:,1])

    x_data_r = np.concatenate(x_data)
    y_data_r = np.concatenate(y_data)

    # Handle the imaginary part
    x_data, y_data = [], []
    i = 2
    for j in ["r", "l"]:
        filename = f"{label}_sigma{i}{j}_{B}T.csv"
        data = np.loadtxt(f"{datapath}/{filename}", 
                          dtype="float", 
                          comments="#", 
                          delimiter=',', 
                          skiprows=1)
        x_data.append(data[:,0])
        y_data.append(data[:,1])

    x_data_i = np.concatenate(x_data)
    y_data_i = np.concatenate(y_data)


    ## Fit ///////////////////////////////////////////////////////////////////////////
    def fit_model(omega, gamma, omega_pn_sq, omega_c, omega_ps_sq, epsilon_inf=1):
        """Compute optical conductivity in the Drude model with given parameters

        All parameters must be given in THz and the result is in (mOhm.cm)^-1"""
        e0 = 8.854e-12
        return 1j * 1e7 * e0 * (omega_pn_sq /(omega - omega_c + 1j*gamma)
                                + omega_ps_sq / omega - omega * (epsilon_inf - 1))


    def compute_diff(pars, reals, imags, d_real, d_imag):
        gamma = pars["gamma"].value
        omega_pn_sq  = pars["omega_pn_sq"].value
        omega_c = pars["omega_c"].value
        omega_ps_sq = pars["omega_ps_sq"].value
        sim_r = np.real(fit_model(reals, gamma, omega_pn_sq, omega_c, omega_ps_sq))
        sim_imag = np.imag(fit_model(imags, gamma, omega_pn_sq, omega_c, omega_ps_sq))
        return np.concatenate([sim_r - d_real, sim_imag - d_imag])


    # Estimate initial value of omega_c
    e = 1.602e-19
    me = 9.109e-31
    omega_c_zero = 1e-12*e*B/(2*me) # Cyclotron frequency in rad.THz

    pars = Parameters()
    pars.add("gamma", value = 1, vary=vary["gamma"], min=0)
    pars.add("omega_pn_sq", value = 1e5, vary=vary["omega_pn"], min=0)
    pars.add("omega_ps_sq", value = 1e3, vary=vary["omega_ps"], min=0)
    pars.add("omega_c", value = 0.1,  vary=vary["omega_c"], min=0)
    pars.add("epsilon_inf", value = 1,  vary=vary["epsilon_inf"])


    out = minimize(compute_diff, pars, args=(x_data_r, x_data_i, y_data_r, y_data_i))

    gamma = out.params["gamma"].value # Gamma in rad.THz
    omega_pn_sq = out.params["omega_pn_sq"].value # OmegaPN in rad**2.THz**2
    omega_ps_sq = out.params["omega_ps_sq"].value
    omega_c = out.params["omega_c"].value # Omegac in rad.THz
    epsilon_inf= out.params["epsilon_inf"].value # 

    dataname = (project_root + f"user_data/{folder}/"
                + f"fitting_free_electron_Drude_{label}_{B}T.dat")

    with open(dataname, 'w') as sys.stdout:
        report_fit(out)


    x_min = -2
    x_max = 2

    x_fit = np.linspace(x_min, x_max, 1000)
    y_fit = fit_model(x_fit, gamma, omega_pn_sq, omega_c,
                      omega_ps_sq, epsilon_inf)


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

    fig, axs = plt.subplots(2, 1, figsize = (9.2, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure

    fig.subplots_adjust(left = 0.15, right = 0.82, bottom = 0.18, top = 0.98) # adjust the box of axes regarding the figure size

    # axes.axhline(y=0, ls ="--", c ="k", linewidth=0.6)

    #############################################
    fig.text(0.17,0.85, "LSCO\n"+f"$p={doping}$", ha="left")
    fig.text(0.79,0.28, fr"$B=${B} T", ha="right")
    fig.text(0.79,0.22, fr"$T=${label[-2:]} K", ha="right")
    #############################################
    tble = axs[0].table([[""],[fr"$\Gamma$ = {gamma:.3}"],
                         [r"$\omega_{\rm{c}}$"+f" = {omega_c:.3}"],
                         [r"$\omega_{\rm{ps}}$"+f" = {np.sqrt(omega_ps_sq):.1f}"],
                         [r"$\omega_{\rm{pn}}$"+f" = {np.sqrt(omega_pn_sq):.1f}"],
                         [r"$\epsilon_{\infty}$"+f" = {epsilon_inf}"],]
                         ,cellLoc="left", colWidths=[0.3], loc=14, fontsize=0.01, edges="open")

    tble[(1, 0)].get_text().set_color('red') if vary["gamma"] else tble[(1, 0)].get_text().set_color('black')
    tble[(2, 0)].get_text().set_color('red') if vary["omega_c"] else tble[(2, 0)].get_text().set_color('black')
    tble[(3, 0)].get_text().set_color('red') if vary["omega_ps"] else tble[(3, 0)].get_text().set_color('black')
    tble[(4, 0)].get_text().set_color('red') if vary["omega_pn"] else tble[(4, 0)].get_text().set_color('black')
    tble[(5, 0)].get_text().set_color('red') if vary["epsilon_inf"] else tble[(5, 0)].get_text().set_color('black')

    tble.scale(1, 3)
    tble.set_fontsize(16)
    #############################################
    for i in range(2):
        axs[i].set_xlim(1.2*x_min, 1.2*x_max)   # limit for xaxis
        # axes.set_ylim(-3.5,1.2) # leave the ymax auto, but fix ymin
        axs[i].tick_params(axis='x', which='major', pad=7)
        axs[i].tick_params(axis='y', which='major', pad=8)
    #############################################
    axs[0].set_ylabel(r"Re($\sigma$)", labelpad = 8)
    axs[1].set_ylabel(r"Im($\sigma$)", labelpad = 8)
    axs[1].set_xlabel(r"$\omega$ ( THz )", labelpad = 8)
    axs[0].get_xaxis().set_ticklabels([])

    ######################################################

    color = 'k'
    line = axs[0].plot(x_fit, np.real(y_fit), label = r"fit")
    plt.setp(line, ls ="--", c = color, lw = 2, marker = "", mfc = color, ms = 7, mec = color, mew= 2)


    indices = np.argsort(x_data_r)
    x_data_r = np.sort(x_data_r)
    y_data_r = np.take_along_axis(y_data_r, indices, axis=0)

    color = 'r'
    line = axs[0].plot(x_data_r[x_data_r>0.1], y_data_r[x_data_r>0.1], label = r"data")
    line1 = axs[0].plot(x_data_r[x_data_r<-0.1], y_data_r[x_data_r<-0.1], label = r"")
    plt.setp(line, ls ="-", c = color, lw = 3, marker = "", mfc = color, ms = 7, mec = color, mew= 2)
    plt.setp(line1, ls ="-", c = color, lw = 3, marker = "", mfc = color, ms = 7, mec = color, mew= 2)
    ######################################################

    ######################################################
    color = 'k'
    line = axs[1].plot(x_fit[x_fit>0.1], np.imag(y_fit[x_fit>0.1]), label = r"fit")
    line1 = axs[1].plot(x_fit[x_fit<-0.1], np.imag(y_fit[x_fit<-0.1]), label = r"")
    plt.setp(line, ls ="--", c = color, lw = 2, marker = "", mfc = color, ms = 7, mec = color, mew= 2)
    plt.setp(line1, ls ="--", c = color, lw = 2, marker = "", mfc = color, ms = 7, mec = color, mew= 2)

    indices = np.argsort(x_data_i)
    x_data_i = np.sort(x_data_i)
    y_data_i = np.take_along_axis(y_data_i, indices, axis=0)

    color = 'r'
    line = axs[1].plot(x_data_i[x_data_i>0.1], y_data_i[x_data_i>0.1], label = r"data")
    line1 = axs[1].plot(x_data_i[x_data_i<-0.1], y_data_i[x_data_i<-0.1], label = r"")
    plt.setp(line, ls ="-", c = color, lw = 3, marker = "", mfc = color, ms = 7, mec = color, mew= 2)
    plt.setp(line1, ls ="-", c = color, lw = 3, marker = "", mfc = color, ms = 7, mec = color, mew= 2)
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
    datapath = project_root+f"user_plots/{folder}"
    figurename = f"{datapath}/fitting_Drude_{label}_{B}T.pdf"

    fig.savefig(figurename, bbox_inches="tight")
    plt.close()


def plot_all_post():
    labels = ["post35", "post40", "post45", "post50"  ]
    dopings = [0.16, 0.16, 0.16, 0.16]
    temperatures = [35, 40, 45, 50]
    fields = [0, 9, 20, 31]
    for (label, doping, T) in zip(labels, dopings, temperatures):
        for B in fields:
            plotOnce("post", label, doping, T, B)


def plot_all_legros():
    labels = ["UD39K", "OD36K", "OD35K", "OD17K", "OD13K", "NSC"]
    dopings = [0.131, 0.202, 0.205, 0.244, 0.251, 0.263]
    temperatures = [35, 25, 20, 20, 15, 15]
    fields = [0, 4, 9, 14, 20, 31]
    for (label, doping, T) in zip(labels, dopings, temperatures):
        for B in fields:
            plotOnce("legros", label, doping, T, B)


if __name__ == "__main__":
    # plot_all_post()
    plot_all_legros()

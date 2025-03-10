import argparse
import run_fits
import plot_chambers_fits
import defaults

free_params = ['gamma_0', 'gamma_k', 'power', 'energy_scale']
default_params = ['gamma_0,1.0,50.0', 'gamma_k,1.0,500.0', 'power,1.0,50.0', 'energy_scale,50.0,100.0']


def fitting(args):
    ranges = dict()
    init_params = defaults.get_init_params()
    extraInfo = ""
    for param, lbound, ubound in args.parameters:
        ranges[param] = [lbound, ubound]
        extraInfo += f"{param}({round(lbound,1)}-{round(ubound,1)})"
    
    for param, value in args.initial:
        init_params[param] = value

    result = run_fits.run_single_fit(args.paper, args.sample, args.fields, ranges, init_params)
    if not args.textonly:
        run_fits.save_fit(result, args.sample, args.fields, extraInfo)

def exporting(args):
    pass

def plotting(args):
    # Bypass case
    if args.bypass:
        values = dict()
        for param, value in args.setparams:
            values[param] = value
            extraInfo += f"{param}(value)"

        plot_chambers_fits.plot_chambers_fit(args.paper, args.sample, args.fields, bypass_fit=True, save_fig=args.savefig)
    # Non bypass case
    else:
        plot_chambers_fits.plot_chambers_fit(args.paper, args.sample, args.fields, save_fig=args.savefig)


def parse_params(tri):
    try:
        param, lower_bound, upper_bound = tri.split(',')
        if param not in free_params: argparse.ArgumentTypeError("Invalid parameter name")
        return param, float(lower_bound), float(upper_bound) 
    except ValueError:
        raise argparse.ArgumentTypeError("Parameters to vary must be specified as 'str,float,float'")

def parse_overrides(pairs):
    try:
        param, lower_bound, upper_bound = pairs.split(',')
        if param not in free_params: argparse.ArgumentTypeError("Invalid parameter name")
        return param, float(lower_bound), float(upper_bound) 
    except ValueError:
        raise argparse.ArgumentTypeError("Parameters to set must be specified as 'str,float'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                                    prog="Chambers' formula fitting",
                                    description="This program handles fits of optical \
                                            conductivity data in LSCO samples using differential \
                                            evolution with a Chambers' formula model",
                                    epilog="")
    
    subparsers = parser.add_subparsers(help='TODO subcommand help')
    ### Subcommand allowing to perform all the fits
    fitting_parser = subparsers.add_parser('fit', help='Perform fits and save them to files')
    ## Required positional arguments
    fitting_parser.add_argument('paper', required=True,
                                help="The paper from which the sample to fit is extracted")
    fitting_parser.add_argument('sample', required=True,
                                help="The sample on which to perform the fit")
    fitting_parser.add_argument('fields',
                                nargs="+",  # expects ≥ 1 arguments
                                type=int,
                                required=True,
                                help="The field values (at least one expected) for \
                                        which to perform the fit. They will be fitted together.")    
    ## Optional flags
    fitting_parser.add_argument('-t', '--textonly',
                                action='store_true',
                                help="Disable saving the results")
    
    fitting_parser.add_argument('-P', '--parallel',
                                action='store_true',
                                nargs='?',
                                type=int,
                                const=-1,
                                default=1,
                                help="Whether to parallelize the process and how many processes to use")
    
    fitting_parser.add_argument('-p', '--parameters',
                                nargs='*',
                                type=parse_params,
                                default=default_params,
                                const=default_params,
                                help="Set parameters to vary")
    
    fitting_parser.add_argument('-i', '--initial',
                             nargs='*',
                             type=parse_overrides,
                             help="Set initial parameter values for fit to override defaults"
                             )

    
    fitting_parser.set_defaults(func=fitting)

    ## Subcommand for exporting the fit results in bulk
    exporting_parser = subparsers.add_parser('export', help='TODO exporting help')
    exporting_parser.set_defaults(func=exporting)

    ## Subcommand for plotting the fit results
    plotting_parser = subparsers.add_parser('plot', help='Use to plot figures of fits')
    plotting_parser.add_argument('paper', required=True,
                                help="The paper from which the sample to fit was extracted")
    plotting_parser.add_argument('sample', required=True,
                                help="The sample on which the fit was done")
    plotting_parser.add_argument('fields',
                                nargs="+",  # expects ≥ 1 arguments
                                type=int,
                                required=True,
                                help="The field values (at least one expected) for \
                                        at which the fit was done.")    
    ## Optional flags        
    plotting_parser.add_argument('-p', '--parameters',
                                nargs='*',
                                type=parse_params,
                                default=default_params,
                                const=default_params,
                                help="Select varied parameters")
    plotting_parser.add_argument('-o', '--override',
                                 nargs='*',
                                 type=parse_overrides,
                                 help="OVERRIDE all computed fit parameters with defaults and specified params"
                                 )
    plotting_parser.set_defaults(func=plotting)

    args = parser.parse_args()
    args.func(args)

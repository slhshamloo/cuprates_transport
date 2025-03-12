import argparse
import run_fits
import plot_chambers_fits
import defaults
import file_operations

free_params = ['gamma_0', 'gamma_k', 'power', 'energy_scale']

def compute_extra_info(parameters, initial, im, re):
    extra_info = ""
        
    if args.parameters is not None:
        for param, lbound, ubound in args.parameters:
            extra_info += f"{param}({round(lbound,1)}-{round(ubound,1)})"

    if args.initial is not None:
        extra_info += "__init"
        for param, value in args.initial:
            extra_info += f"{param}({round(value,1)})"

    if args.im:
        extra_info += "__im_only"
    if args.re:
        extra_info += "__re_only"
    
    return extra_info




def fitting(args):
    ## SETUP : Translate user-provided CL arguments to parameters
    # Generate unique string for chosen experiment
    extra_info = compute_extra_info(args.parameters, args.initial, args.im, args.re)

    # Setting custom bounds (and thus free parameters)
    ranges = dict()
    if args.parameters is not None:
        for param, lbound, ubound in args.parameters:
            ranges[param] = [lbound, ubound]
        print("Using custom ranges")
        print(ranges)
    else:
        print("No ranges provided, using default ranges")
        ranges = defaults.get_ranges()
        print(ranges)
    
    # Setting custom initial parameters
    init_params = defaults.get_init_params()
    if args.initial is not None:
        for param, value in args.initial:
            init_params[param] = value
    
    # The fitting mode allows to fit on Re or Im parts only
    fitting_mode = 0
    if args.re:
        fitting_mode = 1
    if args.im:
        fitting_mode = 2

    print(f"Fitting mode (0: full, 1: real only, 2: imaginary only): {fitting_mode}")
    ## RUN : Load and fit the data using previously determined settings
    # Load and interpolate the experimental data
    omegas, sigmas = run_fits.load_interp_multi_field(
        args.paper, args.sample, args.fields, nsample_polarity=20)
    # Run the fitting procedure
    fit_result = run_fits.run_fit_multi_field_parallel(
        args.fields, omegas, sigmas, init_params, ranges, fitting_mode)

    ## OUTPUT : Save and display results
    if not args.textonly:
        file_operations.save_fit(fit_result, args.sample, args.fields, extra_info)

def exporting(args):
    pass

def plotting(args):
    # Bypass case
    if args.override is not None:
        print("Plotting with custom values (provided override defaults)")
        extra_info = "fixed_"
        values = defaults.get_init_params()
        for param, value in args.override:
            values[param] = value
            extra_info += f"{param}({round(value,1)})"
        plot_chambers_fits.from_parameters(args.paper, args.sample, args.fields, extra_info, None, values)
    # Non bypass case
    else:
        print("Plotting from fitted values")
        non_override = dict()
        if args.initial is not None:
            for param, value in args.initial:
                non_override[param] = value

        extra_info = compute_extra_info(args.parameters, args.initial, args.im, args.re)
        plot_chambers_fits.from_parameters(args.paper, args.sample, args.fields, extra_info, non_override)


def parse_params(tri):
    try:
        param, lower_bound, upper_bound = tri.split(',')
        if param not in free_params: argparse.ArgumentTypeError("Invalid parameter name")
        return param, float(lower_bound), float(upper_bound) 
    except ValueError:
        raise argparse.ArgumentTypeError("Parameters to vary must be specified as 'str,float,float'")

def parse_overrides(pairs):
    try:
        param, value = pairs.split(',')
        if param not in free_params: argparse.ArgumentTypeError("Invalid parameter name")
        return param, float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Parameters to set must be specified as 'str,float'")

def error_no_command(args):
    raise argparse.ArgumentError(None, "Choose a command")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                                    prog="Chambers' formula fitting",
                                    description="This program handles fits of optical \
                                            conductivity data in LSCO samples using differential \
                                            evolution with a Chambers' formula model",
                                    epilog="")
    
    parser.set_defaults(func=error_no_command)
    
    subparsers = parser.add_subparsers(help='TODO subcommand help')
    ### Subcommand allowing to perform all the fits
    fitting_parser = subparsers.add_parser('fit', help='Perform fits and save them to files')
    ## Required positional arguments
    fitting_parser.add_argument('paper',
                                help="The paper from which the sample to fit is extracted")
    fitting_parser.add_argument('sample',
                                help="The sample on which to perform the fit")
    fitting_parser.add_argument('fields',
                                nargs="+",  # expects ≥ 1 arguments
                                type=int,
                                help="The field values (at least one expected) for \
                                        which to perform the fit. They will be fitted together.")    
    ## Optional flags
    fitting_parser.add_argument('-t', '--textonly',
                                action='store_true',
                                help="Disable saving the results")
    
    fitting_parser.add_argument('-P', '--parallel',
                                nargs='?',
                                type=int,
                                const=-1,
                                default=1,
                                help="Whether to parallelize the process and how many processes to use")
    
    fitting_parser.add_argument('-p', '--parameters',
                                nargs='*',
                                type=parse_params,
                                help="Set parameters to vary")
    
    fitting_parser.add_argument('-i', '--initial',
                             nargs='*',
                             type=parse_overrides,
                             help="Set initial parameter values for fit to override defaults"
                             )
    
    group = fitting_parser.add_mutually_exclusive_group()
    group.add_argument('-I', '--im', action='store_true', help="Fit imaginary part only")
    group.add_argument('-R', '--re', action='store_true', help="Fit real part only")

    
    fitting_parser.set_defaults(func=fitting)

    ## Subcommand for exporting the fit results in bulk
    exporting_parser = subparsers.add_parser('export', help='TODO exporting help')
    exporting_parser.set_defaults(func=exporting)

    ## Subcommand for plotting the fit results
    plotting_parser = subparsers.add_parser('plot', help='Use to plot figures of fits')
    plotting_parser.add_argument('paper',
                                help="The paper from which the sample to fit was extracted")
    plotting_parser.add_argument('sample',
                                help="The sample on which the fit was done")
    plotting_parser.add_argument('fields',
                                nargs="+",  # expects ≥ 1 arguments
                                type=int,
                                help="The field values (at least one expected) for \
                                        at which the fit was done.")    
    ## Optional flags 
    group2 = plotting_parser.add_mutually_exclusive_group()
    group2.add_argument('-I', '--im', action='store_true', help="Fit imaginary part only")
    group2.add_argument('-R', '--re', action='store_true', help="Fit real part only")
       
    plotting_parser.add_argument('-p', '--parameters',
                                nargs='*',
                                type=parse_params,
                                help="Parameters that were varied")
    
    plotting_parser.add_argument('-i', '--initial',
                             nargs='*',
                             type=parse_overrides,
                             help="Initial parameter values that were provided"
                             )
    plotting_parser.add_argument('-o', '--override',
                             nargs='*',
                             type=parse_overrides,
                             help="OVERRIDE and BYPASS fitting, plot using defaults and provided parameters")
    plotting_parser.set_defaults(func=plotting)

    args = parser.parse_args()
    args.func(args)

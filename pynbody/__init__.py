#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyNbody

A Python Toolkit for Astrophysical N-Body Simulations.
"""

import sys as _sys

from . import analysis
from . import integrator
from . import io
from . import lib
from . import models
from . import particles
from . import simulation
from . import tests
from . import version


def _main():
    """
    The PyNbody's main function.

    Here we process command line arguments and call specific functions to run a
    new N-body simulation or restart from a previous run.

    NOTE: You shouldn't be able to call this function from a python session.
          Instead you must call pynbody's script directly from a unix shell.
    """
    import argparse

    # create the parser
    parser = argparse.ArgumentParser(description="A Python Toolkit for "
                                     "Astrophysical N-Body Simulations.")
    subparser = parser.add_subparsers(help="Consult specific help for details.")

    # --------------------------------------------------------------------------
    # add subparser newrun
    newrun = subparser.add_parser("newrun",
                                  description="Performs a new N-body simulation.")
    # add the arguments to newrun
    newrun.add_argument("-i", "--input",
                        type=str,
                        default=None,
                        required=True,
                        help="The name of the initial conditions file which "
                             "must be read from (type: str, default: None)."
                       )
    newrun.add_argument("-e", "--eta",
                        type=float,
                        default=None,
                        required=True,
                        help="Parameter for time step determination (type: flo"
                             "at, default: None)."
                       )
    newrun.add_argument("-t", "--tmax",
                        type=float,
                        default=None,
                        required=True,
                         help="Time to end the simulation (type: float, default"
                              ": None)."
                       )
    newrun.add_argument("-m", "--meth",
                        type=str,
                        default="leapfrog",
                        choices=integrator.Integrator.METHS.keys(),
                        help="Integration method name (type: str, default:"
                             " 'leapfrog')."
                       )
    newrun.add_argument("--io_interface",
                        type=str,
                        default="hdf5io",
                        choices=io.IO.INTERFACES.keys(),
                        help="Data Input/Output Interface (type: str, default: "
                             "'hdf5io')."
                       )
    newrun.add_argument("--log_file",
                        type=str,
                        default=_sys.stdout,
                        help="File name where log messages should be written ("
                             "type: str, default: sys.stdout)."
                       )
    newrun.add_argument("-d", "--diag_freq",
                        type=int,
                        default=4,
                        help="Diagnostic frequency of the simulation per time "
                             "unit (type: int, default: 4)."
                       )
    newrun.add_argument("-r", "--res_freq",
                        type=int,
                        default=1,
                        help="Frequency of rewriting the resumption file per t"
                             "ime unit (type: int, default: 1)."
                       )
    newrun.add_argument("--view",
                        action="store_true",
                        help="Enable visualization of the simulation in real time."
                       )
    newrun.add_argument("-g", "--gl_freq",
                        type=int,
                        default=128,
                        help="GLviewer event frequency per time unit (type: in"
                             "t, default: 128)."
                       )
    newrun.add_argument("--debug",
                        action="store_true",
                        help="Enable debug messages."
                       )
    newrun.add_argument("--debug_file",
                        type=str,
                        default=_sys.stderr,
                        help="File name where error messages should be written"
                             " (type: str, default: sys.stderr)."
                       )
#    newrun.add_argument("-o", "--out",
#                        type=int,
#                        default=None,
#                        help="Number of individual time steps between each par"
#                             "ticle output (type: int, default: None)."
#                             " XXX: NOT IMPLEMENTED."
#                       )
    newrun.set_defaults(func=_main_newrun)

    # --------------------------------------------------------------------------
    # add subparser restart
    restart = subparser.add_parser("restart",
                        description="Restart a simulation from a previous run.")
    # add the arguments to restart
    restart.add_argument("-t", "--tmax",
                         type=float,
                         default=None,
                         required=True,
                         help="Time to end the simulation (type: float, default"
                              ": None)."
                        )
    restart.add_argument("-e", "--eta",
                         type=float,
                         default=None,
                         help="Parameter for time step determination (type: flo"
                              "at, default: obtained from the restart file)."
                        )
    restart.add_argument("-r", "--restart_file",
                         type=str,
                         default="restart.pkl.gz",
                         help="The name of the restart file which must be read "
                              "from (type: str, default: 'restart.pkl.gz')."
                        )

    restart.set_defaults(func=_main_restart)

    # --------------------------------------------------------------------------
    # parse the command line
    args = parser.parse_args()

    # call the appropriate function
    args.func(args)


def _main_newrun(args):

    if args.log_file == _sys.stdout:
        args.log_file = _sys.stdout.name

    if args.debug_file == _sys.stderr:
        args.debug_file = _sys.stderr.name

    # --------------------------------------------------------------------------

    viewer = analysis.glviewer.GLviewer() if args.view else None
    mysim = simulation.Simulation(args, viewer)
    mysim.evolve()
    mysim.print_timings()
    return 0


def _main_restart(args):
    import gzip
    import pickle
#    with gzip.open("restart.pkl.gz", "rb") as fobj:
#        mysim = pickle.load(fobj)

    # <py2.6>
    fobj = gzip.open(args.restart_file, "rb")
    mysim = pickle.load(fobj)
    fobj.close()
    # </py2.6>

    # update args
    mysim.args.tmax = args.tmax
    if not args.eta is None:
        mysim.integrator._meth.eta = args.eta

    mysim.evolve()
    mysim.print_timings()
    return 0


########## end of file ##########

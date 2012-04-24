#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TODO.
"""

from __future__ import print_function
import sys
import math
import pickle
import argparse
from pprint import pprint
from .io import IO
from .analysis.glviewer import GLviewer
from .integrator import Integrator
from .lib.extensions import kernel_library
from .lib.utils.timing import timings


__all__ = ['Simulation']


@timings
def myprint(data, fname, fmode):
    if fname == '<stdout>':
        print(data, file=sys.stdout)
    elif fname == '<stderr>':
        print(data, file=sys.stderr)
    else:
        with open(fname, fmode) as fobj:
            print(data, file=fobj)




class Diagnostic(object):
    """

    """
    def __init__(self, fname, particles):
        self.time = 0.0
        self.fname = fname

        particles.update_phi(particles)
        self.ke0 = particles.get_total_kinetic_energy()
        self.pe0 = particles.get_total_potential_energy()
        self.te0 = self.ke0 + self.pe0
        self.ve0 = self.ke0 + self.te0

        self.rcom0 = particles.get_center_of_mass_position()
        self.vcom0 = particles.get_center_of_mass_velocity()
        self.lmom0 = particles.get_total_linear_momentum()
        self.amom0 = particles.get_total_angular_momentum()

        self.ceerr = 0.0
        self.count = 0

        self.print_header()


    def __repr__(self):
        return '{0}'.format(self.__dict__)


    def print_header(self):
        fmt = '{0:13s} {1:10s} '\
              '{2:10s} {3:10s} {4:15s} '\
              '{5:10s} {6:10s} {7:10s} '\
              '{8:10s} {9:10s} {10:10s} '\
              '{11:10s}'
        myprint(fmt.format('#00:time', '#01:tstep',
                           '#02:ke', '#03:pe', '#04:te',
                           '#05:ve', '#06:eerr', '#07:geerr',
                           '#08:rcom', '#09:vcom', '#10:lmom',
                           '#11:amom'),
                self.fname, 'w')

    @timings
    def print_diagnostic(self, time, tstep, particles):
        self.time = time

        particles.update_phi(particles)
        ke = particles.get_total_kinetic_energy()
        pe = particles.get_total_potential_energy()
        te = ke + pe
        ve = ke + te

        rcom = particles.get_center_of_mass_position()
        vcom = particles.get_center_of_mass_velocity()
        lmom = particles.get_total_linear_momentum()
        amom = particles.get_total_angular_momentum()

        eerr = (te-self.te0)/(-pe)
        self.ceerr += eerr**2
        self.count += 1
        geerr = math.sqrt(self.ceerr / self.count)
        dRcom = (((rcom-self.rcom0)**2).sum())**0.5
        dVcom = (((vcom-self.vcom0)**2).sum())**0.5
        dLmom = (((lmom-self.lmom0)**2).sum())**0.5
        dAmom = (((amom-self.amom0)**2).sum())**0.5

        fmt = '{time:< 13.6e} {tstep:< 10.3e} '\
              '{ke:< 10.3e} {pe:< 10.3e} {te:< 15.8e} '\
              '{ve:< 10.3e} {eerr:< 10.3e} {geerr:< 10.3e} '\
              '{rcom:< 10.3e} {vcom:< 10.3e} {lmom:< 10.3e} '\
              '{amom:< 10.3e}'
        myprint(fmt.format(time=time, tstep=tstep,
                           ke=ke, pe=pe,
                           te=te, ve=ve,
                           eerr=eerr, geerr=geerr, rcom=dRcom,
                           vcom=dVcom, lmom=dLmom, amom=dAmom),
                self.fname, 'a')



class Simulation(object):
    """
    The Simulation class is the top level class for N-body simulations.
    """
    @timings
    def __init__(self, args, viewer):
        self.args = args
        self.viewer = viewer

        print('#'*40, file=sys.stderr)
        pprint(args.__dict__, stream=sys.stderr)
        print('#'*40, file=sys.stderr)

        # Read the initial conditions.
        fname = self.args.input_file
        particles = IO(fname).load()

        # Initializes snapshots output.
        self.io = IO("snapshots", output_format=self.args.output_format)

        # Initializes the integrator.
        self.integrator = Integrator(self.args.eta, self.args.t_begin, particles,
                                     method=self.args.meth, dumpper=self.io,
                                     snap_freq=self.args.snap_freq)
        self.integrator.init_for_integration(self.args.t_end)

        # Initializes the diagnostic of the simulation.
        self.dia = Diagnostic(self.args.log_file, particles)
        self.dia.print_diagnostic(self.integrator.current_time,
                                  self.integrator.tstep,
                                  particles)

        # Initializes some counters.
        self.gl_steps = 0
        self.res_steps = 0
        self.diag_steps = 0


    @timings
    def dump_restart_file(self):
        if sys.version_info >= (2, 7):
            with open(self.args.restart_file, 'wb') as fobj:
                pickle.dump(self, fobj, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            fobj = open(self.args.restart_file, 'wb')
            pickle.dump(self, fobj, protocol=pickle.HIGHEST_PROTOCOL)
            fobj.close()


    @timings
    def evolve(self):
        """

        """
        if self.viewer:
            self.viewer.initialize()

        while (self.integrator.current_time < self.args.t_end):
            self.integrator.step(self.args.t_end)
            self.gl_steps += 1
            if (self.gl_steps >= self.args.gl_freq):
                self.gl_steps -= self.args.gl_freq
                if self.viewer:
                    particles = self.integrator.particles
                    self.viewer.show_event(particles)
            self.diag_steps += 1
            if (self.diag_steps >= self.args.diag_freq):
                self.diag_steps -= self.args.diag_freq
                particles = self.integrator.particles
                self.dia.print_diagnostic(self.integrator.current_time,
                                          self.integrator.tstep,
                                          particles)
            self.res_steps += 1
            if (self.res_steps >= self.args.res_freq):
                self.res_steps -= self.args.res_freq
                self.dump_restart_file()

        # final IO/diag operations
        if self.dia.time < self.args.t_end:
            particles = self.integrator.particles
            self.dia.print_diagnostic(self.integrator.current_time,
                                  self.integrator.tstep,
                                  particles)

#        self.io.dump(particles)
        self.dump_restart_file()
        if self.viewer:
            particles = self.integrator.particles
            self.viewer.show_event(particles)
            self.viewer.enter_main_loop()


# ------------------------------------------------------------------------------


def _main_newrun(args):
    if args.log_file == sys.stdout:
        args.log_file = sys.stdout.name

    if args.debug_file == sys.stderr:
        args.debug_file = sys.stderr.name

    # --------------------------------------------------------------------------

    kernel_library.build_kernels(use_cl=args.use_cl)

    viewer = GLviewer() if args.view else None
    mysim = Simulation(args, viewer)
    mysim.evolve()
    return 0


def _main_restart(args):
    if sys.version_info >= (2, 7):
        with open(args.restart_file, "rb") as fobj:
            mysim = pickle.load(fobj)
    else:
        fobj = open(args.restart_file, "rb")
        mysim = pickle.load(fobj)
        fobj.close()

    # update args
    mysim.args.t_end = args.t_end
    if not args.eta is None:
        mysim.integrator._meth.eta = args.eta

    args.use_cl = mysim.args.use_cl
    kernel_library.build_kernels(use_cl=args.use_cl)

    mysim.evolve()
    return 0


def main():
    """
    The PyNbody's main function.

    Here we process command line arguments and call specific functions to run a
    new N-body simulation or restart from a previous run.

    NOTE: You shouldn't be able to call this function from a ipython session.
          Instead you must call pynbody's script directly from a unix shell.
    """

    # create the parser
    parser = argparse.ArgumentParser(description="A Python Toolkit for "
                                     "Astrophysical N-Body Simulations.")
    subparser = parser.add_subparsers(help="Consult specific help for details.")

    # --------------------------------------------------------------------------
    # add subparser newrun
    newrun = subparser.add_parser("newrun",
                                  description="Performs a new N-body simulation.")
    # add the arguments to newrun
    newrun.add_argument("-i", "--input_file",
                        type=str,
                        default=None,
                        required=True,
                        help="The name of the initial conditions file which "
                             "must be read from. The file format, if supported, "
                             "is automatically discovered (type: str, default: None)."
                       )
    newrun.add_argument("-e", "--eta",
                        type=float,
                        default=None,
                        required=True,
                        help="Parameter for time step determination (type: flo"
                             "at, default: None)."
                       )
    newrun.add_argument("-t", "--t_end",
                        type=float,
                        default=None,
                        required=True,
                        help="Time to end the simulation (type: float, default"
                             ": None)."
                       )
    newrun.add_argument("--t_begin",
                        type=float,
                        default=0.0,
                        help="Time to begin the simulation (type: float, default"
                             ": 0.0)."
                       )
    newrun.add_argument("-m", "--meth",
                        type=str,
                        default="leapfrog",
                        choices=Integrator.PROVIDED_METHODS,
                        help="Integration method name (type: str, default:"
                             " 'leapfrog')."
                       )
    newrun.add_argument("-o", "--output_format",
                        type=str,
                        default="hdf5",
                        choices=IO.PROVIDED_FORMATS,
                        help="Output format to store the particle stream. "
                             "(type: str, default: 'hdf5')."
                       )
    newrun.add_argument("--log_file",
                        type=str,
                        default=sys.stdout,
                        help="File name where log messages should be written ("
                             "type: str, default: sys.stdout)."
                       )
    newrun.add_argument("-d", "--diag_freq",
                        type=int,
                        default=64,
                        help="Number of time-steps between diagnostic of the "
                             "simulation (type: int, default: 64)."
                       )
    newrun.add_argument("-s", "--snap_freq",
                        type=int,
                        default=16,
                        help="Number of time-steps between snapshots of the "
                             "simulation (type: int, default: 16)."
                       )
    newrun.add_argument("-r", "--res_freq",
                        type=int,
                        default=1024,
                        help="Number of time-steps between rewrites of the "
                             "restart file (type: int, default: 1024)."
                       )
    newrun.add_argument("--use_cl",
                        action="store_true",
                        help="Enable OpenCL support."
                       )
    newrun.add_argument("--view",
                        action="store_true",
                        help="Enable visualization of the simulation in real time."
                       )
    newrun.add_argument("-g", "--gl_freq",
                        type=int,
                        default=2,
                        help="Number of time-steps between GLviewer events "
                             "(type: int, default: 2)."
                       )
    newrun.add_argument("--debug",
                        action="store_true",
                        help="Enable debug messages."
                       )
    newrun.add_argument("--debug_file",
                        type=str,
                        default=sys.stderr,
                        help="File name where error messages should be written"
                             " (type: str, default: sys.stderr)."
                       )
    newrun.add_argument("--restart_file",
                        type=str,
                        default="restart.pkl",
                        help="The name of the restart file which must be read "
                             "from (type: str, default: 'restart.pkl')."
                       )
    newrun.set_defaults(func=_main_newrun)

    # --------------------------------------------------------------------------
    # add subparser restart
    restart = subparser.add_parser("restart",
                        description="Restart a simulation from a previous run.")
    # add the arguments to restart
    restart.add_argument("-t", "--t_end",
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
    restart.add_argument("--restart_file",
                         type=str,
                         default="restart.pkl",
                         help="The name of the restart file which must be read "
                              "from (type: str, default: 'restart.pkl')."
                        )
    restart.set_defaults(func=_main_restart)

    # --------------------------------------------------------------------------
    # parse the command line
    args = parser.parse_args()

    # call the appropriate function
    args.func(args)


########## end of file ##########

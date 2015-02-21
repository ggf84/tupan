# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
import sys
import math
import pickle
import logging
from .io import HDF5IO
from .integrator import Integrator
from .analysis.glviewer import GLviewer
from .lib.utils.timing import timings, bind_all, Timer


LOGGER = logging.getLogger(__name__)


__all__ = ['Simulation']


@bind_all(timings)
class Diagnostic(object):
    """

    """
    def __init__(self, time, report_freq=4, pn_order=0):
        self.time = time
        self.report_freq = report_freq
        self.include_pn_corrections = True if pn_order else False
        self.nreport = 0
        self.is_initialized = False

    def initialize(self, ps):
        self.ke0 = ps.kinetic_energy
        self.pe0 = ps.potential_energy
        self.te0 = self.ke0 + self.pe0

        self.com_r0 = ps.com_r
        self.com_v0 = ps.com_v
        self.lmom0 = ps.linear_momentum
        self.amom0 = ps.angular_momentum

        self.count = 0
        self.ceerr = 0.0

        self.timer = Timer()
        self.timer.start()

        self.print_header()

        self.is_initialized = True

    def __repr__(self):
        return '{0}'.format(self.__dict__)

    def print_header(self):
        fmt = '{0:13s} {1:10s} '\
              '{2:10s} {3:10s} {4:15s} '\
              '{5:10s} {6:10s} {7:10s} '\
              '{8:10s} {9:10s} {10:10s} '\
              '{11:10s} {12:13s}'
        print(fmt.format('#00:time', '#01:dtime',
                         '#02:ke', '#03:pe', '#04:te',
                         '#05:virial', '#06:eerr', '#07:geerr',
                         '#08:com_r', '#09:com_v', '#10:lmom',
                         '#11:amom', '#12:wct'),
              file=sys.stdout)

    def diagnostic_report(self, ps):
        if not self.is_initialized:
            self.initialize(ps)
        t_curr = ps.t_curr
        if self.nreport % self.report_freq == 0:
            self.print_diagnostic(t_curr, t_curr - self.time, ps)
        self.nreport += 1

    def print_diagnostic(self, time, dtime, ps):
        self.time = time

        ke = ps.kinetic_energy
        pe = ps.potential_energy
        te = ke + pe
        virial = ps.virial_energy

        com_r = ps.com_r
        com_v = ps.com_v
        lmom = ps.linear_momentum
        amom = ps.angular_momentum

        eerr = (te-self.te0)/(-pe)
        self.count += 1
        self.ceerr += eerr**2
        geerr = math.sqrt(self.ceerr / self.count)
        com_dr = (((com_r-self.com_r0)**2).sum())**0.5
        com_dv = (((com_v-self.com_v0)**2).sum())**0.5
        dlmom = (((lmom-self.lmom0)**2).sum())**0.5
        damom = (((amom-self.amom0)**2).sum())**0.5

        fmt = '{time:< 13.6e} {dtime:< 10.3e} '\
              '{ke:< 10.3e} {pe:< 10.3e} {te:< 15.8e} '\
              '{virial:< 10.3e} {eerr:< 10.3e} {geerr:< 10.3e} '\
              '{com_r:< 10.3e} {com_v:< 10.3e} {lmom:< 10.3e} '\
              '{amom:< 10.3e} {wct:< 13.6e}'
        print(fmt.format(time=time, dtime=dtime,
                         ke=ke, pe=pe,
                         te=te, virial=virial,
                         eerr=eerr, geerr=geerr, com_r=com_dr,
                         com_v=com_dv, lmom=dlmom, amom=damom,
                         wct=self.timer.elapsed()),
              file=sys.stdout)


@bind_all(timings)
class Simulation(object):
    """
    The Simulation class is the top level class for N-body simulations.
    """
    def __init__(self, args, viewer):
        self.args = args

        # Read the initial conditions
        fname = self.args.input_file
        with HDF5IO(fname, 'r') as fid:
            ps = fid.read_ic()

        # Initializes output file
        fname = self.args.output_file
        io = HDF5IO(fname, 'w') if fname else None

        # Initializes the diagnostic report of the simulation
        self.dia = Diagnostic(
            self.args.t_begin,
            report_freq=self.args.report_freq,
            pn_order=self.args.pn_order,
            )

        # Initializes the integrator
        self.integrator = Integrator(
            self.args.eta,
            self.args.t_begin,
            ps,
            method=self.args.meth,
            pn_order=self.args.pn_order,
            clight=self.args.clight,
            reporter=self.dia,
            viewer=viewer,
            dumpper=io,
            dump_freq=self.args.dump_freq,
            gl_freq=self.args.gl_freq,
            )

        # Initializes some counters
        self.res_steps = 0

    def dump_restart_file(self):
        with open(self.args.restart_file, 'wb') as fobj:
            pickle.dump(self, fobj, protocol=pickle.HIGHEST_PROTOCOL)

    def evolve(self):
        """

        """
        while abs(self.integrator.time) < self.args.t_end:
            # evolve a single time-step
            self.integrator.evolve_step(self.args.t_end)

            # dump restart file
            if self.res_steps % self.args.restart_freq == 0:
                self.dump_restart_file()
            self.res_steps += 1

        # Finalize the integrator
        self.integrator.finalize(self.args.t_end)


# ------------------------------------------------------------------------


@timings
def main_simulation(args):
    viewer = GLviewer() if args.view else None
    mysim = Simulation(args, viewer)
    mysim.evolve()
    return 0


@timings
def main_restart(args):
    with open(args.restart_file, 'rb') as fobj:
        mysim = pickle.load(fobj)

    ps = mysim.integrator.integrator.ps

    if args.fpwidth == 'fp32':
        # FIXME: recast particles' attributes to fp32
        pass

    # reset eta
    mysim.integrator.integrator.eta = args.eta

    # reset t_end
    type(ps).t_curr = mysim.args.t_end
    mysim.args.t_end = args.t_end

    # reset timer
    mysim.dia.timer.reset_at(mysim.dia.timer.toc)

    viewer = None
    if args.view:
        viewer = mysim.integrator.integrator.viewer
        if not viewer:
            viewer = GLviewer()
    mysim.integrator.integrator.viewer = viewer
    if viewer:
        viewer.exitgl = False
        viewer.is_initialized = False
        viewer.show_event(ps)

    mysim.evolve()
    return 0


@timings
def add_parsers(subparser, parents=None):
    """Here we process the command line arguments to run a new N-body
    simulation or restart from a previous run.
    """
    import argparse
    preparser = argparse.ArgumentParser(add_help=False)
    preparser.add_argument(
        '-e', '--eta',
        type=float,
        required=True,
        help='Time-step parameter (type: %(type)s, required: %(required)s).'
        )
    preparser.add_argument(
        '-t', '--t_end',
        type=float,
        required=True,
        help='Simulation end time (type: %(type)s, required: %(required)s).'
        )
    preparser.add_argument(
        '--restart_file',
        type=str,
        default='restart.pkl',
        help='Restart filename (type: %(type)s, default: %(default)s).'
        )

    if parents is None:
        parents = []
    parents.append(preparser)

    # --- subparser restart ---
    description = 'Restart a simulation from a previous run.'

    # add subparser restart
    restart = subparser.add_parser(
        'restart',
        description=description,
        help=description,
        parents=parents
        )
    # add the arguments to restart
    restart.set_defaults(func=main_restart)

    # -------------------------------------------------------------------------

    # --- subparser simulation ---
    description = 'Performs a new N-body simulation.'

    # add subparser simulation
    simulation = subparser.add_parser(
        'simulation',
        description=description,
        help=description,
        parents=parents
        )
    # add the arguments to simulation
    simulation.add_argument(
        '-i', '--input_file',
        type=str,
        required=True,
        help=('Initial conditions filename '
              '(type: %(type)s, required: %(required)s).')
        )
    simulation.add_argument(
        '-m', '--meth',
        metavar='METH',
        type=str,
        required=True,
        choices=Integrator.PROVIDED_METHODS,
        help=('Integration method name '
              '(type: %(type)s, required: %(required)s, '
              'choices: {%(choices)s}).')
        )
    simulation.add_argument(
        '-o', '--output_file',
        type=str,
        default='',
        help=('Output filename to store the simulation data '
              '(type: %(type)s, default: %(default)s).')
        )
    simulation.add_argument(
        '--t_begin',
        type=float,
        default=0.0,
        help='Simulation begin time (type: %(type)s, default: %(default)s).'
        )
    simulation.add_argument(
        '--pn_order',
        metavar='PN_ORDER',
        type=int,
        default=0,
        choices=[0, 2, 4, 5, 6, 7],
        help=('Order of the Post-Newtonian corrections '
              '(type: %(type)s, default: %(default)s, '
              'choices: {%(choices)s}).')
        )
    simulation.add_argument(
        '--clight',
        type=float,
        default=float('inf'),
        help=('Speed of light value to use in Post-Newtonian corrections '
              '(type: %(type)s, default: %(default)s).')
        )
    simulation.add_argument(
        '-r', '--report_freq',
        type=int,
        default=4,
        help=('Number of time-steps between diagnostic reports of the '
              'simulation (type: %(type)s, default: %(default)s).')
        )
    simulation.add_argument(
        '-d', '--dump_freq',
        type=int,
        default=16,
        help=('Number of time-steps between dump of snapshots '
              '(type: %(type)s, default: %(default)s).')
        )
    simulation.add_argument(
        '--restart_freq',
        type=int,
        default=1,
        help=('Number of time-steps between rewrites of the restart file '
              '(type: %(type)s, default: %(default)s).')
        )
    simulation.add_argument(
        '-g', '--gl_freq',
        type=int,
        default=1,
        help=('Number of time-steps between GLviewer events '
              '(type: %(type)s, default: %(default)s).')
        )
    simulation.set_defaults(func=main_simulation)


# -- End of File --

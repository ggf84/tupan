# -*- coding: utf-8 -*-
#

"""
TODO.
"""

from __future__ import print_function
import sys
import pickle
import logging
from itertools import count
from .io.hdf5io import HDF5IO
from .integrator import Integrator
from .animation import GLviewer
from .lib.utils.timing import Timer


LOGGER = logging.getLogger(__name__)


class Diagnostic(object):
    """

    """
    def __init__(self, time):
        self.time = time

    def init_diagnostic_report(self, ps):
        self.ke0 = ps.kinetic_energy
        self.pe0 = ps.potential_energy
        self.te0 = self.ke0 + self.pe0

        self.com_r0 = ps.com_r
        self.com_v0 = ps.com_v
        self.lmom0 = ps.linear_momentum
        self.amom0 = ps.angular_momentum

        self.avr_err_sum = 0.0
        self.abs_err_sum = 0.0
        self.nreport = count(1)

        self.timer = Timer()
        self.timer.start()

        self.print_header()
        self.diagnostic_report(ps)

    def __repr__(self):
        return '{0}'.format(self.__dict__)

    def print_header(self):
        fmt = '{0:13s} {1:10s} '\
              '{2:10s} {3:10s} {4:15s} '\
              '{5:10s} {6:10s} {7:10s} '\
              '{8:10s} {9:10s} {10:10s} '\
              '{11:10s} {12:10s} {13:13s}'
        print(fmt.format('#00:time', '#01:dtime',
                         '#02:ke', '#03:pe', '#04:te',
                         '#05:virial', '#06:eerr', '#07:abserr',
                         '#08:avrerr', '#09:com_r', '#10:com_v',
                         '#11:lmom', '#12:amom', '#13:wct'),
              file=sys.stdout)

    def diagnostic_report(self, ps):
        time = ps.time[0]
        self.print_diagnostic(time, ps)

    def print_diagnostic(self, time, ps):
        dtime = time - self.time
        self.time = time

        ke = ps.kinetic_energy
        pe = ps.potential_energy
        te = ke + pe
        virial = ps.virial_energy

        com_r = ps.com_r
        com_v = ps.com_v
        lmom = ps.linear_momentum
        amom = ps.angular_momentum

        eerr = (te - self.te0) / (-pe)
        self.abs_err_sum += abs(eerr)
        self.avr_err_sum += eerr
        nreport = next(self.nreport)
        abs_err = self.abs_err_sum / nreport
        avr_err = self.avr_err_sum / nreport
        com_dr = (((com_r - self.com_r0)**2).sum())**0.5
        com_dv = (((com_v - self.com_v0)**2).sum())**0.5
        dlmom = (((lmom - self.lmom0)**2).sum())**0.5
        damom = (((amom - self.amom0)**2).sum())**0.5

        fmt = '{time:< 13.6e} {dtime:< 10.3e} '\
              '{ke:< 10.3e} {pe:< 10.3e} {te:< 15.8e} '\
              '{virial:< 10.3e} {eerr:< 10.3e} {abs_err:< 10.3e} '\
              '{avr_err:< 10.3e} {com_r:< 10.3e} {com_v:< 10.3e} '\
              '{lmom:< 10.3e} {amom:< 10.3e} {wct:< 13.6e}'
        print(fmt.format(time=time, dtime=dtime,
                         ke=ke, pe=pe,
                         te=te, virial=virial,
                         eerr=eerr, abs_err=abs_err,
                         avr_err=avr_err, com_r=com_dr,
                         com_v=com_dv, lmom=dlmom, amom=damom,
                         wct=self.timer.elapsed()),
              file=sys.stdout)


class Simulation(object):
    """
    The Simulation class is the top level class for N-body simulations.
    """
    def __init__(self, args, viewer):
        self.args = args

        # Read the initial conditions
        fname = self.args.input_file
        with HDF5IO(fname, 'r') as fid:
            ps = fid.load_snap()

        # Initializes output file
        fname = self.args.output_file
        io = HDF5IO(fname, 'w') if fname else None

        # Initializes the diagnostic report of the simulation
        self.dia = Diagnostic(self.args.t_begin)

        # Initializes the integrator
        self.integrator = Integrator(
            ps,
            self.args.eta,
            self.args.dt_max,
            self.args.t_begin,
            self.args.meth,
            pn=self.args.pn,
            reporter=self.dia,
            viewer=viewer,
            dumpper=io,
            dump_freq=self.args.dump_freq,
        )

        # Initializes restart file counter
        self.nrestart = count(1)
        self.dump_restart_file()

    def dump_restart_file(self):
        with open(self.args.restart_file, 'wb') as fobj:
            pickle.dump(self, fobj, protocol=pickle.HIGHEST_PROTOCOL)

    def evolve(self):
        """

        """
        t_end = self.args.t_end
        while abs(self.integrator.time) < t_end:
            # evolve a single time-step
            self.integrator.evolve_step(t_end)

            # dump restart file
            if next(self.nrestart) % self.args.restart_freq == 0:
                self.dump_restart_file()

        # Finalize the integrator
        self.integrator.finalize(t_end)


# ------------------------------------------------------------------------


def main_simulation(args):
    viewer = GLviewer() if bool(args.view) else None
    mysim = Simulation(args, viewer)
    mysim.evolve()
    return 0


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
    ps.time[...] = mysim.args.t_end
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
        '--dt_max',
        type=float,
        default=0.5,
        help='Maximum time-step size (type: %(type)s, default: %(default)s).'
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
        help=('Speed of light (type: %(type)s), default: %(default)s).')
        )
    simulation.add_argument(
        '-d', '--dump_freq',
        type=int,
        default=1,
        help=('Number of time-steps between dump of snapshots '
              '(type: %(type)s, default: %(default)s).')
        )
    simulation.add_argument(
        '--restart_freq',
        type=int,
        default=4,
        help=('Number of time-steps between rewrites of the restart file '
              '(type: %(type)s, default: %(default)s).')
        )
    simulation.set_defaults(func=main_simulation)
    return simulation


# -- End of File --

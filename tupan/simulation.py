# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import sys
import logging
from itertools import count
from .io.hdf5io import HDF5IO
from .integrator import Integrator
from .utils.timing import Timer


LOGGER = logging.getLogger(__name__)


class Diagnostic(object):
    """

    """
    def __init__(self, ps, cli):
        self.ps = ps
        self.cli = cli
        self.wct = Timer()
        self.wct.start()

        ps.set_phi(ps)
        self.com_r0 = ps.com_r
        self.com_v0 = ps.com_v
        self.ke0 = ps.kinetic_energy
        self.pe0 = ps.potential_energy
        self.te0 = self.ke0 + self.pe0
        self.lmom0 = ps.linear_momentum
        self.amom0 = ps.angular_momentum

        self.time = cli.t_begin
        self.avr_err_sum = 0.0
        self.abs_err_sum = 0.0
        self.counter = count(1)

    def __enter__(self):
        LOGGER.debug(type(self).__name__+'.__enter__')
        self.print_header()
        self.print_diagnostic(self.ps)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        LOGGER.debug(type(self).__name__+'.__exit__')

    def print_header(self):
        fmt = '{0:13s} {1:10s} '\
              '{2:10s} {3:10s} {4:10s} '\
              '{5:15s} {6:10s} {7:10s} '\
              '{8:10s} {9:10s} {10:10s} '\
              '{11:10s} {12:10s} {13:13s}'
        print(fmt.format('#00:time', '#01:dtime',
                         '#02:ke', '#03:pe', '#04:ve',
                         '#05:te', '#06:eerr', '#07:avrerr',
                         '#08:abserr', '#09:com_r', '#10:com_v',
                         '#11:lmom', '#12:amom', '#13:wct'),
              file=sys.stdout)

    def print_diagnostic(self, ps):
        self.ps = ps
        time = ps.global_time
        dtime = time - self.time
        self.time = time

        ps.set_phi(ps)
        com_r = ps.com_r
        com_v = ps.com_v
        ke = ps.kinetic_energy
        pe = ps.potential_energy
        te = ke + pe
        ve = ke + te
        lmom = ps.linear_momentum
        amom = ps.angular_momentum

        eerr = (te - self.te0) / (-pe)
        self.abs_err_sum += abs(eerr)
        self.avr_err_sum += eerr
        counter = next(self.counter)
        abs_err = self.abs_err_sum / counter
        avr_err = self.avr_err_sum / counter
        com_dr = (((com_r - self.com_r0)**2).sum())**0.5
        com_dv = (((com_v - self.com_v0)**2).sum())**0.5
        dlmom = (((lmom - self.lmom0)**2).sum())**0.5
        damom = (((amom - self.amom0)**2).sum())**0.5

        fmt = '{time:< 13.6e} {dtime:< 10.3e} '\
              '{ke:< 10.3e} {pe:< 10.3e} {ve:< 10.3e} '\
              '{te:< 15.8e} {eerr:< 10.3e} {avr_err:< 10.3e} '\
              '{abs_err:< 10.3e} {com_r:< 10.3e} {com_v:< 10.3e} '\
              '{lmom:< 10.3e} {amom:< 10.3e} {wct:< 13.6e}'
        print(fmt.format(time=time, dtime=dtime,
                         ke=ke, pe=pe,
                         ve=ve, te=te,
                         eerr=eerr, avr_err=avr_err,
                         abs_err=abs_err, com_r=com_dr,
                         com_v=com_dv, lmom=dlmom, amom=damom,
                         wct=self.wct.elapsed()),
              file=sys.stdout)


# ------------------------------------------------------------------------


class Run(object):
    """
    Run a new N-body simulation
    """
    def __init__(self, subparser):
        from . import config
        parser = subparser.add_parser(
            'run',
            parents=[config.parser],
            description=self.__doc__,
        )
        parser.add_argument(
            '-e', '--eta',
            type=float,
            required=True,
            help='Time-step parameter (type: %(type)s, required: %(required)s)'
        )
        parser.add_argument(
            '-t', '--t_end',
            type=float,
            required=True,
            help='Simulation end time (type: %(type)s, required: %(required)s)'
        )
        parser.add_argument(
            '-i', '--input_file',
            type=str,
            required=True,
            help=('Initial conditions filename '
                  '(type: %(type)s, required: %(required)s)')
        )
        parser.add_argument(
            '-m', '--method',
            metavar='NAME',
            type=str,
            required=True,
            choices=Integrator.PROVIDED_METHODS,
            help=('Integration method '
                  '(type: %(type)s, required: %(required)s, '
                  'choices: {%(choices)s})')
        )
        parser.add_argument(
            '--dt_max',
            type=float,
            default=0.5,
            help=('Maximum time-step size '
                  '(type: %(type)s, default: %(default)s)')
        )
        parser.add_argument(
            '-o', '--output_file',
            type=str,
            help=('Output filename to store the simulation data '
                  '(type: %(type)s, default: %(default)s)')
        )
        parser.add_argument(
            '--t_begin',
            type=float,
            default=0.0,
            help='Simulation begin time (type: %(type)s, default: %(default)s)'
        )
        parser.add_argument(
            '--pn_order',
            metavar='PN_ORDER',
            type=int,
            default=0,
            choices=[0, 2, 4, 5, 6, 7],
            help=('Order of the Post-Newtonian corrections '
                  '(type: %(type)s, default: %(default)s, '
                  'choices: {%(choices)s})')
        )
        parser.add_argument(
            '--clight',
            type=float,
            help='Speed of light (type: %(type)s), default: %(default)s)'
        )
        parser.add_argument(
            '-d', '--dump_freq',
            metavar='NSTEPS',
            type=int,
            default=1,
            help=('Number of time-steps between simulation data dumps '
                  '(type: %(type)s, default: %(default)s)')
        )
        self.parser = parser

    def __call__(self, cli):
        # pre-process some args
        cli.pn = {}
        if cli.pn_order > 0:
            if not cli.clight:
                self.parser.error(
                    'the --clight argument is required if --pn_order > 0'
                )
            cli.pn = {'order': cli.pn_order, 'clight': cli.clight}

        # read initial conditions
        with HDF5IO(cli.input_file, 'r') as fid:
            ps = fid.load_snap()

        # set initial time for all particles
        for p in ps.members.values():
            if p.n:
                p.time[...] = cli.t_begin

        # main function
        def run(ps, cli, viewer=None):
            dumpper = None
            if cli.output_file:
                with HDF5IO(cli.output_file, 'w') as io:
                    io.dump_snap(ps, tag=0)
                dumpper = HDF5IO(cli.output_file, 'a')

            with Diagnostic(ps, cli) as checker:
                with Integrator(ps, cli,
                                viewer=viewer,
                                dumpper=dumpper,
                                checker=checker) as mysim:
                    mysim.evolve(cli.t_end)

        # call main function!
        if cli.view:
            from .animation import GLviewer
            with GLviewer() as viewer:
                run(ps, cli, viewer=viewer)
        else:
            run(ps, cli)


class Restart(object):
    """
    Restart from a previous simulation
    """
    def __init__(self, subparser):
        from . import config
        parser = subparser.add_parser(
            'restart',
            parents=[config.parser],
            description=self.__doc__,
        )
        parser.add_argument(
            '-e', '--eta',
            type=float,
            help=('Time-step parameter. If None, it defaults to the value in '
                  'restart file (type: %(type)s, default: %(default)s)')
        )
        parser.add_argument(
            '-t', '--t_end',
            type=float,
            required=True,
            help='Simulation end time (type: %(type)s, required: %(required)s)'
        )
        self.parser = parser

    def __call__(self, cli):
        import pickle

        # read restart file
        with open('restart.pkl', 'rb') as fobj:
            simulation = pickle.load(fobj)

        # reset cli arguments
        eta = vars(cli).pop('eta')
        if eta:
            vars(simulation.cli).update({'eta': eta})
        vars(simulation.cli).update(vars(cli))

        # main function
        def restart(simulation, cli):
            wct = simulation.checker.wct    # reset wall-clock-time to
            wct.reset_at(wct.toc)           # the point where it stopped
            with simulation as mysim:
                mysim.evolve(cli.t_end)

        # call main function!
        if cli.view:
            from .animation import GLviewer
            with GLviewer() as viewer:
                simulation.viewer = viewer
                restart(simulation, cli)
        else:
            restart(simulation, cli)


# -- End of File --

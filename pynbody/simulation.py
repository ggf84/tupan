#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TODO.
"""

from __future__ import print_function
import sys
import pickle
import math
from ggf84decor import selftimer
from pynbody.io import HDF5IO
from pynbody.integrator import (METH_NAMES, METHS)


__all__ = ['Simulation']


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
    def __init__(self, fname,
                 e0, rcom0, lmom0, amom0,
                 ceerr=0.0, ceerr_count=0):
        self.fname = fname
        self.e0 = e0
        self.rcom0 = rcom0
        self.lmom0 = lmom0
        self.amom0 = amom0
        self.ceerr = ceerr
        self.ceerr_count = ceerr_count
        self.print_header()


    def __repr__(self):
        return '{0}'.format(self.__dict__)


    def print_header(self):
        fmt = '{0:12s} '\
              '{1:12s} {2:12s} {3:16s} '\
              '{4:12s} {5:11s} {6:11s} '\
              '{7:10s} {8:10s} {9:10s} '\
              '{10:10s} {11:10s} {12:10s} '\
              '{13:10s} {14:10s} {15:10s}'
        myprint(fmt.format('#time:00',
                           '#ekin:01', '#epot:02', '#etot:03',
                           '#evir:04', '#eerr:05', '#geerr:06',
                           '#rcomX:07', '#rcomY:08', '#rcomZ:09',
                           '#lmomX:10', '#lmomY:11', '#lmomZ:12',
                           '#amomX:13', '#amomY:14', '#amomZ:15'),
                self.fname, 'w')


    def print_diagnostic(self, time, particles):
        particles['body'].set_phi(particles['body'])
        e = particles['body'].get_total_energies()
        e1 = e.tot
        rcom1 = particles['body'].get_center_of_mass_pos()
        lmom1 = particles['body'].get_total_linmom()
        amom1 = particles['body'].get_total_angmom()

        eerr = (e1-self.e0)/abs(self.e0)
        self.ceerr += eerr**2
        self.ceerr_count += 1
        geerr = math.sqrt(self.ceerr / self.ceerr_count)
        rcom = (rcom1-self.rcom0)
        lmom = (lmom1-self.lmom0)
        amom = (amom1-self.amom0)

        fmt = '{time:< 12.6g} '\
              '{ekin:< 12.6g} {epot:< 12.6g} {etot:< 16.10g} '\
              '{evir:< 12.6g} {eerr:< 11.5g} {geerr:< 11.5g} '\
              '{rcom[0]:< 10.4g} {rcom[1]:< 10.4g} {rcom[2]:< 10.4g} '\
              '{lmom[0]:< 10.4g} {lmom[1]:< 10.4g} {lmom[2]:< 10.4g} '\
              '{amom[0]:< 10.4g} {amom[1]:< 10.4g} {amom[2]:< 10.4g}'
        myprint(fmt.format(time=time,
                           ekin=e.kin, epot=e.pot, etot=e.tot,
                           evir=e.vir, eerr=eerr, geerr=geerr,
                           rcom=rcom, lmom=lmom, amom=amom),
                self.fname, 'a')



class Simulation(object):
    """
    The Simulation class is the top level class for N-body simulations.
    """
    def __init__(self, args, viewer):
        self.args = args
        self.viewer = viewer

        # Read the initial conditions.
        io = HDF5IO(self.args.input)
        particles = io.read_snapshot()

        # Set the method of integration.
        self.Integrator = METHS[METH_NAMES.index(self.args.meth)]

        # Compute the initial value of all quantities necessary for the run.
        (e0, rcom0, lmom0, amom0) = self._setup_initial_quantities(particles)

        # Initializes the integrator.
        self.integrator = self.Integrator(0.0, self.args.eta, particles)

        # Initializes the diagnostic of the simulation.
        self.dia = Diagnostic(self.args.log_file,
                              e0, rcom0, lmom0, amom0)
        self.dia.print_diagnostic(self.integrator.time, particles)

        # Initializes snapshots output.
        self.iosnaps = HDF5IO('snapshots.hdf5')
        self.iosnaps.snap_number = 0
        self.iosnaps.write_snapshot(particles)

        # Initializes times for output a couple of things.
        self.dt_gl = 1.0 / self.args.gl_freq
        self.dt_dia = 1.0 / self.args.diag_freq
        self.dt_res = 1.0 / self.args.res_freq
        self.oldtime_gl = self.integrator.time
        self.oldtime_dia = self.integrator.time
        self.oldtime_res = self.integrator.time


    def _setup_initial_quantities(self, particles):
        """

        """
        particles['body'].set_phi(particles['body'])
        particles['body'].set_acc(particles['body'])
        particles['body'].stepdens[:,1] = particles['body'].stepdens[:,0].copy()

        e0 = particles['body'].get_total_etot()
        rcom0 = particles['body'].get_center_of_mass_pos()
        lmom0 = particles['body'].get_total_linmom()
        amom0 = particles['body'].get_total_angmom()

        return (e0, rcom0, lmom0, amom0)


    def restart(self):
        self.integrator = self.Integrator(self.integrator.time, self.args.eta,
                                          self.integrator.gather().copy())


    @selftimer
    def evolve(self):
        """

        """
        if self.viewer:
            self.viewer.initialize()

        while (self.integrator.time < self.args.tmax):
            self.integrator.step()
            if (self.integrator.time - self.oldtime_gl >= self.dt_gl):
                self.oldtime_gl += self.dt_gl
                if self.viewer:
                    self.viewer.show_event(self.integrator)
            if (self.integrator.time - self.oldtime_dia >= self.dt_dia):
                self.oldtime_dia += self.dt_dia
                particles = self.integrator.gather()
                self.dia.print_diagnostic(self.integrator.time, particles)
                self.iosnaps.snap_number += 1
                self.iosnaps.write_snapshot(particles)
            if (self.integrator.time - self.oldtime_res >= self.dt_res):
                self.oldtime_res += self.dt_res
                with open('restart.pkl', 'w') as fobj:
                    pickle.dump(self, fobj, protocol=pickle.HIGHEST_PROTOCOL)

        if self.viewer:
            self.viewer.enter_main_loop()


########## end of file ##########

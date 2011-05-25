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
from pynbody.analysis.glviewer import GLviewer
from pynbody.integrator import (METH_NAMES, METHS)


RUN_MODES = ['newrun', 'restart']


class Diagnostic(object):
    """

    """
    def __init__(self, e0, rcom0, lmom0, amom0, fobj, ceerr=0.0, ceerr_count=0):
        self.fobj = fobj
        self.e0 = e0
        self.rcom0 = rcom0
        self.lmom0 = lmom0
        self.amom0 = amom0
        self.ceerr = ceerr
        self.ceerr_count = ceerr_count


    def __repr__(self):
        return '{0}'.format(self.__dict__)


    def print_header(self):
        fmt = '{0:12s} '\
              '{1:12s} {2:12s} {3:16s} '\
              '{4:12s} {5:11s} {6:11s} '\
              '{7:10s} {8:10s} {9:10s} '\
              '{10:10s} {11:10s} {12:10s} '\
              '{13:10s} {14:10s} {15:10s}'
        print(fmt.format('#time:00',
                         '#ekin:01', '#epot:02', '#etot:03',
                         '#evir:04', '#eerr:05', '#geerr:06',
                         '#rcomX:07', '#rcomY:08', '#rcomZ:09',
                         '#lmomX:10', '#lmomY:11', '#lmomZ:12',
                         '#amomX:13', '#amomY:14', '#amomZ:15'),
              file=self.fobj)


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
        print(fmt.format(time=time,
                         ekin=e.kin, epot=e.pot, etot=e.tot,
                         evir=e.vir, eerr=eerr, geerr=geerr,
                         rcom=rcom, lmom=lmom, amom=amom),
              file=self.fobj)



class Simulation(object):
    """
    The Simulation class is the top level class for N-body simulations.
    """
    def __init__(self, args):
        self.args = args
        if self.args.view:
            self.viewer = GLviewer()

        # Set the method of integration.
        if self.args.meth in METH_NAMES:
            self.Integrator = METHS[METH_NAMES.index(self.args.meth)]
        else:
            print('Typo or invalid name of the method of integration.')
            print('Available methods:', METH_NAMES)
            print('exiting...')
            sys.exit(1)



        print('#'*25, file=args.debug_file)
        print(self.args.__dict__, file=args.debug_file)
        print('#'*25, file=args.debug_file)

        self.snapcount = 0

        if self.args.smod == 'restart':
            io = HDF5IO('restart.hdf5')
            particles = io.read_snapshot()
            self.snapcount = 16  # XXX: read snapcount
            e = particles['body'].get_total_energies()
            e0 = e.tot
            rcom0 = particles['body'].get_center_of_mass_pos()
            lmom0 = particles['body'].get_total_linmom()
            amom0 = particles['body'].get_total_angmom()
        else:
            io = HDF5IO(self.args.input)
            particles = io.read_snapshot()
            particles['body'].set_phi(particles['body'])
            e = particles['body'].get_total_energies()
            e0 = e.tot
            rcom0 = particles['body'].get_center_of_mass_pos()
            lmom0 = particles['body'].get_total_linmom()
            amom0 = particles['body'].get_total_angmom()

        self.dia = Diagnostic(e0, rcom0, lmom0, amom0, self.args.log_file)

        if not self.args.smod == 'restart':
            self.dia.print_header()
            self.dia.print_diagnostic(0.0, particles)
            particles['body'].set_acc(particles['body'])
            particles['body'].stepdens[:,1] = particles['body'].stepdens[:,0].copy()
            io = HDF5IO('snapshots.hdf5')
            io.write_snapshot(particles, group_id=self.snapcount)


        self.integrator = self.Integrator(0.0, self.args.eta, particles)


    @selftimer
    def evolve(self):
        """

        """
        diadt = 1.0 / self.args.diag_freq
        gldt = 1.0 / self.args.gl_freq
        io = HDF5IO('snapshots.hdf5')
        iorestart = HDF5IO('restart.hdf5', 'w')

        if self.args.view:
            self.viewer.initialize()

        while (self.integrator.time < self.args.tmax):
            old_restime = self.integrator.time
            while ((self.integrator.time - old_restime < self.args.resdt) and
                   (self.integrator.time < self.args.tmax)):
                self.snapcount += 1
                old_diatime = self.integrator.time
                while ((self.integrator.time - old_diatime < diadt) and
                       (self.integrator.time < self.args.tmax)):
                    old_gltime = self.integrator.time
                    while ((self.integrator.time - old_gltime < gldt) and
                           (self.integrator.time < self.args.tmax)):
                        self.integrator.step()
                    if self.args.view:
                        self.viewer.show_event(self.integrator)
                particles = self.integrator.gather()
                self.dia.print_diagnostic(self.integrator.time, particles)
                io.write_snapshot(particles, group_id=self.snapcount)
            if (self.integrator.time - old_restime >= self.args.resdt):
                iorestart.write_snapshot(particles)

        if self.args.view:
            self.viewer.enter_main_loop()


########## end of file ##########

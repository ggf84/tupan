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
    def __init__(self, fname, fmode,
                 e0, rcom0, lmom0, amom0,
                 ceerr=0.0, ceerr_count=0):
        self.fname = fname
        self.fmode = fmode
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
        myprint(fmt.format('#time:00',
                           '#ekin:01', '#epot:02', '#etot:03',
                           '#evir:04', '#eerr:05', '#geerr:06',
                           '#rcomX:07', '#rcomY:08', '#rcomZ:09',
                           '#lmomX:10', '#lmomY:11', '#lmomZ:12',
                           '#amomX:13', '#amomY:14', '#amomZ:15'),
                self.fname, self.fmode)


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
    def __init__(self, args):
        self.args = args

        print('#'*25, file=sys.stderr)
        print(self.args, file=sys.stderr)
        print('#'*25, file=sys.stderr)




        io = HDF5IO(self.args.input)
        particles = io.read_snapshot()
        particles['body'].set_phi(particles['body'])
        particles['body'].set_acc(particles['body'])
        particles['body'].stepdens[:,1] = particles['body'].stepdens[:,0].copy()

        e = particles['body'].get_total_energies()
        e0 = e.tot
        rcom0 = particles['body'].get_center_of_mass_pos()
        lmom0 = particles['body'].get_total_linmom()
        amom0 = particles['body'].get_total_angmom()

        self.dia = Diagnostic(self.args.log_file, self.args.fmode,
                              e0, rcom0, lmom0, amom0)

        self.dia.print_header()
        self.dia.print_diagnostic(0.0, particles)


        # Set viewer if enabled.
        self.viewer = None
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

        # Initialize the integrator.
        self.integrator = self.Integrator(0.0, self.args.eta, particles)


        self.snapcount = 0
        self.iosnaps = HDF5IO('snapshots.hdf5')
        self.iosnaps.write_snapshot(particles, group_id=self.snapcount)


        self.diadt = 1.0 / self.args.diag_freq
        self.gldt = 1.0 / self.args.gl_freq


        self.old_restime = self.integrator.time
        self.old_diatime = self.integrator.time
        self.old_gltime = self.integrator.time




    @selftimer
    def evolve(self):
        """

        """
        self.integrator = self.Integrator(self.integrator.time, self.args.eta,
                                          self.integrator.gather().copy())

        if self.viewer:
            self.viewer.initialize()

        while (self.integrator.time < self.args.tmax):
            self.integrator.step()
            if (self.integrator.time - self.old_gltime >= self.gldt):
                self.old_gltime += self.gldt
                if self.viewer:
                    self.viewer.show_event(self.integrator)
            if (self.integrator.time - self.old_diatime >= self.diadt):
                self.old_diatime += self.diadt
                self.snapcount += 1
                particles = self.integrator.gather()
                self.dia.print_diagnostic(self.integrator.time, particles)
                self.iosnaps.write_snapshot(particles, group_id=self.snapcount)
            if (self.integrator.time - self.old_restime >= self.args.resdt):
                self.old_restime += self.args.resdt
                with open('restart.pickle', 'w') as fobj:
                    pickle.dump(self, fobj, protocol=pickle.HIGHEST_PROTOCOL)

        if self.viewer:
            self.viewer.enter_main_loop()


########## end of file ##########

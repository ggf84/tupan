# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import h5py
import pickle
import logging
from ..units import ureg
from collections import OrderedDict


LOGGER = logging.getLogger(__name__)


def do_pickle(obj):
    protocol = 0  # ensures backward compatibility with Python 2.x
    return pickle.dumps(type(obj), protocol=protocol)


def dump_ps(parent, ps):
    psys = parent.create_group('PartSystem')
    psys.attrs['npart'] = ps.n
    psys.attrs['class'] = do_pickle(ps)
    for member in ps.members.values():
        ptype = psys.create_group('PartType#' + str(member.part_type))
        ptype.attrs['npart'] = member.n
        ptype.attrs['class'] = do_pickle(member)
        for attr in member.default_attrs:
            array = getattr(member, attr)
            dset = ptype.create_dataset(
                attr,
                data=array.T,
                chunks=True,
                shuffle=True,
                compression='gzip',
                compression_opts=9,
            )
            dset.attrs['unit'] = array.units.format_babel()


def load_ps(parent):
    psys = parent['PartSystem']
    members = {}
    for part_type in psys.values():
        n = part_type.attrs['npart']
        cls = pickle.loads(part_type.attrs['class'])
        member = cls(n)
        for attr, dset in part_type.items():
            array = getattr(member, attr)
            array[...] = dset[...].T * ureg(dset.attrs['unit'])
        members[member.name] = member
    cls = pickle.loads(psys.attrs['class'])
    return cls.from_members(**members)


class HDF5IO(object):
    """

    """
    def __init__(self, fname, fmode='r'):
        if not fname.endswith('.hdf5'):
            fname += '.hdf5'
        self.fname = fname
        self.fmode = fmode

    def __enter__(self):
        msg = "{clsname}('{fname}', '{fmode}').__enter__"
        LOGGER.debug(
            msg.format(
                clsname=type(self).__name__,
                fname=self.fname,
                fmode=self.fmode
            )
        )
        self.open_file = h5py.File(self.fname, self.fmode)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        msg = "{clsname}('{fname}', '{fmode}').__exit__"
        LOGGER.debug(
            msg.format(
                clsname=type(self).__name__,
                fname=self.fname,
                fmode=self.fmode
            )
        )
        self.open_file.close()
        del self.open_file

    @property
    def n_eras(self):
        return len(self.open_file)
    n_snaps = n_eras

    def append_data(self, ps):
        try:
            self.stream += ps
        except:
            self.stream = ps.copy()

    def dump_snap(self, snap, tag=0):
        group = self.open_file.create_group('Era#%d/Snap' % tag)
        dump_ps(group, snap)

    def load_snap(self, tag=0):
        group = self.open_file['Era#%d/Snap' % tag]
        return load_ps(group)

    def dump_stream(self, stream, tag=0):
        group = self.open_file.create_group('Era#%d/Stream' % tag)
        dump_ps(group, stream)

    def load_stream(self, tag=0):
        group = self.open_file['Era#%d/Stream' % tag]
        return load_ps(group)

    def flush_stream(self, tag=0):
        if hasattr(self, 'stream'):
            self.dump_stream(self.stream, tag=tag)
            del self.stream

    def load_era(self, tag=0):
        era = self.load_snap(tag)
        era += self.load_stream(tag)
        era += self.load_snap(tag+1)
        return era

    def load_full_era(self):
        era = self.load_snap(0)
        for tag in range(self.n_eras-1):
            era += self.load_stream(tag)
            era += self.load_snap(tag+1)
        return era

    def era2snaps(self, era, n_snaps):
        """
        convert from worldline to snapshot layout (experimental!)
        """
        import numpy as np
#        from scipy.interpolate import interp1d
        from scipy.interpolate import CubicSpline

        ps = self.load_era(era)
        t_begin = min(np.min(p.time) for p in ps.members.values())
        t_end = max(np.max(p.time) for p in ps.members.values())
        times = np.arange(t_begin, t_end, (t_end - t_begin)/n_snaps)

        snaps = OrderedDict()
        for t in times:
            snaps[t] = type(ps)()

#        # ---
#        for member in ps.members.values():
#            if member.n:
#                pids = set(member.pid)
#                n = len(pids)
#                for t in times:
#                    snap = type(member)(n)
#                    members = snaps[t].members
#                    members[snap.name] = snap
#                    snaps[t].update_members(**members)
#        pids = set(ps.pid)
#        n = len(pids)
#        for i, pid in enumerate(pids):
#            pstream = ps[ps.pid == pid]
#            for attr in ps.default_attrs:
#                array = getattr(pstream, attr)
#                # alen = len(array.T)
#                # kdeg = 3 if alen > 3 else (2 if alen > 2 else 1)
#                # f = interp1d(pstream.time, array, kind=kdeg)
#                f = CubicSpline(pstream.time, array,
#                                axis=-1, bc_type='natural')
#                for t, snap in snaps.items():
#                    ary = getattr(snap, attr)
#                    ary[..., i] = pid if attr == 'pid' else f(t)
#        # ---

        # ---
        for member in ps.members.values():
            if member.n:
                name = member.name
                pids = set(member.pid)
                n = len(pids)
                for t in times:
                    snap = type(member)(n)
                    members = snaps[t].members
                    members[snap.name] = snap
                    snaps[t].update_members(**members)
                for pid in pids:
                    pstream = member[member.pid == pid]
                    for attr in member.default_attrs:
                        array = getattr(pstream, attr)
                        # alen = len(array.T)
                        # kdeg = 3 if alen > 3 else (2 if alen > 2 else 1)
                        # f = interp1d(pstream.time, array, kind=kdeg)
                        f = CubicSpline(pstream.time, array,
                                        axis=-1, bc_type='natural')
                        for t in times:
                            snap = snaps[t]
                            obj = snap.members[name]
                            ary = getattr(obj, attr)
                            ary[..., pid] = pid if attr == 'pid' else f(t)
        # ---

        return snaps

    def compare_wl(self, t_begin, t_end, nsnaps):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.interpolate import interp1d
        with HDF5IO('snap0.hdf5', 'r') as fid:
            ps0 = fid.load_full_era()
        with HDF5IO('snap1.hdf5', 'r') as fid:
            ps1 = fid.load_full_era()
        with HDF5IO('snap2.hdf5', 'r') as fid:
            ps2 = fid.load_full_era()

        index = ps0[ps0.nstep == ps0.nstep.max()].pid[0]
        a0 = ps0[ps0.pid == index]
        a1 = ps1[ps1.pid == index]
        a2 = ps2[ps2.pid == index]

        x = np.linspace(t_begin, t_end, nsnaps+1)
        f = {}
        f[0] = interp1d(a2.time, a2.pos[0], kind=3)
        f[1] = interp1d(a2.time, a2.pos[1], kind=3)
        f[2] = interp1d(a2.time, a2.pos[2], kind=3)

        plt.plot(a0.pos[0], a0.pos[1], '-o', label="PBaSS: l=1")
        plt.plot(a1.pos[0], a1.pos[1], '-o', label="PBaSS: l=4")
        plt.plot(a2.pos[0], a2.pos[1], '-o', label="PBaSS: l=16")
        plt.plot(f[0](x), f[1](x), '-', label="interp. function")
        plt.legend(loc="best", shadow=True,
                   fancybox=True, borderaxespad=0.75)
        plt.show()

        axis = 0
        plt.plot(a0.time, a0.pos[axis], '-o', label="PBaSS: l=1")
        plt.plot(a1.time, a1.pos[axis], '-o', label="PBaSS: l=4")
        plt.plot(a2.time, a2.pos[axis], '-o', label="PBaSS: l=16")
        plt.plot(x, f[axis](x), '-', label="interp. function")
        plt.legend(loc="best", shadow=True,
                   fancybox=True, borderaxespad=0.75)
        plt.show()

        axis = 0
        plt.plot(a0.time, a0.pos[axis] - f[axis](a0.time),
                 '-o', label="PBaSS: l=1")
        plt.plot(a1.time, a1.pos[axis] - f[axis](a1.time),
                 '-o', label="PBaSS: l=4")
        plt.plot(a2.time, a2.pos[axis] - f[axis](a2.time),
                 '-o', label="PBaSS: l=16")
        plt.legend(loc="best", shadow=True,
                   fancybox=True, borderaxespad=0.75)
        plt.show()


# -- End of File --

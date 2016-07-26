# -*- coding: utf-8 -*-
#

"""
TODO.
"""

from __future__ import print_function
import h5py
import pickle
from itertools import count
from collections import OrderedDict
from ..lib.utils.timing import timings, bind_all


PICKLE_PROTOCOL = 0  # ensures backward compatibility with Python 2.x


def do_pickle(obj):
    return pickle.dumps(type(obj), protocol=PICKLE_PROTOCOL)


def dump_ps(parent, ps):
    psys = parent.create_group('PartSys')
    psys.attrs['npart'] = ps.n
    psys.attrs['class'] = do_pickle(ps)
    for member in ps.members.values():
        ptype = psys.create_group('PartType#' + str(member.part_type))
        ptype.attrs['npart'] = member.n
        ptype.attrs['class'] = do_pickle(member)
        for name in member.default_attr_names:
            array = getattr(member, name)
            ptype.create_dataset(name, data=array.T,
                                 chunks=True, shuffle=True,
                                 compression='gzip')


def load_ps(parent):
    psys = parent['PartSys']
    members = {}
    for part_type in psys.values():
        n = part_type.attrs['npart']
        cls = pickle.loads(part_type.attrs['class'])
        member = cls(n)
        for name, dset in part_type.items():
            attr = getattr(member, name)
            attr[...] = dset[...].T
        members[member.name] = member
    cls = pickle.loads(psys.attrs['class'])
    return cls.from_members(**members)


@bind_all(timings)
class HDF5IO(object):
    """

    """
    def __init__(self, fname, fmode='r'):
        if not fname.endswith('.hdf5'):
            fname += '.hdf5'
        self.file = h5py.File(fname, fmode)
        self.snap_id = count()
        self.stream_id = count()
        self.data_stream = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.file.close()

    def dump_snap(self, ps, tag=0):
        snap = self.file.create_group('Snap#%d' % tag)
        dump_ps(snap, ps)

    def load_snap(self, tag=0):
        snap = self.file['Snap#%d' % tag]
        return load_ps(snap)

    def dump_stream(self, ps, tag=0):
        stream = self.file.create_group('Snap#%d/Stream' % tag)
        dump_ps(stream, ps)

    def load_stream(self, tag=0):
        stream = self.file['Snap#%d/Stream' % tag]
        return load_ps(stream)

    def append_data(self, ps):
        self.data_stream.append(ps.copy())

    def flush_data_stream(self):
        if self.data_stream:
            data = self.data_stream.pop(0)
            for d in self.data_stream:
                data.append(d)
            self.dump_stream(data, tag=next(self.stream_id))
            del self.data_stream[:]

    def init_new_era(self, ps):
        self.flush_data_stream()
        self.dump_snap(ps, tag=next(self.snap_id))
        self.file.flush()

    def load_era(self, tag=0):
        era = self.load_snap(tag)
        era.append(self.load_stream(tag))
        era.append(self.load_snap(tag+1))
        return era

    def load_full_era(self):
        era = self.load_snap(0)
        for i in range(1, len(self.file)):
            era.append(self.load_stream(i-1))
            era.append(self.load_snap(i))
        return era

    def era2snaps(self, era, n_snaps):
        """
        convert from worldline to snapshot layout (experimental!)
        """
        import numpy as np
        from scipy.interpolate import interp1d

        ps = self.load_era(era)
        t_begin, t_end = np.min(ps.time), np.max(ps.time)
        times = np.arange(t_begin, t_end, (t_end - t_begin)/n_snaps)

        snaps = OrderedDict()
        for t in times:
            snaps[t] = type(ps)()

        # ---
        for member in ps.members.values():
            if member.n:
                pids = set(member.pid)
                n = len(pids)
                for t in times:
                    snap = type(member)(n)
                    members = snaps[t].members
                    members[snap.name] = snap
                    snaps[t].update_members(**members)
        pids = set(ps.pid)
        n = len(pids)
        for i, pid in enumerate(pids):
            pstream = ps[ps.pid == pid]
            for attr in ps.default_attr_names:
                array = getattr(pstream, attr)
                alen = len(array.T)
                kdeg = 3 if alen > 3 else (2 if alen > 2 else 1)
                f = interp1d(pstream.time, array, kind=kdeg)
                for t, snap in snaps.items():
                    ary = getattr(snap, attr)
                    ary[..., i] = pid if attr == 'pid' else f(t)
        # ---

#        # ---
#        for member in ps.members.values():
#            if member.n:
#                name = member.name
#                pids = set(member.pid)
#                n = len(pids)
#                for t in times:
#                    snap = type(member)(n)
#                    members = snaps[t].members
#                    members[snap.name] = snap
#                    snaps[t].update_members(**members)
#                for i, pid in enumerate(pids):
#                    pstream = member[member.pid == pid]
#                    for attr in member.default_attr_names:
#                        array = getattr(pstream, attr)
#                        alen = len(array.T)
#                        kdeg = 3 if alen > 3 else (2 if alen > 2 else 1)
#                        f = interp1d(pstream.time, array, kind=kdeg)
#                        for t in times:
#                            snap = snaps[t]
#                            obj = getattr(snap.members, name)
#                            ary = getattr(obj, attr)
#                            ary[..., i] = pid if attr == 'pid' else f(t)
#        # ---

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
        f[0] = interp1d(a2.time, a2.rdot[0][0], kind=3)
        f[1] = interp1d(a2.time, a2.rdot[0][1], kind=3)
        f[2] = interp1d(a2.time, a2.rdot[0][2], kind=3)

        plt.plot(a0.rdot[0][0], a0.rdot[0][1], '-o', label="PBaSS: l=1")
        plt.plot(a1.rdot[0][0], a1.rdot[0][1], '-o', label="PBaSS: l=4")
        plt.plot(a2.rdot[0][0], a2.rdot[0][1], '-o', label="PBaSS: l=16")
        plt.plot(f[0](x), f[1](x), '-', label="interp. function")
        plt.legend(loc="best", shadow=True,
                   fancybox=True, borderaxespad=0.75)
        plt.show()

        axis = 0
        plt.plot(a0.time, a0.rdot[0][axis], '-o', label="PBaSS: l=1")
        plt.plot(a1.time, a1.rdot[0][axis], '-o', label="PBaSS: l=4")
        plt.plot(a2.time, a2.rdot[0][axis], '-o', label="PBaSS: l=16")
        plt.plot(x, f[axis](x), '-', label="interp. function")
        plt.legend(loc="best", shadow=True,
                   fancybox=True, borderaxespad=0.75)
        plt.show()

        axis = 0
        plt.plot(a0.time, a0.rdot[0][axis] - f[axis](a0.time),
                 '-o', label="PBaSS: l=1")
        plt.plot(a1.time, a1.rdot[0][axis] - f[axis](a1.time),
                 '-o', label="PBaSS: l=4")
        plt.plot(a2.time, a2.rdot[0][axis] - f[axis](a2.time),
                 '-o', label="PBaSS: l=16")
        plt.legend(loc="best", shadow=True,
                   fancybox=True, borderaxespad=0.75)
        plt.show()


# -- End of File --

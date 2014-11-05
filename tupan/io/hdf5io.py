# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
import sys
import h5py
import pickle
from ..lib.utils.timing import timings, bind_all


__all__ = ['HDF5IO']

IS_PY3K = True if sys.version_info.major > 2 else False
PICKLE_PROTOCOL = 0  # ensures backward compatibility with Python 2.x


@bind_all(timings)
class HDF5IO(object):
    """

    """
    def __init__(self, fname, fmode='r'):
        if not fname.endswith('.hdf5'):
            fname += '.hdf5'
        self.file = h5py.File(fname, fmode)
        self.wl_id = None
        self.wl = None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()

    def write_ic(self, ps):
        """

        """
        cls = pickle.dumps(type(ps), protocol=PICKLE_PROTOCOL)
        cls = cls.decode('utf-8') if IS_PY3K else cls
        ps_group = self.file.require_group(ps.name)
        ps_group.attrs['CLASS'] = cls
        for member in ps.members:
            if member.n:
                cls = pickle.dumps(type(member), protocol=PICKLE_PROTOCOL)
                cls = cls.decode('utf-8') if IS_PY3K else cls
                member_group = ps_group.require_group(member.name)
                member_group.attrs['CLASS'] = cls
                member_group.attrs['N'] = member.n
                attr_group = member_group.require_group('attributes')
                for name in member.dtype.names:
                    ary = getattr(member, name)
                    dset = attr_group.require_dataset(
                        name,
                        shape=ary.T.shape,
                        dtype=ary.dtype,
                        chunks=True,
                        shuffle=True,
                        compression='gzip',
                        )
                    dset[...] = ary.T

    def read_ic(self):
        """

        """
        ps_group = list(self.file.values())[0]
        ps_cls = pickle.loads(ps_group.attrs['CLASS'])
        members = []
        for member_group in ps_group.values():
            n = member_group.attrs['N']
            member_cls = pickle.loads(member_group.attrs['CLASS'])
            member = member_cls(n)
            for name, dset in member_group['attributes'].items():
                attr = getattr(member, name)
                attr[...] = dset[...].T
            members.append(member)
        return ps_cls.from_members(members)

    def write_snapshot(self, ps, snap_id):
        """

        """
        snap_name = 'Snapshot_' + str(snap_id).zfill(6)
        snap_group = self.file.require_group(snap_name)
        cls = pickle.dumps(type(ps), protocol=PICKLE_PROTOCOL)
        cls = cls.decode('utf-8') if IS_PY3K else cls
        ps_group = snap_group.require_group(ps.name)
        ps_group.attrs['CLASS'] = cls
        for member in ps.members:
            if member.n:
                cls = pickle.dumps(type(member), protocol=PICKLE_PROTOCOL)
                cls = cls.decode('utf-8') if IS_PY3K else cls
                member_group = ps_group.require_group(member.name)
                member_group.attrs['CLASS'] = cls
                member_group.attrs['N'] = member.n
                attr_group = member_group.require_group('attributes')
                for name in member.dtype.names:
                    ary = getattr(member, name)
                    dset = attr_group.require_dataset(
                        name,
                        shape=ary.T.shape,
                        dtype=ary.dtype,
                        chunks=True,
                        shuffle=True,
                        compression='gzip',
                        )
                    dset[...] = ary.T

    def read_snapshot(self, snap_id):
        """

        """
        snap_name = 'Snapshot_' + str(snap_id).zfill(6)
        snap_group = self.file[snap_name]
        ps_group = list(snap_group.values())[0]
        ps_cls = pickle.loads(ps_group.attrs['CLASS'])
        members = []
        for member_group in ps_group.values():
            n = member_group.attrs['N']
            member_cls = pickle.loads(member_group.attrs['CLASS'])
            member = member_cls(n)
            for name, dset in member_group['attributes'].items():
                attr = getattr(member, name)
                attr[...] = dset[...].T
            members.append(member)
        return ps_cls.from_members(members)

    def init_worldline(self, ps):
        """

        """
        self.wl = ps.copy()
        self.wl_id = 0

    def flush_worldline(self):
        self.write_worldline(self.wl)
        self.flush()
        self.wl = type(self.wl)()
        self.wl_id += 1

    def write_worldline(self, ps):
        """

        """
        wl_name = 'Worldline_' + str(self.wl_id).zfill(6)
        wl_group = self.file.require_group(wl_name)
        cls = pickle.dumps(type(ps), protocol=PICKLE_PROTOCOL)
        cls = cls.decode('utf-8') if IS_PY3K else cls
        ps_group = wl_group.require_group(ps.name)
        ps_group.attrs['CLASS'] = cls
        for member in ps.members:
            if member.n:
                cls = pickle.dumps(type(member), protocol=PICKLE_PROTOCOL)
                cls = cls.decode('utf-8') if IS_PY3K else cls
                member_group = ps_group.require_group(member.name)
                member_group.attrs['CLASS'] = cls
                member_group.attrs['N'] = member.n
                attr_group = member_group.require_group('attributes')
                for name in member.dtype.names:
                    ary = getattr(member, name)
                    dset = attr_group.require_dataset(
                        name,
                        shape=ary.T.shape,
                        dtype=ary.dtype,
                        chunks=True,
                        shuffle=True,
                        compression='gzip',
                        )
                    dset[...] = ary.T

    def read_worldline(self):
        """

        """
        wl = None
        for wl_group in self.file.values():
            ps_group = list(wl_group.values())[0]
            ps_cls = pickle.loads(ps_group.attrs['CLASS'])
            members = []
            for member_group in ps_group.values():
                n = member_group.attrs['N']
                member_cls = pickle.loads(member_group.attrs['CLASS'])
                member = member_cls(n)
                for name, dset in member_group['attributes'].items():
                    attr = getattr(member, name)
                    attr[...] = dset[...].T
                members.append(member)
            ps = ps_cls.from_members(members)
            if wl is None:
                wl = ps
            else:
                wl.append(ps)
        return wl

    def worldline_to_snapshots(self, t_begin, t_end, nsnaps):
        import time
        import numpy as np
        from scipy.interpolate import InterpolatedUnivariateSpline as spline
        from tupan.analysis.glviewer import GLviewer
        viewer = GLviewer()

        ps = self.read_worldline()

        times = np.linspace(t_begin, t_end, nsnaps+1)

        snaps = {}
        for t in times:
            snaps[t] = type(ps)()

#        # ---
#        start = time.time()
#        for member in ps.members:
#            if member.n:
#                pids = set(member.pid)
#                n = len(pids)
#                for t in times:
#                    snap = type(member)(n)
#                    snaps[t].update_members([snap] + snaps[t].members)
#        pids = set(ps.pid)
#        n = len(pids)
#        for i, pid in enumerate(pids):
#            pstream = ps[ps.pid == pid]
#            for attr in ps.dtype.names:
#                array = getattr(pstream, attr)
#                if array.ndim > 1:
#                    for axis in range(array.shape[0]):
#                        f = spline(pstream.time, array[axis, ...], k=3)
#                        for t, snap in snaps.items():
#                            ary = getattr(snap, attr)
#                            ary[axis, i] = f(t)
#                else:
#                    f = spline(pstream.time, array[...], k=3)
#                    for t, snap in snaps.items():
#                        ary = getattr(snap, attr)
#                        ary[i] = pid if attr == 'pid' else f(t)
#        # ---

        # ---
        start = time.time()
        for member in ps.members:
            if member.n:
                name = member.name
                pids = set(member.pid)
                n = len(pids)
                for t in times:
                    snap = type(member)(n)
                    snaps[t].update_members(snaps[t].members + [snap])
                for i, pid in enumerate(pids):
                    pstream = member[member.pid == pid]
                    for attr in member.dtype.names:
                        array = getattr(pstream, attr)
                        if array.ndim > 1:
                            for axis in range(array.shape[0]):
                                f = spline(pstream.time, array[axis, ...], k=3)
                                for t in times:
                                    snap = snaps[t]
                                    obj = getattr(snap.members, name)
                                    ary = getattr(obj, attr)
                                    ary[axis, i] = f(t)
                        else:
                            f = spline(pstream.time, array[...], k=3)
                            for t in times:
                                snap = snaps[t]
                                obj = getattr(snap.members, name)
                                ary = getattr(obj, attr)
                                ary[i] = pid if attr == 'pid' else f(t)
        # ---

        print('elapsed:', time.time() - start)

        with HDF5IO('snapshots.hdf5', 'w') as fid:
            for i, t in enumerate(times):
                fid.write_snapshot(snaps[t], snap_id=i)

        with HDF5IO('snapshots.hdf5', 'r') as fid:
            for i, t in enumerate(times):
                snap = fid.read_snapshot(snap_id=i)
                viewer.show_event(snap)
        viewer.enter_main_loop()

    def compare_wl(self, t_begin, t_end, nsnaps):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.interpolate import InterpolatedUnivariateSpline as spline
        with HDF5IO('snap0.hdf5', 'r') as fid:
            ps0 = fid.read_worldline()
        with HDF5IO('snap1.hdf5', 'r') as fid:
            ps1 = fid.read_worldline()
        with HDF5IO('snap2.hdf5', 'r') as fid:
            ps2 = fid.read_worldline()

        index = ps0[ps0.nstep == ps0.nstep.max()].pid[0]
        a0 = ps0[ps0.pid == index]
        a1 = ps1[ps1.pid == index]
        a2 = ps2[ps2.pid == index]

        x = np.linspace(t_begin, t_end, nsnaps+1)
        f = {}
        f[0] = spline(a2.time, a2.pos[0, ...], k=3)
        f[1] = spline(a2.time, a2.pos[1, ...], k=3)
        f[2] = spline(a2.time, a2.pos[2, ...], k=3)

        plt.plot(a0.pos[0, ...], a0.pos[1, ...], '-o', label="PBaSS: l=1")
        plt.plot(a1.pos[0, ...], a1.pos[1, ...], '-o', label="PBaSS: l=4")
        plt.plot(a2.pos[0, ...], a2.pos[1, ...], '-o', label="PBaSS: l=16")
        plt.plot(f[0](x), f[1](x), '-', label="interp. function")
        plt.legend(loc="best", shadow=True,
                   fancybox=True, borderaxespad=0.75)
        plt.show()

        axis = 0
        plt.plot(a0.time, a0.pos[axis, ...], '-o', label="PBaSS: l=1")
        plt.plot(a1.time, a1.pos[axis, ...], '-o', label="PBaSS: l=4")
        plt.plot(a2.time, a2.pos[axis, ...], '-o', label="PBaSS: l=16")
        plt.plot(x, f[axis](x), '-', label="interp. function")
        plt.legend(loc="best", shadow=True,
                   fancybox=True, borderaxespad=0.75)
        plt.show()

        axis = 0
        plt.plot(a0.time, a0.pos[axis, ...] - f[axis](a0.time), '-o', label="PBaSS: l=1")
        plt.plot(a1.time, a1.pos[axis, ...] - f[axis](a1.time), '-o', label="PBaSS: l=4")
        plt.plot(a2.time, a2.pos[axis, ...] - f[axis](a2.time), '-o', label="PBaSS: l=16")
        plt.legend(loc="best", shadow=True,
                   fancybox=True, borderaxespad=0.75)
        plt.show()


# -- End of File --

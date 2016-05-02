# -*- coding: utf-8 -*-
#


"""
This script converts a simulation data stream into a serie of snapshots.
"""


from __future__ import print_function
import time
from tupan.io import HDF5IO


# number of snapshots per era.
#
n_snaps = 4


with HDF5IO('datastream.hdf5', 'r') as fin:
    n_eras = len(fin.file)
    msg = '# taking {0} snapshots per era from a total of {1} eras.'
    print(msg.format(n_snaps, n_eras-1))
    with HDF5IO('snapshots.hdf5', 'w') as fout:
        start = time.time()

        for era in range(0, n_eras-1):
            print('# processing era', era)
            snaps = fin.era2snaps(era, n_snaps)
            for tag, snap in enumerate(snaps.values(), start=era*n_snaps):
                fout.dump_snap(snap, tag=tag)

        era += 1
        print('# processing era', era)    # just copy the last snapshot
        snap = fin.load_snap(era)
        fout.dump_snap(snap, tag=era*n_snaps)

        elapsed = time.time() - start
        print('# elapsed time (s):', elapsed)
        print('# average per era (s):', elapsed/era)


# -- End of File --

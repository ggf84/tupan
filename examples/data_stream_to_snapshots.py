# -*- coding: utf-8 -*-
#


"""
This script converts a simulation data stream into a serie of snapshots.
"""


from tupan.io.hdf5io import HDF5IO
from tupan.utils.timing import Timer


# number of snapshots per era.
#
snaps_per_era = 10


with HDF5IO('datastream.hdf5', 'r') as fin:
    n_eras = fin.n_eras
    msg = '# taking {0} snapshots per era from a total of {1} eras.'
    print(msg.format(snaps_per_era, n_eras-1))
    with HDF5IO('snapshots.hdf5', 'w') as fout:
        timer = Timer()
        timer.start()

        for era in range(n_eras-1):
            print('# processing era', era)
            snaps = fin.era2snaps(era, snaps_per_era)
            start = era * snaps_per_era
            for tag, snap in enumerate(snaps.values(), start=start):
                fout.dump_snap(snap, tag=tag)

        # just copy the last snapshot
        era += 1
        snap = fin.load_snap(era)
        fout.dump_snap(snap, tag=era*snaps_per_era)

        elapsed = timer.elapsed()
        print('# elapsed time (s):', elapsed)
        print('# average per era (s):', elapsed/era)


# -- End of File --

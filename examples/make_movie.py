# -*- coding: utf-8 -*-
#


"""
This script converts a serie of snapshots into an OpenGL animation.
To record the animation into a 'movie.mp4' file, please add the option
'--record' to the command line.
"""


from tupan.io.hdf5io import HDF5IO
from tupan.animation import GLviewer


with GLviewer() as viewer:
    with HDF5IO('snapshots.hdf5', 'r') as fid:
        for i in range(len(fid.file)):
            snap = fid.load_snap(tag=i)
            viewer.show_event(snap)


# -- End of File --

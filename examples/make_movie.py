# -*- coding: utf-8 -*-
#


"""
This script converts a serie of snapshots into an OpenGL animation.
To record the animation into a 'movie.mp4' file, please add the option
'--record' to the command line.
"""


from tupan.io import HDF5IO
from tupan.animation import GLviewer


viewer = GLviewer()
with HDF5IO('snapshots.hdf5', 'r') as fid:
    for i in range(len(fid.file)):
        snap = fid.load_snap(tag=i)
        viewer.show_event(snap)
    viewer.enter_main_loop()


# -- End of File --

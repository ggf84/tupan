# -*- coding: utf-8 -*-
#

"""
A Python Toolkit for Astrophysical N-Body Simulations.
"""


from __future__ import print_function
import os
import sys
import logging
import getpass
from .config import cfg

print("# PID: {0}".format(os.getpid()), file=sys.stderr)

CACHE_DIR = os.path.join(
                os.path.abspath(
                    os.path.expanduser(
                        cfg['cache']['prefix']
                    )
                ),
                cfg['cache']['base'] + "-uid{0}-py{1}".format(
                    getpass.getuser(),
                    ".".join(str(i) for i in sys.version_info)
                )
            )
try:
    os.makedirs(CACHE_DIR)
except OSError:
    pass

# set up logging to file
LOG_FILENAME = os.path.join(CACHE_DIR, "tupan-{0}.log".format(os.getpid()))
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG,
                    format=LOG_FORMAT,
                    filename=LOG_FILENAME,
                    filemode="w",
                    )

## set up logging to console
#console = logging.StreamHandler()
#console.setLevel(logging.INFO)
#
## set a simpler format for console use
#formatter = logging.Formatter("# %(name)s - %(levelname)s - %(message)s")
#console.setFormatter(formatter)
#
## add the handler to the root logger
#logging.getLogger("").addHandler(console)


########## end of file ##########

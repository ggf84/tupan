# -*- coding: utf-8 -*-
#

"""
A Python Toolkit for Astrophysical N-Body Simulations.
"""


import os
import logging
import tempfile

# set up logging to file
LOG_FILENAME = os.path.join(tempfile.gettempdir(), "tupan.log")
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

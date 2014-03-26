# -*- coding: utf-8 -*-
#


import os
import sys
import getpass
from collections import defaultdict
try:
    import ConfigParser as configparser     # Py2
except:
    import configparser                     # Py3


PATH = os.path.dirname(__file__)
FILENAME = os.path.join(PATH, 'tupan.cfg')

config = configparser.ConfigParser()
config.read(FILENAME)


cfg = defaultdict(dict)
for section in config.sections():
    for option in config.options(section):
        cfg[section][option] = config.get(section, option)


CACHE_DIR = os.path.abspath(
                os.path.join(
                    PATH,
                    os.path.expanduser(
                        cfg['cache']['prefix']
                    ),
                    cfg['cache']['base'] + "-uid{0}-py{1}".format(
                        getpass.getuser(),
                        ".".join(str(i) for i in sys.version_info)
                    )
                )
            )
try:
    os.makedirs(CACHE_DIR)
except OSError:
    pass


########## end of file ##########

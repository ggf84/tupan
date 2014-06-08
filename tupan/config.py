# -*- coding: utf-8 -*-
#


import os
import sys
import getpass
from collections import defaultdict
try:
    import ConfigParser as configparser     # Py2
except ImportError:
    import configparser                     # Py3


PATH = os.path.dirname(__file__)
FILENAME = os.path.join(PATH, 'tupan.cfg')

CONFIG = configparser.ConfigParser()
CONFIG.read(FILENAME)


CFG = defaultdict(dict)
for section in CONFIG.sections():
    for option in CONFIG.options(section):
        CFG[section][option] = CONFIG.get(section, option)


CACHE_DIR = \
    os.path.abspath(
        os.path.join(
            PATH,
            os.path.expanduser(
                CFG['cache']['prefix']
            ),
            CFG['cache']['base'] + '-uid{0}-py{1}'.format(
                getpass.getuser(),
                '.'.join(str(i) for i in sys.version_info)
            )
        )
    )
try:
    os.makedirs(CACHE_DIR)
except OSError:
    pass


# -- End of File --

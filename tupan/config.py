# -*- coding: utf-8 -*-
#


import os
from collections import defaultdict
try:
    import configparser                     # Py3
except:
    import ConfigParser as configparser     # Py2


PATH = os.path.dirname(__file__)
FILENAME = os.path.join(PATH, 'tupan.cfg')

config = configparser.ConfigParser()
config.read(FILENAME)


cfg = defaultdict(dict)
for section in config.sections():
    for option in config.options(section):
        cfg[section][option] = config.get(section, option)


########## end of file ##########

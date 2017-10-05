# -*- coding: utf-8 -*-
#

"""
Units support.
"""

from pint import UnitRegistry


ureg = UnitRegistry()

ureg.define('G = 6.67408e-11 m^3 kg^-1 s^-2')
ureg.define('speed_of_light = 2.99792458e8 m s^-1 = c')
ureg.define('astronomical_unit = 1.49597870700e11 m = au')

ureg.define('R_sun = 6.957e8 m')
ureg.define('L_sun = 3.828e26 W')
ureg.define('GM_sun = 1.3271244e20 m^3 s^-2')
ureg.define('M_sun = GM_sun / G')

ureg.define('year = 365.25 * day = yr')
ureg.define('parsec = (648000.0 / pi) * au = pc')
ureg.define('light_year = speed_of_light * year = ly')


#ureg.define('uM = 1 * M_sun')
#ureg.define('uL = 1 * pc')
#ureg.define('uE = 0.25 * G * uM**2 / uL')
#ureg.define('uT = G * (uM**2.5) / (4 * uE)**1.5')

#ureg.define('uM = 1 * M_sun')
#ureg.define('uT = 1 * Myr')
#ureg.define('uL = (G * uM * uT**2)**(1/3)')
#ureg.define('uE = 0.25 * G * uM**2 / uL')

ureg.define('uL = 4 * pc')
ureg.define('uT = 1 * Myr')
ureg.define('uM = uL**3 / (G * uT**2)')
ureg.define('uE = 0.25 * uL**5 / (G * uT**4)')
#ureg.define('uE = 0.25 * uM * uL**2 / uT**2')
#ureg.define('uE = 0.25 * G * uM**2 / uL')


if __name__ == '__main__':
    print('G:', (1.0 * ureg.G).to('uL**3 / (uM * uT**2)'))
    print('M:', (1.0 * ureg.uM).to('uM'))
    print('L:', (1.0 * ureg.uL).to('uL'))
    print('T:', (1.0 * ureg.uT).to('uT'))
    print('E:', (1.0 * ureg.uE).to('uM * uL**2 / uT**2'))
    print('c:', (1.0 * ureg.c).to('uL / uT'))
    print()
    print('G:', (1.0 * ureg.G).to('pc**3 / (M_sun * Myr**2)'))
    print('M:', (1.0 * ureg.uM).to('M_sun'))
    print('L:', (1.0 * ureg.uL).to('pc'))
    print('T:', (1.0 * ureg.uT).to('Myr'))
    print('E:', (1.0 * ureg.uE).to('M_sun * pc**2 / Myr**2'))
    print('c:', (1.0 * ureg.c).to('pc / Myr'))
    print()
    print('G:', (1.0 * ureg.G).to_base_units())
    print('M:', (1.0 * ureg.uM).to_base_units())
    print('L:', (1.0 * ureg.uL).to_base_units())
    print('T:', (1.0 * ureg.uT).to_base_units())
    print('E:', (1.0 * ureg.uE).to_base_units())
    print('c:', (1.0 * ureg.c).to_base_units())


# -- End of File --

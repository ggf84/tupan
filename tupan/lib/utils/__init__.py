# -*- coding: utf-8 -*-
#

"""
TODO.
"""


def typed_property(name, expected_type,
                   doc=None, can_get=True,
                   can_set=True, can_del=False):
    storage_name = '_' + name

    def get_value(self):
        import numpy as np
        if hasattr(self, 'members'):
            arrays = [getattr(member, name)
                      for member in self.members.values()]
            return np.concatenate(arrays)
        return expected_type(self)

    def fget(self):
        try:
            return getattr(self, storage_name)
        except AttributeError:
            value = get_value(self)
            setattr(self, name, value)
            return value

    def fset(self, value):
        if hasattr(self, 'members'):
            if not hasattr(self, storage_name):
                ns = 0
                nf = 0
                for member in self.members.values():
                    nf += member.n
                    setattr(member, name, value[ns:nf])
                    ns += member.n
        setattr(self, storage_name, value)

    def fdel(self):
        if hasattr(self, storage_name):
            delattr(self, storage_name)
        if hasattr(self, 'members'):
            for member in self.members.values():
                delattr(member, name)

    fget.__name__ = name
    fset.__name__ = name
    fdel.__name__ = name
    return property(fget if can_get else None,
                    fset if can_set else None,
                    fdel if can_del else None,
                    doc)


# -- End of File --

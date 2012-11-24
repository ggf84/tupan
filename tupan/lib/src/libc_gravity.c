#include"Python.h"
#include"numpy/arrayobject.h"

#include"common.h"
#include"p2p_phi_kernel.h"
#include"p2p_acc_kernel.h"
#include"p2p_acc_jerk_kernel.h"
#include"p2p_tstep_kernel.h"
#include"p2p_pnacc_kernel.h"
#include"bios_kernel.h"

//
// Python methods' interface
////////////////////////////////////////////////////////////////////////////////
static PyObject *
p2p_phi_kernel(PyObject *_self, PyObject *_args)
{
    return _p2p_phi_kernel(_args);
}


static PyObject *
p2p_acc_kernel(PyObject *_self, PyObject *_args)
{
    return _p2p_acc_kernel(_args);
}


static PyObject *
p2p_acc_jerk_kernel(PyObject *_self, PyObject *_args)
{
    return _p2p_acc_jerk_kernel(_args);
}


static PyObject *
p2p_tstep_kernel(PyObject *_self, PyObject *_args)
{
    return _p2p_tstep_kernel(_args);
}


static PyObject *
p2p_pnacc_kernel(PyObject *_self, PyObject *_args)
{
    return _p2p_pnacc_kernel(_args);
}


static PyObject *
bios_kernel(PyObject *_self, PyObject *_args)
{
    return _bios_kernel(_args);
}


static PyMethodDef libc_gravity_meths[] = {
    {"p2p_phi_kernel", (PyCFunction)p2p_phi_kernel, METH_VARARGS,
                "returns the Newtonian gravitational potential."},
    {"p2p_acc_kernel", (PyCFunction)p2p_acc_kernel, METH_VARARGS,
                "returns the Newtonian gravitational acceleration."},
    {"p2p_acc_jerk_kernel", (PyCFunction)p2p_acc_jerk_kernel, METH_VARARGS,
                "returns the Newtonian gravitational acceleration and jerk."},
    {"p2p_tstep_kernel", (PyCFunction)p2p_tstep_kernel, METH_VARARGS,
                "returns the next time-step due to gravitational interaction."},
    {"p2p_pnacc_kernel", (PyCFunction)p2p_pnacc_kernel, METH_VARARGS,
                "returns the post-Newtonian gravitational acceleration."},
    {"bios_kernel", (PyCFunction)bios_kernel, METH_VARARGS,
                "returns updated positions and velocities after application"
                " of the BIOS integrator."},
    {NULL, NULL, 0, NULL},
};


//
// Module initialization
// (http://docs.python.org/3/howto/cporting.html#module-initialization-and-state)
////////////////////////////////////////////////////////////////////////////////

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef32 = {
        PyModuleDef_HEAD_INIT,
        "libc32_gravity",
        "An extension module for Tupan.",
        -1,
        libc_gravity_meths,
        NULL,
        NULL,
        NULL,
        NULL
};

PyObject *
PyInit_libc32_gravity(void)
{
    PyObject *module = PyModule_Create(&moduledef32);

    import_array();

    if (module == NULL)
        return NULL;

    return module;
}

#else
PyMODINIT_FUNC
initlibc32_gravity(void)
{
    PyObject *module = Py_InitModule3("libc32_gravity", libc_gravity_meths,
                                      "An extension module for Tupan.");
    import_array();

    if (module == NULL)
        return;
}
#endif


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef64 = {
        PyModuleDef_HEAD_INIT,
        "libc64_gravity",
        "An extension module for Tupan.",
        -1,
        libc_gravity_meths,
        NULL,
        NULL,
        NULL,
        NULL
};

PyObject *
PyInit_libc64_gravity(void)
{
    PyObject *module = PyModule_Create(&moduledef64);

    import_array();

    if (module == NULL)
        return NULL;

    return module;
}

#else
PyMODINIT_FUNC
initlibc64_gravity(void)
{
    PyObject *module = Py_InitModule3("libc64_gravity", libc_gravity_meths,
                                      "An extension module for Tupan.");
    import_array();

    if (module == NULL)
        return;
}
#endif


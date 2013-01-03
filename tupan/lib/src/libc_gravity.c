#include"Python.h"
#include"numpy/arrayobject.h"

#include"common.h"
#include"p2p_phi_kernel.h"
#include"p2p_acc_kernel.h"
#include"p2p_acc_jerk_kernel.h"
#include"p2p_tstep_kernel.h"
#include"p2p_pnacc_kernel.h"
#include"bios_kernel.h"
#include"nreg_kernels.h"

//
// Python Method Defs
////////////////////////////////////////////////////////////////////////////////
static PyMethodDef libTupan_meths[] = {
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
    {"nreg_Xkernel", (PyCFunction)nreg_Xkernel, METH_VARARGS,
                "nreg_Xkernel for NREG integrator."},
    {"nreg_Vkernel", (PyCFunction)nreg_Vkernel, METH_VARARGS,
                "nreg_Vkernel for NREG integrator."},
    {NULL, NULL, 0, NULL},
};


//
// Module initialization
// (http://docs.python.org/3/howto/cporting.html#module-initialization-and-state)
////////////////////////////////////////////////////////////////////////////////

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef32 = {
        PyModuleDef_HEAD_INIT,
        "libTupanSP",
        "An extension module for Tupan.",
        -1,
        libTupan_meths,
        NULL,
        NULL,
        NULL,
        NULL
};

PyObject *
PyInit_libTupanSP(void)
{
    PyObject *module = PyModule_Create(&moduledef32);

    import_array();

    if (module == NULL)
        return NULL;

    return module;
}

#else
PyMODINIT_FUNC
initlibTupanSP(void)
{
    PyObject *module = Py_InitModule3("libTupanSP", libTupan_meths,
                                      "An extension module for Tupan.");
    import_array();

    if (module == NULL)
        return;
}
#endif


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef64 = {
        PyModuleDef_HEAD_INIT,
        "libTupanDP",
        "An extension module for Tupan.",
        -1,
        libTupan_meths,
        NULL,
        NULL,
        NULL,
        NULL
};

PyObject *
PyInit_libTupanDP(void)
{
    PyObject *module = PyModule_Create(&moduledef64);

    import_array();

    if (module == NULL)
        return NULL;

    return module;
}

#else
PyMODINIT_FUNC
initlibTupanDP(void)
{
    PyObject *module = Py_InitModule3("libTupanDP", libTupan_meths,
                                      "An extension module for Tupan.");
    import_array();

    if (module == NULL)
        return;
}
#endif


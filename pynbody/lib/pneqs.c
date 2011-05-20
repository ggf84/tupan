#include "Python.h"
#include<math.h>


double get_pnacc(double a)
{
    return sqrt(a);
}



static PyObject *pneqsError;


static PyObject *_get_pnacc(PyObject *_self, PyObject *_a)
{
    double a = PyFloat_AsDouble(_a);
    return Py_BuildValue("dd", a, get_pnacc(a));
}


static PyMethodDef pneqs_meths[] = {
    {"get_pnacc", (PyCFunction)_get_pnacc, METH_O,
                  "returns PN corrections of accel."},
    {NULL, NULL, 0, NULL},
};


PyMODINIT_FUNC initpneqs(void)
{
    PyObject *ret;

    ret = Py_InitModule("pneqs", pneqs_meths);

    if (ret == NULL)
        return;

    pneqsError = PyErr_NewException("pneqs.error", NULL, NULL);
    Py_INCREF(pneqsError);
    PyModule_AddObject(ret, "error", pneqsError);
}


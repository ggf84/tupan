#include"Python.h"
#include"numpy/arrayobject.h"

#include"common.h"
#include"p2p_phi_kernel_core.h"
#include"p2p_acc_kernel_core.h"
#include"p2p_pnacc_kernel_core.h"


//
// Phi methods
////////////////////////////////////////////////////////////////////////////////
static PyObject *
_p2p_phi_kernel(PyObject *_args)
{
    unsigned int ni, nj;
    PyObject *_ipos = NULL, *_ieps2 = NULL;
    PyObject *_jpos = NULL, *_jeps2 = NULL;

    if (!PyArg_ParseTuple(_args, "OOOOII", &_ipos, &_ieps2,
                                           &_jpos, &_jeps2,
                                           &ni, &nj))
        return NULL;

    // i-data
    PyObject *_ipos_arr = PyArray_FROM_OTF(_ipos, NPY_FLOAT64, NPY_IN_ARRAY);
    double *ipos_ptr = (double *)PyArray_DATA(_ipos_arr);
    PyObject *_ieps2_arr = PyArray_FROM_OTF(_ieps2, NPY_FLOAT64, NPY_IN_ARRAY);
    double *ieps2_ptr = (double *)PyArray_DATA(_ieps2_arr);

    // j-data
    PyObject *_jpos_arr = PyArray_FROM_OTF(_jpos, NPY_FLOAT64, NPY_IN_ARRAY);
    double *jpos_ptr = (double *)PyArray_DATA(_jpos_arr);
    PyObject *_jeps2_arr = PyArray_FROM_OTF(_jeps2, NPY_FLOAT64, NPY_IN_ARRAY);
    double *jeps2_ptr = (double *)PyArray_DATA(_jeps2_arr);

    // allocate a PyArrayObject to be returned
    npy_intp dims[1] = {ni};
    PyArrayObject *ret = (PyArrayObject *)PyArray_EMPTY(1, dims, NPY_FLOAT64, 0);
    double *ret_ptr = (double *)PyArray_DATA(ret);

    // main calculation
    unsigned int i, iiii, j, jjjj;
    for (i = 0; i < ni; ++i) {
        iiii = 4*i;
        REAL iphi = 0.0;
        REAL4 ri = {ipos_ptr[iiii  ], ipos_ptr[iiii+1],
                    ipos_ptr[iiii+2], ipos_ptr[iiii+3]};
        REAL ieps2 = ieps2_ptr[i];
        for (j = 0; j < nj; ++j) {
            jjjj = 4*j;
            REAL4 rj = {jpos_ptr[jjjj  ], jpos_ptr[jjjj+1],
                        jpos_ptr[jjjj+2], jpos_ptr[jjjj+3]};
            REAL jeps2 = jeps2_ptr[j];
            iphi = p2p_phi_kernel_core(iphi, ri, ieps2, rj, jeps2);
        }
        ret_ptr[i] = iphi;
    }

    // Decrement the reference counts for i-objects
    Py_DECREF(_ipos_arr);
    Py_DECREF(_ieps2_arr);

    // Decrement the reference counts for j-objects
    Py_DECREF(_jpos_arr);
    Py_DECREF(_jeps2_arr);

    // returns a PyArrayObject
    return PyArray_Return(ret);
}


//
// Acc methods
////////////////////////////////////////////////////////////////////////////////
static PyObject *
_p2p_acc_kernel(PyObject *_args)
{
    unsigned int ni, nj;
    double tstep;
    PyObject *_ipos = NULL, *_ivel = NULL;
    PyObject *_jpos = NULL, *_jvel = NULL;

    if (!PyArg_ParseTuple(_args, "OOOOIId", &_ipos, &_ivel,
                                            &_jpos, &_jvel,
                                            &ni, &nj, &tstep))
        return NULL;

    // i-data
    PyObject *_ipos_arr = PyArray_FROM_OTF(_ipos, NPY_FLOAT64, NPY_IN_ARRAY);
    double *ipos_ptr = (double *)PyArray_DATA(_ipos_arr);
    PyObject *_ivel_arr = PyArray_FROM_OTF(_ivel, NPY_FLOAT64, NPY_IN_ARRAY);
    double *ivel_ptr = (double *)PyArray_DATA(_ivel_arr);

    // j-data
    PyObject *_jpos_arr = PyArray_FROM_OTF(_jpos, NPY_FLOAT64, NPY_IN_ARRAY);
    double *jpos_ptr = (double *)PyArray_DATA(_jpos_arr);
    PyObject *_jvel_arr = PyArray_FROM_OTF(_jvel, NPY_FLOAT64, NPY_IN_ARRAY);
    double *jvel_ptr = (double *)PyArray_DATA(_jvel_arr);

    // allocate a PyArrayObject to be returned
    npy_intp dims[2] = {ni, 4};
    PyArrayObject *ret = (PyArrayObject *)PyArray_EMPTY(2, dims, NPY_FLOAT64, 0);
    double *ret_ptr = (double *)PyArray_DATA(ret);

    // main calculation
    unsigned int i, iiii, j, jjjj;
    for (i = 0; i < ni; ++i) {
        iiii = 4*i;
        REAL4 iacc = {0.0, 0.0, 0.0, 0.0};
        REAL4 ri = {ipos_ptr[iiii  ], ipos_ptr[iiii+1],
                    ipos_ptr[iiii+2], ipos_ptr[iiii+3]};
        REAL4 vi = {ivel_ptr[iiii  ], ivel_ptr[iiii+1],
                    ivel_ptr[iiii+2], ivel_ptr[iiii+3]};
        for (j = 0; j < nj; ++j) {
            jjjj = 4*j;
            REAL4 rj = {jpos_ptr[jjjj  ], jpos_ptr[jjjj+1],
                        jpos_ptr[jjjj+2], jpos_ptr[jjjj+3]};
            REAL4 vj = {jvel_ptr[jjjj  ], jvel_ptr[jjjj+1],
                        jvel_ptr[jjjj+2], jvel_ptr[jjjj+3]};
            iacc = p2p_acc_kernel_core(iacc, ri, vi, rj, vj, tstep);
        }
        ret_ptr[iiii  ] = iacc.x;
        ret_ptr[iiii+1] = iacc.y;
        ret_ptr[iiii+2] = iacc.z;
        ret_ptr[iiii+3] = iacc.w;
    }

    // Decrement the reference counts for i-objects
    Py_DECREF(_ipos_arr);
    Py_DECREF(_ivel_arr);

    // Decrement the reference counts for j-objects
    Py_DECREF(_jpos_arr);
    Py_DECREF(_jvel_arr);

    // returns a PyArrayObject
    return PyArray_Return(ret);
}


//
// PN Acc methods
////////////////////////////////////////////////////////////////////////////////
static PyObject *
_p2p_pnacc_kernel(PyObject *_args)
{
    unsigned int ni, nj;
    CLIGHT clight;
    PyObject *_ipos = NULL, *_ivel = NULL;
    PyObject *_jpos = NULL, *_jvel = NULL;

    if (!PyArg_ParseTuple(_args, "OOOOIIIddddddd", &_ipos, &_ivel,
                                                   &_jpos, &_jvel,
                                                   &ni, &nj,
                                                   &clight.order, &clight.inv1,
                                                   &clight.inv2, &clight.inv3,
                                                   &clight.inv4, &clight.inv5,
                                                   &clight.inv6, &clight.inv7))
        return NULL;

    // i-data
    PyObject *_ipos_arr = PyArray_FROM_OTF(_ipos, NPY_FLOAT64, NPY_IN_ARRAY);
    double *ipos_ptr = (double *)PyArray_DATA(_ipos_arr);
    PyObject *_ivel_arr = PyArray_FROM_OTF(_ivel, NPY_FLOAT64, NPY_IN_ARRAY);
    double *ivel_ptr = (double *)PyArray_DATA(_ivel_arr);

    // j-data
    PyObject *_jpos_arr = PyArray_FROM_OTF(_jpos, NPY_FLOAT64, NPY_IN_ARRAY);
    double *jpos_ptr = (double *)PyArray_DATA(_jpos_arr);
    PyObject *_jvel_arr = PyArray_FROM_OTF(_jvel, NPY_FLOAT64, NPY_IN_ARRAY);
    double *jvel_ptr = (double *)PyArray_DATA(_jvel_arr);

    // allocate a PyArrayObject to be returned
    npy_intp dims[2] = {ni, 3};
    PyArrayObject *ret = (PyArrayObject *)PyArray_EMPTY(2, dims, NPY_FLOAT64, 0);
    double *ret_ptr = (double *)PyArray_DATA(ret);

    // main calculation
    unsigned int i, iii, iiii, j, jjjj;
    for (i = 0; i < ni; ++i) {
        iii = 3*i;
        iiii = 4*i;
        REAL3 ipnacc = {0.0, 0.0, 0.0};
        REAL4 ri = {ipos_ptr[iiii  ], ipos_ptr[iiii+1],
                    ipos_ptr[iiii+2], ipos_ptr[iiii+3]};
        REAL4 vi = {ivel_ptr[iiii  ], ivel_ptr[iiii+1],
                    ivel_ptr[iiii+2], ivel_ptr[iiii+3]};
        for (j = 0; j < nj; ++j) {
            jjjj = 4*j;
            REAL4 rj = {jpos_ptr[jjjj  ], jpos_ptr[jjjj+1],
                        jpos_ptr[jjjj+2], jpos_ptr[jjjj+3]};
            REAL4 vj = {jvel_ptr[jjjj  ], jvel_ptr[jjjj+1],
                        jvel_ptr[jjjj+2], jvel_ptr[jjjj+3]};
            ipnacc = p2p_pnacc_kernel_core(ipnacc, ri, vi, rj, vj, clight);
        }
        ret_ptr[iii  ] = ipnacc.x;
        ret_ptr[iii+1] = ipnacc.y;
        ret_ptr[iii+2] = ipnacc.z;
    }

    // Decrement the reference counts for i-objects
    Py_DECREF(_ipos_arr);
    Py_DECREF(_ivel_arr);

    // Decrement the reference counts for j-objects
    Py_DECREF(_jpos_arr);
    Py_DECREF(_jvel_arr);

    // returns a PyArrayObject
    return PyArray_Return(ret);
}


//
// Python interface methods
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
p2p_pnacc_kernel(PyObject *_self, PyObject *_args)
{
    return _p2p_pnacc_kernel(_args);
}


static PyMethodDef libc_gravity_meths[] = {
    {"p2p_phi_kernel", (PyCFunction)p2p_phi_kernel, METH_VARARGS,
                "returns the Newtonian gravitational potential."},
    {"p2p_acc_kernel", (PyCFunction)p2p_acc_kernel, METH_VARARGS,
                "returns the Newtonian gravitational acceleration."},
    {"p2p_pnacc_kernel", (PyCFunction)p2p_pnacc_kernel, METH_VARARGS,
                "returns the post-Newtonian gravitational acceleration."},
    {NULL, NULL, 0, NULL},
};


PyMODINIT_FUNC initlibc32_gravity(void)
{
    PyObject *ret;

    ret = Py_InitModule3("libc32_gravity", libc_gravity_meths,
                         "A extension module for Newtonian and post-Newtonian"
                         " gravity.");

    import_array();

    if (ret == NULL)
        return;
}


PyMODINIT_FUNC initlibc64_gravity(void)
{
    PyObject *ret;

    ret = Py_InitModule3("libc64_gravity", libc_gravity_meths,
                         "A extension module for Newtonian and post-Newtonian"
                         " gravity.");

    import_array();

    if (ret == NULL)
        return;
}


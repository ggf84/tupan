#include"Python.h"
#include"numpy/arrayobject.h"

#include"common.h"
#include"gravity_kernels.h"


//
// Phi methods
////////////////////////////////////////////////////////////////////////////////
static PyObject *
_p2p_phi_kernel(PyObject *_args)
{
    unsigned int ni, nj;
    PyObject *_ipos = NULL, *_ieps2 = NULL;
    PyObject *_jpos = NULL, *_jeps2 = NULL;

    int typenum;
    char *fmt = NULL;
    if (sizeof(REAL) == sizeof(double)) {
        fmt = "IOOIOO";
        typenum = NPY_FLOAT64;
    } else if (sizeof(REAL) == sizeof(float)) {
        fmt = "IOOIOO";
        typenum = NPY_FLOAT32;
    }

    if (!PyArg_ParseTuple(_args, fmt, &ni, &_ipos, &_ieps2,
                                      &nj, &_jpos, &_jeps2))
        return NULL;

    // i-data
    PyObject *_ipos_arr = PyArray_FROM_OTF(_ipos, typenum, NPY_IN_ARRAY);
    REAL *ipos_ptr = (REAL *)PyArray_DATA(_ipos_arr);
    PyObject *_ieps2_arr = PyArray_FROM_OTF(_ieps2, typenum, NPY_IN_ARRAY);
    REAL *ieps2_ptr = (REAL *)PyArray_DATA(_ieps2_arr);

    // j-data
    PyObject *_jpos_arr = PyArray_FROM_OTF(_jpos, typenum, NPY_IN_ARRAY);
    REAL *jpos_ptr = (REAL *)PyArray_DATA(_jpos_arr);
    PyObject *_jeps2_arr = PyArray_FROM_OTF(_jeps2, typenum, NPY_IN_ARRAY);
    REAL *jeps2_ptr = (REAL *)PyArray_DATA(_jeps2_arr);

    // allocate a PyArrayObject to be returned
    npy_intp dims[1] = {ni};
    PyArrayObject *ret = (PyArrayObject *)PyArray_EMPTY(1, dims, typenum, 0);
    REAL *ret_ptr = (REAL *)PyArray_DATA(ret);

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
    PyObject *_ipos = NULL, *_ieps2 = NULL;
    PyObject *_jpos = NULL, *_jeps2 = NULL;

    int typenum;
    char *fmt = NULL;
    if (sizeof(REAL) == sizeof(double)) {
        fmt = "IOOIOO";
        typenum = NPY_FLOAT64;
    } else if (sizeof(REAL) == sizeof(float)) {
        fmt = "IOOIOO";
        typenum = NPY_FLOAT32;
    }

    if (!PyArg_ParseTuple(_args, fmt, &ni, &_ipos, &_ieps2,
                                      &nj, &_jpos, &_jeps2))
        return NULL;

    // i-data
    PyObject *_ipos_arr = PyArray_FROM_OTF(_ipos, typenum, NPY_IN_ARRAY);
    REAL *ipos_ptr = (REAL *)PyArray_DATA(_ipos_arr);
    PyObject *_ieps2_arr = PyArray_FROM_OTF(_ieps2, typenum, NPY_IN_ARRAY);
    REAL *ieps2_ptr = (REAL *)PyArray_DATA(_ieps2_arr);

    // j-data
    PyObject *_jpos_arr = PyArray_FROM_OTF(_jpos, typenum, NPY_IN_ARRAY);
    REAL *jpos_ptr = (REAL *)PyArray_DATA(_jpos_arr);
    PyObject *_jeps2_arr = PyArray_FROM_OTF(_jeps2, typenum, NPY_IN_ARRAY);
    REAL *jeps2_ptr = (REAL *)PyArray_DATA(_jeps2_arr);

    // allocate a PyArrayObject to be returned
    npy_intp dims[2] = {ni, 3};
    PyArrayObject *ret = (PyArrayObject *)PyArray_EMPTY(2, dims, typenum, 0);
    REAL *ret_ptr = (REAL *)PyArray_DATA(ret);

    // main calculation
    unsigned int i, iii, iiii, j, jjjj;
    for (i = 0; i < ni; ++i) {
        iii = 3*i;
        iiii = 4*i;
        REAL3 iacc = {0.0, 0.0, 0.0};
        REAL4 ri = {ipos_ptr[iiii  ], ipos_ptr[iiii+1],
                    ipos_ptr[iiii+2], ipos_ptr[iiii+3]};
        REAL ieps2 = ieps2_ptr[i];
        for (j = 0; j < nj; ++j) {
            jjjj = 4*j;
            REAL4 rj = {jpos_ptr[jjjj  ], jpos_ptr[jjjj+1],
                        jpos_ptr[jjjj+2], jpos_ptr[jjjj+3]};
            REAL jeps2 = jeps2_ptr[j];
            iacc = p2p_acc_kernel_core(iacc, ri, ieps2, rj, jeps2);
        }
        ret_ptr[iii  ] = iacc.x;
        ret_ptr[iii+1] = iacc.y;
        ret_ptr[iii+2] = iacc.z;
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
// AccTstep methods
////////////////////////////////////////////////////////////////////////////////
static PyObject *
_p2p_acctstep_kernel(PyObject *_args)
{
    unsigned int ni, nj;
    REAL eta;
    PyObject *_ipos = NULL, *_ivel = NULL;
    PyObject *_jpos = NULL, *_jvel = NULL;

    int typenum;
    char *fmt = NULL;
    if (sizeof(REAL) == sizeof(double)) {
        fmt = "IOOIOOd";
        typenum = NPY_FLOAT64;
    } else if (sizeof(REAL) == sizeof(float)) {
        fmt = "IOOIOOf";
        typenum = NPY_FLOAT32;
    }

    if (!PyArg_ParseTuple(_args, fmt, &ni, &_ipos, &_ivel,
                                      &nj, &_jpos, &_jvel,
                                      &eta))
        return NULL;

    // i-data
    PyObject *_ipos_arr = PyArray_FROM_OTF(_ipos, typenum, NPY_IN_ARRAY);
    REAL *ipos_ptr = (REAL *)PyArray_DATA(_ipos_arr);
    PyObject *_ivel_arr = PyArray_FROM_OTF(_ivel, typenum, NPY_IN_ARRAY);
    REAL *ivel_ptr = (REAL *)PyArray_DATA(_ivel_arr);

    // j-data
    PyObject *_jpos_arr = PyArray_FROM_OTF(_jpos, typenum, NPY_IN_ARRAY);
    REAL *jpos_ptr = (REAL *)PyArray_DATA(_jpos_arr);
    PyObject *_jvel_arr = PyArray_FROM_OTF(_jvel, typenum, NPY_IN_ARRAY);
    REAL *jvel_ptr = (REAL *)PyArray_DATA(_jvel_arr);

    // allocate a PyArrayObject to be returned
    npy_intp dims[2] = {ni, 4};
    PyArrayObject *ret = (PyArrayObject *)PyArray_EMPTY(2, dims, typenum, 0);
    REAL *ret_ptr = (REAL *)PyArray_DATA(ret);

    // main calculation
    unsigned int i, iiii, j, jjjj;
    for (i = 0; i < ni; ++i) {
        iiii = 4*i;
        REAL4 iacctstep = {0.0, 0.0, 0.0, 0.0};
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
            iacctstep = p2p_acctstep_kernel_core(iacctstep, ri, vi, rj, vj, eta);
        }
        ret_ptr[iiii  ] = iacctstep.x;
        ret_ptr[iiii+1] = iacctstep.y;
        ret_ptr[iiii+2] = iacctstep.z;
        ret_ptr[iiii+3] = iacctstep.w;
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
// Tstep methods
////////////////////////////////////////////////////////////////////////////////
static PyObject *
_p2p_tstep_kernel(PyObject *_args)
{
    unsigned int ni, nj;
    REAL eta;
    PyObject *_idata = NULL;
    PyObject *_jdata = NULL;

    int typenum;
    char *fmt = NULL;
    if (sizeof(REAL) == sizeof(double)) {
        fmt = "IOIOd";
        typenum = NPY_FLOAT64;
    } else if (sizeof(REAL) == sizeof(float)) {
        fmt = "IOIOf";
        typenum = NPY_FLOAT32;
    }

    if (!PyArg_ParseTuple(_args, fmt, &ni, &_idata,
                                      &nj, &_jdata,
                                      &eta))
        return NULL;

    // i-data
    PyObject *_idata_arr = PyArray_FROM_OTF(_idata, typenum, NPY_IN_ARRAY);
    REAL *idata_ptr = (REAL *)PyArray_DATA(_idata_arr);

    // j-data
    PyObject *_jdata_arr = PyArray_FROM_OTF(_jdata, typenum, NPY_IN_ARRAY);
    REAL *jdata_ptr = (REAL *)PyArray_DATA(_jdata_arr);

    // allocate a PyArrayObject to be returned
    npy_intp dims[1] = {ni};
    PyArrayObject *ret = (PyArrayObject *)PyArray_EMPTY(1, dims, typenum, 0);
    REAL *ret_ptr = (REAL *)PyArray_DATA(ret);

    // main calculation
    unsigned int i, i8, j, j8;
    for (i = 0; i < ni; ++i) {
        i8 = 8*i;
        REAL iinv_tstep = 0.0;
        REAL4 ri = {idata_ptr[i8  ], idata_ptr[i8+1],
                    idata_ptr[i8+2], idata_ptr[i8+3]};
        REAL4 vi = {idata_ptr[i8+4], idata_ptr[i8+5],
                    idata_ptr[i8+6], idata_ptr[i8+7]};
        for (j = 0; j < nj; ++j) {
            j8 = 8*j;
            REAL4 rj = {jdata_ptr[j8  ], jdata_ptr[j8+1],
                        jdata_ptr[j8+2], jdata_ptr[j8+3]};
            REAL4 vj = {jdata_ptr[j8+4], jdata_ptr[j8+5],
                        jdata_ptr[j8+6], jdata_ptr[j8+7]};
            iinv_tstep = p2p_tstep_kernel_core(iinv_tstep, ri, vi, rj, vj, eta);
        }
        ret_ptr[i] = iinv_tstep;
    }

    // Decrement the reference counts for i-objects
    Py_DECREF(_idata_arr);

    // Decrement the reference counts for j-objects
    Py_DECREF(_jdata_arr);

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

    int typenum;
    char *fmt = NULL;
    if (sizeof(REAL) == sizeof(double)) {
        fmt = "IOOIOOIddddddd";
        typenum = NPY_FLOAT64;
    } else if (sizeof(REAL) == sizeof(float)) {
        fmt = "IOOIOOIfffffff";
        typenum = NPY_FLOAT32;
    }

    if (!PyArg_ParseTuple(_args, fmt, &ni, &_ipos, &_ivel,
                                      &nj, &_jpos, &_jvel,
                                      &clight.order, &clight.inv1,
                                      &clight.inv2, &clight.inv3,
                                      &clight.inv4, &clight.inv5,
                                      &clight.inv6, &clight.inv7))
        return NULL;

    // i-data
    PyObject *_ipos_arr = PyArray_FROM_OTF(_ipos, typenum, NPY_IN_ARRAY);
    REAL *ipos_ptr = (REAL *)PyArray_DATA(_ipos_arr);
    PyObject *_ivel_arr = PyArray_FROM_OTF(_ivel, typenum, NPY_IN_ARRAY);
    REAL *ivel_ptr = (REAL *)PyArray_DATA(_ivel_arr);

    // j-data
    PyObject *_jpos_arr = PyArray_FROM_OTF(_jpos, typenum, NPY_IN_ARRAY);
    REAL *jpos_ptr = (REAL *)PyArray_DATA(_jpos_arr);
    PyObject *_jvel_arr = PyArray_FROM_OTF(_jvel, typenum, NPY_IN_ARRAY);
    REAL *jvel_ptr = (REAL *)PyArray_DATA(_jvel_arr);

    // allocate a PyArrayObject to be returned
    npy_intp dims[2] = {ni, 3};
    PyArrayObject *ret = (PyArrayObject *)PyArray_EMPTY(2, dims, typenum, 0);
    REAL *ret_ptr = (REAL *)PyArray_DATA(ret);

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
p2p_acctstep_kernel(PyObject *_self, PyObject *_args)
{
    return _p2p_acctstep_kernel(_args);
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


static PyMethodDef libc_gravity_meths[] = {
    {"p2p_phi_kernel", (PyCFunction)p2p_phi_kernel, METH_VARARGS,
                "returns the Newtonian gravitational potential."},
    {"p2p_acc_kernel", (PyCFunction)p2p_acc_kernel, METH_VARARGS,
                "returns the Newtonian gravitational acceleration."},
    {"p2p_acctstep_kernel", (PyCFunction)p2p_acctstep_kernel, METH_VARARGS,
                "returns the Newtonian gravitational acceleration and inverse of the timestep."},
    {"p2p_tstep_kernel", (PyCFunction)p2p_tstep_kernel, METH_VARARGS,
                "returns the inverse of the timestep."},
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


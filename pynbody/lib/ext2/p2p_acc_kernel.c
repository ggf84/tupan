#ifndef P2P_ACC_KERNEL_C
#define P2P_ACC_KERNEL_C

#include"common.h"
#include"p2p_acc_kernel_core.h"


static PyObject *
p2p_acc_kernel(PyObject *_args)
{
    unsigned int ni, nj;
    double eta;
    PyObject *_ipos = NULL, *_ivel = NULL;
    PyObject *_jpos = NULL, *_jvel = NULL;

    if (!PyArg_ParseTuple(_args, "OOOOIId", &_ipos, &_ivel,
                                            &_jpos, &_jvel,
                                            &ni, &nj, &eta))
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
            iacc = p2p_acc_kernel_core(iacc, ri, vi, rj, vj, eta);
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

#endif  // P2P_ACC_KERNEL_C


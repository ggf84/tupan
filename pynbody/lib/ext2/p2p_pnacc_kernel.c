#ifndef P2P_PNACC_KERNEL_C
#define P2P_PNACC_KERNEL_C

#include"common.h"
#include"p2p_pnacc_kernel_core.h"


static PyObject *
p2p_pnacc_kernel(PyObject *_args)
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
    unsigned int i, iii, iiii, j, jjj, jjjj;
    for (i = 0; i < ni; ++i) {
        iii = 3*i;
        iiii = 4*i;
        REAL3 ipnacc = {0.0, 0.0, 0.0};
        REAL vi2 = ivel_ptr[iii  ] * ivel_ptr[iii  ]
                 + ivel_ptr[iii+1] * ivel_ptr[iii+1]
                 + ivel_ptr[iii+2] * ivel_ptr[iii+2];
        REAL4 ri = {ipos_ptr[iiii  ], ipos_ptr[iiii+1],
                    ipos_ptr[iiii+2], ipos_ptr[iiii+3]};
        REAL4 vi = {ivel_ptr[iii  ], ivel_ptr[iii+1],
                    ivel_ptr[iii+2], vi2};
        for (j = 0; j < nj; ++j) {
            jjj = 3*j;
            jjjj = 4*j;
            REAL vj2 = jvel_ptr[jjj  ] * jvel_ptr[jjj  ]
                     + jvel_ptr[jjj+1] * jvel_ptr[jjj+1]
                     + jvel_ptr[jjj+2] * jvel_ptr[jjj+2];
            REAL4 rj = {jpos_ptr[jjjj  ], jpos_ptr[jjjj+1],
                        jpos_ptr[jjjj+2], jpos_ptr[jjjj+3]};
            REAL4 vj = {jvel_ptr[jjj  ], jvel_ptr[jjj+1],
                        jvel_ptr[jjj+2], vj2};
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

#endif  // P2P_PNACC_KERNEL_C


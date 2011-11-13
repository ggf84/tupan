#ifndef P2P_PHI_KERNEL_C
#define P2P_PHI_KERNEL_C

#include"common.h"
#include"p2p_phi_kernel_core.h"


static PyObject *
p2p_phi_kernel(PyObject *_args)
{
    unsigned int ni, nj;
    PyObject *_ipos = NULL, *_ieps2 = NULL;
    PyObject *_jpos = NULL, *_jeps2 = NULL;

    if (!PyArg_ParseTuple(_args, "IIOOOO", &ni, &nj,
                                           &_ipos, &_ieps2,
                                           &_jpos, &_jeps2))
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
        REAL4 bi = {ipos_ptr[iiii  ], ipos_ptr[iiii+1],
                    ipos_ptr[iiii+2], ipos_ptr[iiii+3]};
        REAL ieps2 = ieps2_ptr[i];
        for (j = 0; j < nj; ++j) {
            jjjj = 4*j;
            REAL4 bj = {jpos_ptr[jjjj  ], jpos_ptr[jjjj+1],
                        jpos_ptr[jjjj+2], jpos_ptr[jjjj+3]};
            REAL jeps2 = jeps2_ptr[j];
            iphi = p2p_phi_kernel_core(iphi, bi, ieps2, bj, jeps2);
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

#endif  // P2P_PHI_KERNEL_C


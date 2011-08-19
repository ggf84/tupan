#include"Python.h"
#include"numpy/arrayobject.h"
#include<math.h>

#define rsqrt(x) (1.0/sqrt(x))

typedef double REAL;


typedef struct real2_struct {
  REAL x;
  REAL y;
} REAL2, *pREAL2;


typedef struct real3_struct {
  REAL x;
  REAL y;
  REAL z;
} REAL3, *pREAL3;


typedef struct real4_struct {
  REAL x;
  REAL y;
  REAL z;
  REAL w;
} REAL4, *pREAL4;



#include"p2p_acc_kernel_core.h"
#include"p2p_phi_kernel_core.h"


static PyObject *_set_phi(PyObject *_self, PyObject *_bi, PyObject *_bj)
{
    /* i-data */
    unsigned long ni = (unsigned long)PyObject_Length(_bi);
    PyObject *_imass = PyObject_GetAttrString(_bi, "mass");
    PyObject *_imass_arr = PyArray_FROM_OTF(_imass, NPY_FLOAT64, NPY_IN_ARRAY);
    double *imass = (double *)PyArray_DATA(_imass_arr);
    PyObject *_ieps2 = PyObject_GetAttrString(_bi, "eps2");
    PyObject *_ieps2_arr = PyArray_FROM_OTF(_ieps2, NPY_FLOAT64, NPY_IN_ARRAY);
    double *ieps2 = (double *)PyArray_DATA(_ieps2_arr);
    PyObject *_ipos = PyObject_GetAttrString(_bi, "pos");
    PyObject *_ipos_arr = PyArray_FROM_OTF(_ipos, NPY_FLOAT64, NPY_IN_ARRAY);
    double *ipos = (double *)PyArray_DATA(_ipos_arr);

    /* j-data */
    unsigned long nj = (unsigned long)PyObject_Length(_bj);
    PyObject *_jmass = PyObject_GetAttrString(_bj, "mass");
    PyObject *_jmass_arr = PyArray_FROM_OTF(_jmass, NPY_FLOAT64, NPY_IN_ARRAY);
    double *jmass = (double *)PyArray_DATA(_jmass_arr);
    PyObject *_jeps2 = PyObject_GetAttrString(_bj, "eps2");
    PyObject *_jeps2_arr = PyArray_FROM_OTF(_jeps2, NPY_FLOAT64, NPY_IN_ARRAY);
    double *jeps2 = (double *)PyArray_DATA(_jeps2_arr);
    PyObject *_jpos = PyObject_GetAttrString(_bj, "pos");
    PyObject *_jpos_arr = PyArray_FROM_OTF(_jpos, NPY_FLOAT64, NPY_IN_ARRAY);
    double *jpos = (double *)PyArray_DATA(_jpos_arr);

    /* allocate a PyArrayObject to be returned */
    npy_intp dims[1] = {ni};
    PyArrayObject *ret = (PyArrayObject *)PyArray_EMPTY(1, dims, NPY_FLOAT64, 0);
    double *ret_ptr = (double *)PyArray_DATA(ret);

    /* main calculation */
    unsigned long i, ii, j, jj;
    for (i = 0; i < ni; ++i) {
        ii = 3*i;
        REAL iphi = 0.0;
        REAL4 bi = {ipos[ii  ], ipos[ii+1], ipos[ii+2], ieps2[i]};
        REAL mi = imass[i];
        for (j = 0; j < nj; ++j) {
            jj = 3*j;
            REAL4 bj = {jpos[jj  ], jpos[jj+1], jpos[jj+2], jeps2[j]};
            REAL mj = jmass[j];
            iphi = p2p_phi_kernel_core(iphi, bi, mi, bj, mj);
        }
        ret_ptr[i] = iphi;
    }

    /* Decrement the reference counts for i-objects */
    Py_DECREF(_imass_arr);
    Py_DECREF(_imass);
    Py_DECREF(_ieps2_arr);
    Py_DECREF(_ieps2);
    Py_DECREF(_ipos_arr);
    Py_DECREF(_ipos);

    /* Decrement the reference counts for j-objects */
    Py_DECREF(_jmass_arr);
    Py_DECREF(_jmass);
    Py_DECREF(_jeps2_arr);
    Py_DECREF(_jeps2);
    Py_DECREF(_jpos_arr);
    Py_DECREF(_jpos);

    /* returns a PyArrayObject */
    return PyArray_Return(ret);
}



static PyObject *_set_acc(PyObject *_self, PyObject *_bi, PyObject *_bj)
{
    /* i-data */
    unsigned long ni = (unsigned long)PyObject_Length(_bi);
    PyObject *_imass = PyObject_GetAttrString(_bi, "mass");
    PyObject *_imass_arr = PyArray_FROM_OTF(_imass, NPY_FLOAT64, NPY_IN_ARRAY);
    double *imass = (double *)PyArray_DATA(_imass_arr);
    PyObject *_ieps2 = PyObject_GetAttrString(_bi, "eps2");
    PyObject *_ieps2_arr = PyArray_FROM_OTF(_ieps2, NPY_FLOAT64, NPY_IN_ARRAY);
    double *ieps2 = (double *)PyArray_DATA(_ieps2_arr);
    PyObject *_ipos = PyObject_GetAttrString(_bi, "pos");
    PyObject *_ipos_arr = PyArray_FROM_OTF(_ipos, NPY_FLOAT64, NPY_IN_ARRAY);
    double *ipos = (double *)PyArray_DATA(_ipos_arr);

    /* j-data */
    unsigned long nj = (unsigned long)PyObject_Length(_bj);
    PyObject *_jmass = PyObject_GetAttrString(_bj, "mass");
    PyObject *_jmass_arr = PyArray_FROM_OTF(_jmass, NPY_FLOAT64, NPY_IN_ARRAY);
    double *jmass = (double *)PyArray_DATA(_jmass_arr);
    PyObject *_jeps2 = PyObject_GetAttrString(_bj, "eps2");
    PyObject *_jeps2_arr = PyArray_FROM_OTF(_jeps2, NPY_FLOAT64, NPY_IN_ARRAY);
    double *jeps2 = (double *)PyArray_DATA(_jeps2_arr);
    PyObject *_jpos = PyObject_GetAttrString(_bj, "pos");
    PyObject *_jpos_arr = PyArray_FROM_OTF(_jpos, NPY_FLOAT64, NPY_IN_ARRAY);
    double *jpos = (double *)PyArray_DATA(_jpos_arr);

    /* allocate a PyArrayObject to be returned */
    npy_intp dims[2] = {ni, 4};
    PyArrayObject *ret = (PyArrayObject *)PyArray_EMPTY(2, dims, NPY_FLOAT64, 0);
    double *ret_ptr = (double *)PyArray_DATA(ret);

    /* main calculation */
    unsigned long i, ii, iii, j, jj;
    for (i = 0; i < ni; ++i) {
        ii = 3*i;
        REAL4 iret = {0.0, 0.0, 0.0, 0.0};
        REAL4 bi = {ipos[ii  ], ipos[ii+1], ipos[ii+2], ieps2[i]};
        REAL mi = imass[i];
        for (j = 0; j < nj; ++j) {
            jj = 3*j;
            REAL4 bj = {jpos[jj  ], jpos[jj+1], jpos[jj+2], jeps2[j]};
            REAL mj = jmass[j];
            iret = p2p_acc_kernel_core(iret, bi, mi, bj, mj);
        }
        iii = 4*i;
        ret_ptr[iii  ] = iret.x;
        ret_ptr[iii+1] = iret.y;
        ret_ptr[iii+2] = iret.z;
        ret_ptr[iii+3] = iret.w;
    }

    /* Decrement the reference counts for i-objects */
    Py_DECREF(_imass_arr);
    Py_DECREF(_imass);
    Py_DECREF(_ieps2_arr);
    Py_DECREF(_ieps2);
    Py_DECREF(_ipos_arr);
    Py_DECREF(_ipos);

    /* Decrement the reference counts for j-objects */
    Py_DECREF(_jmass_arr);
    Py_DECREF(_jmass);
    Py_DECREF(_jeps2_arr);
    Py_DECREF(_jeps2);
    Py_DECREF(_jpos_arr);
    Py_DECREF(_jpos);

    /* returns a PyArrayObject */
    return PyArray_Return(ret);
}


static PyObject *_wrapper(PyObject *_self, PyObject *_args,
                          PyObject *(*func)(PyObject *,
                                            PyObject *,
                                            PyObject *))
{
    PyObject *_bi=NULL, *_bj=NULL;
    if(!PyArg_ParseTuple(_args,"OO", &_bi, &_bj)) {
        return NULL;
    }
    return func(_self, _bi, _bj);
}


static PyObject *set_acc(PyObject *_self, PyObject *_args)
{
    return _wrapper(_self, _args, _set_acc);
}


static PyObject *set_phi(PyObject *_self, PyObject *_args)
{
    return _wrapper(_self, _args, _set_phi);
}


static PyMethodDef libgravity_meths[] = {
    {"c_p2p_acc", (PyCFunction)set_acc, METH_VARARGS,
                "returns the Newtonian gravitational acceleration."},
    {"c_p2p_phi", (PyCFunction)set_phi, METH_VARARGS,
                "returns the Newtonian gravitational potential."},

    {NULL, NULL, 0, NULL},
};


PyMODINIT_FUNC initlibgravity(void)
{
    PyObject *ret;

    ret = Py_InitModule3("libgravity", libgravity_meths,
                         "A extension module for Newtonian gravity.");

    import_array();

    if (ret == NULL)
        return;
}


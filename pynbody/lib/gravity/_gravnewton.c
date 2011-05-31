#include"Python.h"
#include"numpy/arrayobject.h"
#include<math.h>

#define rsqrt(x) (1.0/sqrt(x))

typedef double REAL;


typedef struct vec3_struct {
  REAL x;
  REAL y;
  REAL z;
} REAL3, *pREAL3;


typedef struct vec4_struct {
  REAL x;
  REAL y;
  REAL z;
  REAL w;
} REAL4, *pREAL4;



#include"p2p_acc_kernel_core.h"
#include"p2p_phi_kernel_core.h"


REAL set_phi( unsigned long iindex,
              double imass,
              double ieps2,
              double *ipos,
              double *ivel,
              unsigned long *jindex_array,
              double *jmass_array,
              double *jeps2_array,
              double *jpos_array,
              double *jvel_array,
              unsigned long nj
            )
{
    unsigned long j, jj;
    REAL phi = 0.0;
    REAL4 bi = {ipos[0], ipos[1], ipos[2], ieps2};

    for (j = 0; j < nj; ++j) {
        jj = 3*j;
        REAL4 bj = {jpos_array[jj  ], jpos_array[jj+1],
                    jpos_array[jj+2], jeps2_array[j]};
        REAL mj = jmass_array[j];
        phi = p2p_phi_kernel_core(phi, bi, bj, mj);
    }

    return phi;
}


REAL4 set_acc( unsigned long iindex,
              double imass,
              double ieps2,
              double *ipos,
              double *ivel,
              unsigned long *jindex_array,
              double *jmass_array,
              double *jeps2_array,
              double *jpos_array,
              double *jvel_array,
              unsigned long nj
            )
{
    unsigned long j, jj;
    REAL4 acc = {0.0, 0.0, 0.0, 0.0};
    REAL4 bip = {ipos[0], ipos[1], ipos[2], ieps2};
    REAL4 biv = {ivel[0], ivel[1], ivel[2], imass};

    for (j = 0; j < nj; ++j) {
        jj = 3*j;
        REAL4 bjp = {jpos_array[jj  ], jpos_array[jj+1],
                     jpos_array[jj+2], jeps2_array[j]};
        REAL4 bjv = {jvel_array[jj  ], jvel_array[jj+1],
                     jvel_array[jj+2], jmass_array[j]};
        acc = p2p_acc_kernel_core(acc, bip, biv, bjp, bjv);
    }

    return acc;
}



/*static PyObject *_gravityError;*/


#define pcast_to_ulong(p) ((unsigned long *) (((PyArrayObject *)p)->data))
#define pcast_to_double(p) ((double *) (((PyArrayObject *)p)->data))

static PyObject *_set_phi(PyObject *_self, PyObject *_args)
{
    unsigned long iindex, *jindex_array;
    double imass, ieps2, *ipos, *ivel;
    double *jmass_array, *jeps2_array, *jpos_array, *jvel_array;
    PyObject *_ipos, *_ivel;
    PyObject *_jindex_array, *_jmass_array, *_jeps2_array,
             *_jpos_array, *_jvel_array;

    if(!PyArg_ParseTuple(_args,"kddOOOOOOO",
                           &iindex, &imass, &ieps2, &_ipos, &_ivel,
                           &_jindex_array, &_jmass_array, &_jeps2_array,
                           &_jpos_array, &_jvel_array)) {
        return NULL;
    }

    unsigned long nj = (unsigned long)PyObject_Length(_jindex_array);

/*    iindex = pcast_to_ulong(_iindex);*/
/*    imass = pcast_to_double(_imass);*/
/*    ieps2 = pcast_to_double(_ieps2);*/
    ipos = pcast_to_double(_ipos);
    ivel = pcast_to_double(_ivel);
    jindex_array = pcast_to_ulong(_jindex_array);
    jeps2_array = pcast_to_double(_jeps2_array);
    jmass_array = pcast_to_double(_jmass_array);
    jpos_array = pcast_to_double(_jpos_array);
    jvel_array = pcast_to_double(_jvel_array);

    double phi = set_phi(iindex, imass, ieps2, ipos, ivel,
                         jindex_array, jmass_array, jeps2_array,
                         jpos_array, jvel_array, nj);

    return Py_BuildValue("d", phi);
}



static PyObject *_set_acc(PyObject *_self, PyObject *_args)
{
    unsigned long iindex, *jindex_array;
    double imass, ieps2, *ipos, *ivel;
    double *jmass_array, *jeps2_array, *jpos_array, *jvel_array;
    PyObject *_ipos, *_ivel;
    PyObject *_jindex_array, *_jmass_array, *_jeps2_array,
             *_jpos_array, *_jvel_array;

    if(!PyArg_ParseTuple(_args,"kddOOOOOOO",
                           &iindex, &imass, &ieps2, &_ipos, &_ivel,
                           &_jindex_array, &_jmass_array, &_jeps2_array,
                           &_jpos_array, &_jvel_array)) {
        return NULL;
    }

    unsigned long nj = (unsigned long)PyObject_Length(_jindex_array);

/*    iindex = pcast_to_ulong(_iindex);*/
/*    imass = pcast_to_double(_imass);*/
/*    ieps2 = pcast_to_double(_ieps2);*/
    ipos = pcast_to_double(_ipos);
    ivel = pcast_to_double(_ivel);
    jindex_array = pcast_to_ulong(_jindex_array);
    jeps2_array = pcast_to_double(_jeps2_array);
    jmass_array = pcast_to_double(_jmass_array);
    jpos_array = pcast_to_double(_jpos_array);
    jvel_array = pcast_to_double(_jvel_array);

    REAL4 acc = set_acc(iindex, imass, ieps2, ipos, ivel,
                        jindex_array, jmass_array, jeps2_array,
                        jpos_array, jvel_array, nj);

    return Py_BuildValue("dddd", acc.x, acc.y, acc.z, acc.w);
}


static PyMethodDef _gravnewton_meths[] = {
    {"set_acc", (PyCFunction)_set_acc, METH_VARARGS,
                "returns the Newtonian gravitational acceleration."},
    {"set_phi", (PyCFunction)_set_phi, METH_VARARGS,
                "returns the Newtonian gravitational potential."},
    {NULL, NULL, 0, NULL},
};


PyMODINIT_FUNC init_gravnewton(void)
{
    PyObject *ret;

    ret = Py_InitModule3("_gravnewton", _gravnewton_meths,
                         "A extension module for Newtonian gravity.");

    import_array();

    if (ret == NULL)
        return;

/*    _gravityError = PyErr_NewException("_gravity.error", NULL, NULL);*/
/*    Py_INCREF(_gravityError);*/
/*    PyModule_AddObject(ret, "error", _gravityError);*/
}


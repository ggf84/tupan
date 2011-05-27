#include "Python.h"
#include "numpy/arrayobject.h"
#include<math.h>


typedef struct vec3_struct {
  double x;
  double y;
  double z;
} VEC3, *pVEC3;


typedef struct vec4_struct {
  double x;
  double y;
  double z;
  double w;
} VEC4, *pVEC4;




double set_phi( unsigned long iindex,
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
    double phi = 0.0;

    unsigned long j, jj;
    VEC3 dr;
    double dr2, eps2, rinv;
    for (j = 0; j < nj; ++j) {
        if (iindex != jindex_array[j]) {
            jj = 3*j;
            dr.x = ipos[0] - jpos_array[jj  ];
            dr.y = ipos[1] - jpos_array[jj+1];
            dr.z = ipos[2] - jpos_array[jj+2];
            eps2 = ieps2 + jeps2_array[j];
            dr2 = ((dr.x * dr.x + dr.y * dr.y) + dr.z * dr.z) + eps2;
            rinv = 1.0/sqrt(dr2);
            phi -= jmass_array[j] * rinv;
        }
    }

    return phi;
}





VEC4 set_acc( unsigned long iindex,
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
    VEC4 acc = {0.0, 0.0, 0.0, 0.0};

    unsigned long j, jj;
    VEC3 dr, dv;
    double dr2, dv2, eps2, Mpair, elagr, rinv, rinv3;
    for (j = 0; j < nj; ++j) {
        if (iindex != jindex_array[j]) {
            jj = 3*j;
            dr.x = ipos[0] - jpos_array[jj  ];
            dr.y = ipos[1] - jpos_array[jj+1];
            dr.z = ipos[2] - jpos_array[jj+2];
            dv.x = ivel[0] - jvel_array[jj  ];
            dv.y = ivel[1] - jvel_array[jj+1];
            dv.z = ivel[2] - jvel_array[jj+2];
            eps2 = ieps2 + jeps2_array[j];
            Mpair = imass + jmass_array[j];
            dr2 = ((dr.x * dr.x + dr.y * dr.y) + dr.z * dr.z) + eps2;
            dv2 = ((dv.x * dv.x + dv.y * dv.y) + dv.z * dv.z);
            rinv = 1.0/sqrt(dr2);
            rinv3 = rinv * rinv;
            elagr = 0.0 * dv2 + Mpair * rinv;
            acc.w += elagr * rinv3;
            rinv3 *= jmass_array[j] * rinv;
            acc.x -= rinv3 * dr.x;
            acc.y -= rinv3 * dr.y;
            acc.z -= rinv3 * dr.z;
        }
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

    VEC4 acc = set_acc(iindex, imass, ieps2, ipos, ivel,
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


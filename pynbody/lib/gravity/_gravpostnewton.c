#include "Python.h"
#include "numpy/arrayobject.h"
#include<math.h>


typedef struct vector_struct {
  double x;
  double y;
  double z;
} VECT, *pVECT;


typedef struct clight_struct {
  double inv1;
  double inv2;
  double inv3;
  double inv4;
  double inv5;
  double inv6;
  double inv7;
} CLIGHT, *pCLIGHT;




VECT set_pnacc(unsigned long nj,
               unsigned long *jindex_array,
               double *jpos_array,
               double *jvel_array,
               double *jspin_array,
               unsigned long *iindex,
               double *ipos,
               double *ivel,
               double *ispin,
               CLIGHT clight
              )
{
    VECT pnacc = {0.0, 0.0, 0.0};

    pnacc.x = jpos_array[10];
    pnacc.y = iindex[2];
    pnacc.z = nj;

    int j, k;
    for (j = 0; j < nj; ++j) {
        for (k = 0; k < 3; ++k) {
            printf("%f\n", jpos_array[3*j+k]);
        }
    }

    return pnacc;
}



/*static PyObject *pneqsError;*/


#define cast_to_ulong(p) ((unsigned long *) (((PyArrayObject *)p)->data))
#define cast_to_double(p) ((double *) (((PyArrayObject *)p)->data))

static PyObject *_set_pnacc(PyObject *_self, PyObject *_args)
{
    CLIGHT clight;
    unsigned long nj, *iindex, *jindex_array;
    double *ipos, *ivel, *ispin;
    double *jpos_array, *jvel_array, *jspin_array;
    PyObject *_iindex, *_ipos, *_ivel, *_ispin;
    PyObject *_jindex_array, *_jpos_array, *_jvel_array, *_jspin_array;

    PyArg_ParseTuple(_args,"kOOOOOOOOddddddd", &nj,
                            &_jindex_array, &_jpos_array,
                            &_jvel_array, &_jspin_array,
                            &_iindex, &_ipos, &_ivel, &_ispin,
                            &clight.inv1, &clight.inv2, &clight.inv3,
                            &clight.inv4, &clight.inv5, &clight.inv6,
                            &clight.inv7);

    iindex = cast_to_ulong(_iindex);
    ipos = cast_to_double(_ipos);
    ivel = cast_to_double(_ivel);
    ispin = cast_to_double(_ispin);
    jindex_array = cast_to_ulong(_jindex_array);
    jpos_array = cast_to_double(_jpos_array);
    jvel_array = cast_to_double(_jvel_array);
    jspin_array = cast_to_double(_jspin_array);

    VECT pnacc = set_pnacc(nj,
                           jindex_array, jpos_array,
                           jvel_array, jspin_array,
                           iindex, ipos, ivel, ispin,
                           clight
                          );

    return Py_BuildValue("ddd", pnacc.x, pnacc.y, pnacc.z);
}


static PyMethodDef _gravpostnewton_meths[] = {
    {"set_pnacc", (PyCFunction)_set_pnacc, METH_VARARGS,
                  "returns the Post-Newtonian gravitational acceleration."},
    {NULL, NULL, 0, NULL},
};


PyMODINIT_FUNC init_gravpostnewton(void)
{
    PyObject *ret;

    ret = Py_InitModule3("_gravpostnewton", _gravpostnewton_meths,
                         "A extension module for Post-Newtonian gravity.");

    import_array();

    if (ret == NULL)
        return;

/*    pneqsError = PyErr_NewException("pneqs.error", NULL, NULL);*/
/*    Py_INCREF(pneqsError);*/
/*    PyModule_AddObject(ret, "error", pneqsError);*/
}


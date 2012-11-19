#ifndef P2P_ACC_KERNEL_H
#define P2P_ACC_KERNEL_H

#include"common.h"
#include"smoothing.h"

//
// p2p_acc_kernel_core
////////////////////////////////////////////////////////////////////////////////
inline REAL3
p2p_acc_kernel_core(REAL3 acc,
                    const REAL4 rmi, const REAL4 vei,
                    const REAL4 rmj, const REAL4 vej)
{
    REAL3 r;
    r.x = rmi.x - rmj.x;                                             // 1 FLOPs
    r.y = rmi.y - rmj.y;                                             // 1 FLOPs
    r.z = rmi.z - rmj.z;                                             // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs
    REAL inv_r3 = smoothed_inv_r3(r2, vei.w + vej.w);                // 5 FLOPs

    inv_r3 *= rmj.w;                                                 // 1 FLOPs

    acc.x -= inv_r3 * r.x;                                           // 2 FLOPs
    acc.y -= inv_r3 * r.y;                                           // 2 FLOPs
    acc.z -= inv_r3 * r.z;                                           // 2 FLOPs
    return acc;
}
// Total flop count: 20


#ifdef __OPENCL_VERSION__
//
// OpenCL implementation
////////////////////////////////////////////////////////////////////////////////
inline REAL3
p2p_accum_acc(REAL3 iAcc,
              const REAL8 iData,
              uint j_begin,
              uint j_end,
              __local REAL8 *sharedJData
             )
{
    uint j;
    for (j = j_begin; j < j_end; ++j) {
        REAL8 jData = sharedJData[j];
        iAcc = p2p_acc_kernel_core(iAcc,
                                   iData.lo, iData.hi,
                                   jData.lo, jData.hi);
    }
    return iAcc;
}


inline REAL3
p2p_acc_kernel_main_loop(const REAL8 iData,
                         const uint nj,
                         __global const REAL8 *jdata,
                         __local REAL8 *sharedJData
                        )
{
    uint lsize = get_local_size(0);

    REAL3 iAcc = (REAL3){0, 0, 0};

    uint tile;
    uint numTiles = (nj - 1)/lsize + 1;
    for (tile = 0; tile < numTiles; ++tile) {
        uint nb = min(lsize, (nj - (tile * lsize)));

        event_t e[1];
        e[0] = async_work_group_copy(sharedJData, jdata + tile * lsize, nb, 0);
        wait_group_events(1, e);

        uint j = 0;
        uint j_max = (nb > (JUNROLL - 1)) ? (nb - (JUNROLL - 1)):(0);
        for (; j < j_max; j += JUNROLL) {
            iAcc = p2p_accum_acc(iAcc, iData,
                                 j, j + JUNROLL,
                                 sharedJData);
        }
        iAcc = p2p_accum_acc(iAcc, iData,
                             j, nb,
                             sharedJData);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return iAcc;
}


__kernel void p2p_acc_kernel(const uint ni,
                             __global const REAL8 *idata,
                             const uint nj,
                             __global const REAL8 *jdata,
                             __global REAL3 *iacc,
                             __local REAL8 *sharedJData
                            )
{
    uint gid = get_global_id(0);
    uint i = (gid < ni) ? (gid) : (ni-1);
    iacc[i] = p2p_acc_kernel_main_loop(idata[i],
                                       nj, jdata,
                                       sharedJData);
}

#else
//
// C implementation
////////////////////////////////////////////////////////////////////////////////
static PyObject *
_p2p_acc_kernel(PyObject *_args)
{
    unsigned int ni, nj;
    PyObject *_idata = NULL;
    PyObject *_jdata = NULL;
    PyObject *_output = NULL;

    int typenum;
    char *fmt = NULL;
    if (sizeof(REAL) == sizeof(double)) {
        fmt = "IOIOO!";
        typenum = NPY_FLOAT64;
    } else if (sizeof(REAL) == sizeof(float)) {
        fmt = "IOIOO!";
        typenum = NPY_FLOAT32;
    }

    if (!PyArg_ParseTuple(_args, fmt, &ni, &_idata,
                                      &nj, &_jdata,
                                      &PyArray_Type, &_output))
        return NULL;

    // i-data
    PyObject *_idata_arr = PyArray_FROM_OTF(_idata, typenum, NPY_IN_ARRAY);
    REAL *idata_ptr = (REAL *)PyArray_DATA(_idata_arr);

    // j-data
    PyObject *_jdata_arr = PyArray_FROM_OTF(_jdata, typenum, NPY_IN_ARRAY);
    REAL *jdata_ptr = (REAL *)PyArray_DATA(_jdata_arr);

    // output-array
    PyObject *ret = PyArray_FROM_OTF(_output, typenum, NPY_INOUT_ARRAY);
    REAL *ret_ptr = (REAL *)PyArray_DATA(ret);
    int k = PyArray_DIM((PyArrayObject *)ret, 1);

    // main calculation
    unsigned int i, ik, i8, j, j8;
    for (i = 0; i < ni; ++i) {
        i8 = 8*i;
        REAL3 iacc = (REAL3){0, 0, 0};
        REAL4 rmi = {idata_ptr[i8  ], idata_ptr[i8+1],
                     idata_ptr[i8+2], idata_ptr[i8+3]};
        REAL4 vei = {idata_ptr[i8+4], idata_ptr[i8+5],
                     idata_ptr[i8+6], idata_ptr[i8+7]};
        for (j = 0; j < nj; ++j) {
            j8 = 8*j;
            REAL4 rmj = {jdata_ptr[j8  ], jdata_ptr[j8+1],
                         jdata_ptr[j8+2], jdata_ptr[j8+3]};
            REAL4 vej = {jdata_ptr[j8+4], jdata_ptr[j8+5],
                         jdata_ptr[j8+6], jdata_ptr[j8+7]};
            iacc = p2p_acc_kernel_core(iacc, rmi, vei, rmj, vej);
        }
        ik = i * k;
        ret_ptr[ik  ] = iacc.x;
        ret_ptr[ik+1] = iacc.y;
        ret_ptr[ik+2] = iacc.z;
    }

    // Decrement the reference counts for i-objects
    Py_DECREF(_idata_arr);

    // Decrement the reference counts for j-objects
    Py_DECREF(_jdata_arr);

    // Decrement the reference counts for ret-objects
    Py_DECREF(ret);

    Py_INCREF(Py_None);
    return Py_None;
}

#endif  // __OPENCL_VERSION__


#endif  // P2P_ACC_KERNEL_H


#ifndef P2P_ACC_JERK_KERNEL_H
#define P2P_ACC_JERK_KERNEL_H

#include"common.h"
#include"smoothing.h"

//
// p2p_acc_jerk_kernel_core
////////////////////////////////////////////////////////////////////////////////
inline REAL8
p2p_acc_jerk_kernel_core(REAL8 accjerk,
                         const REAL4 ri, const REAL4 vi,
                         const REAL4 rj, const REAL4 vj)
{
    REAL4 r;
    r.x = ri.x - rj.x;                                               // 1 FLOPs
    r.y = ri.y - rj.y;                                               // 1 FLOPs
    r.z = ri.z - rj.z;                                               // 1 FLOPs
    REAL4 v;
    v.x = vi.x - vj.x;                                               // 1 FLOPs
    v.y = vi.y - vj.y;                                               // 1 FLOPs
    v.z = vi.z - vj.z;                                               // 1 FLOPs
    v.w = vi.w + vj.w;                                               // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs
    REAL rv = r.x * v.x + r.y * v.y + r.z * v.z;                     // 5 FLOPs
    REAL2 ret = accjerk_smooth(r2, v.w);                             // 4 FLOPs
    REAL inv_r2 = ret.x;
    REAL inv_r3 = ret.y;

    inv_r3 *= rj.w;                                                  // 1 FLOPs
    rv *= 3 * inv_r2;                                                // 2 FLOPs

    accjerk.s0 -= inv_r3 * r.x;                                      // 2 FLOPs
    accjerk.s1 -= inv_r3 * r.y;                                      // 2 FLOPs
    accjerk.s2 -= inv_r3 * r.z;                                      // 2 FLOPs
    accjerk.s3  = 0;
    accjerk.s4 -= inv_r3 * (v.x - rv * r.x);                         // 4 FLOPs
    accjerk.s5 -= inv_r3 * (v.y - rv * r.y);                         // 4 FLOPs
    accjerk.s6 -= inv_r3 * (v.z - rv * r.z);                         // 4 FLOPs
    accjerk.s7  = 0;

    return accjerk;
}
// Total flop count: 42


#ifdef __OPENCL_VERSION__
//
// OpenCL implementation
////////////////////////////////////////////////////////////////////////////////
inline REAL8
p2p_accum_acc_jerk(REAL8 myAccJerk,
                   const REAL8 myData,
                   uint j_begin,
                   uint j_end,
                   __local REAL8 *sharedJData
                  )
{
    uint j;
    for (j = j_begin; j < j_end; ++j) {
        myAccJerk = p2p_acc_jerk_kernel_core(myAccJerk, myData.lo, myData.hi,
                                             sharedJData[j].lo, sharedJData[j].hi);
    }
    return myAccJerk;
}


inline REAL8
p2p_acc_jerk_kernel_main_loop(const REAL8 myData,
                              const uint nj,
                              __global const REAL8 *jdata,
                              __local REAL8 *sharedJData
                             )
{
    uint lsize = get_local_size(0);

    REAL8 myAccJerk = (REAL8){0, 0, 0, 0, 0, 0, 0, 0};

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
            myAccJerk = p2p_accum_acc_jerk(myAccJerk, myData,
                                           j, j + JUNROLL,
                                           sharedJData);
        }
        myAccJerk = p2p_accum_acc_jerk(myAccJerk, myData,
                                       j, nb,
                                       sharedJData);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return myAccJerk;
}


__kernel void p2p_acc_jerk_kernel(const uint ni,
                                  __global const REAL8 *idata,
                                  const uint nj,
                                  __global const REAL8 *jdata,
                                  __global REAL8 *iaccjerk,
//                                  __global REAL4 *iacc,
//                                  __global REAL4 *ijerk,
                                  __local REAL8 *sharedJData
                                 )
{
    uint gid = get_global_id(0);
    uint i = (gid < ni) ? (gid) : (ni-1);
    iaccjerk[i] = p2p_acc_jerk_kernel_main_loop(idata[i],
                                                nj, jdata,
                                                sharedJData);
//    REAL8 myAccJerk = p2p_acc_jerk_kernel_main_loop(idata[i],
//                                                    nj, jdata,
//                                                    sharedJData);
//    iacc[i] = myAccJerk.lo;
//    ijerk[i] = myAccJerk.hi;
}

#else
//
// C implementation
////////////////////////////////////////////////////////////////////////////////
static PyObject *
_p2p_acc_jerk_kernel(PyObject *_args)
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

    // main calculation
    unsigned int i, i8, j, j8;
    for (i = 0; i < ni; ++i) {
        i8 = 8*i;
        REAL8 iaccjerk = (REAL8){0, 0, 0, 0, 0, 0, 0, 0};
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
            iaccjerk = p2p_acc_jerk_kernel_core(iaccjerk, ri, vi, rj, vj);
        }
        ret_ptr[i8  ] = iaccjerk.s0;
        ret_ptr[i8+1] = iaccjerk.s1;
        ret_ptr[i8+2] = iaccjerk.s2;
        ret_ptr[i8+3] = iaccjerk.s3;
        ret_ptr[i8+4] = iaccjerk.s4;
        ret_ptr[i8+5] = iaccjerk.s5;
        ret_ptr[i8+6] = iaccjerk.s6;
        ret_ptr[i8+7] = iaccjerk.s7;
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


#endif  // P2P_ACC_JERK_KERNEL_H


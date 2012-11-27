#ifndef P2P_TSTEP_KERNEL_H
#define P2P_TSTEP_KERNEL_H

#include"common.h"
#include"smoothing.h"

//
// p2p_tstep_kernel_core
////////////////////////////////////////////////////////////////////////////////
inline REAL
p2p_tstep_kernel_core(REAL iomega,
                      const REAL4 rmi, const REAL4 vei,
                      const REAL4 rmj, const REAL4 vej,
                      const REAL eta)
{
    REAL4 r;
    r.x = rmi.x - rmj.x;                                             // 1 FLOPs
    r.y = rmi.y - rmj.y;                                             // 1 FLOPs
    r.z = rmi.z - rmj.z;                                             // 1 FLOPs
    r.w = rmi.w + rmj.w;                                             // 1 FLOPs
    REAL4 v;
    v.x = vei.x - vej.x;                                             // 1 FLOPs
    v.y = vei.y - vej.y;                                             // 1 FLOPs
    v.z = vei.z - vej.z;                                             // 1 FLOPs
    v.w = vei.w + vej.w;                                             // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs
    REAL v2 = v.x * v.x + v.y * v.y + v.z * v.z;                     // 5 FLOPs
    REAL rv = r.x * v.x + r.y * v.y + r.z * v.z;                     // 5 FLOPs

    REAL2 ret = smoothed_inv_r2r3(r2, v.w);                          // 4 FLOPs
    REAL inv_r2 = ret.x;
    REAL inv_r3 = ret.y;

    REAL alpha = v2 * inv_r2;                                        // 1 FLOPs
    REAL beta = 2 * r.w * inv_r3;                                    // 2 FLOPs
    REAL omega2 = alpha + beta + 1;                                      // 1 FLOPs
    REAL omega = sqrt(omega2);                                       // 1 FLOPs
    REAL gamma = 1 + (beta-1) / omega2;                                  // 2 FLOPs
    gamma = (r2 > 0) ? (gamma):(0);
    REAL dln_omega = -gamma * rv * inv_r2;                           // 2 FLOPs
    omega += eta * dln_omega;   // factor 1/2 included in 'eta'      // 2 FLOPs

//    iomega = (omega > iomega) ? (omega):(iomega);
    iomega += omega * omega;
    return iomega;
}
// Total flop count: 38


#ifdef __OPENCL_VERSION__
//
// OpenCL implementation
////////////////////////////////////////////////////////////////////////////////
inline REAL
p2p_accum_tstep(REAL iOmega,
                const REAL8 iData,
                const REAL eta,
                uint j_begin,
                uint j_end,
                __local REAL8 *sharedJData
               )
{
    uint j;
    for (j = j_begin; j < j_end; ++j) {
        REAL8 jData = sharedJData[j];
        iOmega = p2p_tstep_kernel_core(iOmega,
                                       iData.lo, iData.hi,
                                       jData.lo, jData.hi,
                                       eta);
    }
    return iOmega;
}


inline REAL
p2p_tstep_kernel_main_loop(const REAL8 iData,
                           const uint nj,
                           __global const REAL8 *jdata,
                           const REAL eta,
                           __local REAL8 *sharedJData
                          )
{
    uint lsize = get_local_size(0);

    REAL iOmega = (REAL)0;

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
            iOmega = p2p_accum_tstep(iOmega, iData,
                                     eta, j, j + JUNROLL,
                                     sharedJData);
        }
        iOmega = p2p_accum_tstep(iOmega, iData,
                                 eta, j, nb,
                                 sharedJData);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return iOmega;
}


__kernel void p2p_tstep_kernel(const uint ni,
                               __global const REAL8 *idata,
                               const uint nj,
                               __global const REAL8 *jdata,
                               const REAL eta,
                               __global REAL *itstep,
                               __local REAL8 *sharedJData
                              )
{
    uint gid = get_global_id(0);
    uint i = (gid < ni) ? (gid) : (ni-1);
    REAL iomega = p2p_tstep_kernel_main_loop(idata[i],
                                             nj, jdata,
                                             eta,
                                             sharedJData);
//    itstep[i] = 2 * eta / iomega;
    itstep[i] = 2 * eta / sqrt(iomega);
}

#else
//
// C implementation
////////////////////////////////////////////////////////////////////////////////
static PyObject *
_p2p_tstep_kernel(PyObject *_args)
{
    unsigned int ni, nj;
    REAL eta;
    PyObject *_idata = NULL;
    PyObject *_jdata = NULL;
    PyObject *_output = NULL;

    int typenum;
    char *fmt = NULL;
    if (sizeof(REAL) == sizeof(double)) {
        fmt = "IOIOdO!";
        typenum = NPY_FLOAT64;
    } else if (sizeof(REAL) == sizeof(float)) {
        fmt = "IOIOfO!";
        typenum = NPY_FLOAT32;
    }

    if (!PyArg_ParseTuple(_args, fmt, &ni, &_idata,
                                      &nj, &_jdata,
                                      &eta,
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
        REAL iomega = (REAL)0;
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
            iomega = p2p_tstep_kernel_core(iomega, rmi, vei, rmj, vej, eta);
        }
//        ret_ptr[i] = 2 * eta / iomega;
        ret_ptr[i] = 2 * eta / sqrt(iomega);
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


#endif  // P2P_TSTEP_KERNEL_H


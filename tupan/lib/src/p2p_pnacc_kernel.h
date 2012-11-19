#ifndef P2P_PNACC_KERNEL_H
#define P2P_PNACC_KERNEL_H

#include"common.h"
#include"smoothing.h"
#include"pn_terms.h"

//
// p2p_pnacc_kernel_core
////////////////////////////////////////////////////////////////////////////////
inline REAL3
p2p_pnacc_kernel_core(REAL3 pnacc,
                      const REAL4 rmi, const REAL4 vei,
                      const REAL4 rmj, const REAL4 vej,
                      const CLIGHT clight)
{
    REAL3 r;
    r.x = rmi.x - rmj.x;                                             // 1 FLOPs
    r.y = rmi.y - rmj.y;                                             // 1 FLOPs
    r.z = rmi.z - rmj.z;                                             // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs

    REAL3 v;
    v.x = vei.x - vej.x;                                             // 1 FLOPs
    v.y = vei.y - vej.y;                                             // 1 FLOPs
    v.z = vei.z - vej.z;                                             // 1 FLOPs
    REAL v2 = v.x * v.x + v.y * v.y + v.z * v.z;                     // 5 FLOPs

    REAL3 ret = smoothed_inv_r1r2r3(r2, vei.w + vej.w);
    REAL inv_r = ret.x;
    REAL inv_r2 = ret.y;
    REAL inv_r3 = ret.z;

    REAL mij = rmi.w + rmj.w;                                        // 1 FLOPs
    REAL r_sch = 2 * mij * clight.inv2;
    REAL gamma = r_sch * inv_r;

    if (16777216*gamma > 1) {
//    if (mij > 1.9) {
//        printf("mi: %e, mj: %e, mij: %e\n", rmi.w, rmj.w, mij);
        REAL3 vi = {vei.x, vei.y, vei.z};
        REAL3 vj = {vej.x, vej.y, vej.z};
        REAL2 pn = p2p_pnterms(rmi.w, rmj.w,
                               r, v, v2,
                               vi, vj,
                               inv_r, inv_r2, inv_r3,
                               clight);                              // ? FLOPs

        pnacc.x += pn.x * r.x + pn.y * v.x;                          // 4 FLOPs
        pnacc.y += pn.x * r.y + pn.y * v.y;                          // 4 FLOPs
        pnacc.z += pn.x * r.z + pn.y * v.z;                          // 4 FLOPs
    }

    return pnacc;
}
// Total flop count: 36+???


#ifdef __OPENCL_VERSION__
//
// OpenCL implementation
////////////////////////////////////////////////////////////////////////////////
inline REAL3
p2p_accum_pnacc(REAL3 iPNAcc,
                const REAL8 iData,
                const CLIGHT clight,
                uint j_begin,
                uint j_end,
                __local REAL8 *sharedJData
               )
{
    uint j;
    for (j = j_begin; j < j_end; ++j) {
        REAL8 jData = sharedJData[j];
        iPNAcc = p2p_pnacc_kernel_core(iPNAcc,
                                       iData.lo, iData.hi,
                                       jData.lo, jData.hi,
                                       clight);
    }
    return iPNAcc;
}


inline REAL3
p2p_pnacc_kernel_main_loop(const REAL8 iData,
                           const uint nj,
                           __global const REAL8 *jdata,
                           const CLIGHT clight,
                           __local REAL8 *sharedJData
                          )
{
    uint lsize = get_local_size(0);

    REAL3 iPNAcc = (REAL3){0, 0, 0};

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
            iPNAcc = p2p_accum_pnacc(iPNAcc, iData,
                                     clight, j, j + JUNROLL,
                                     sharedJData);
        }
        iPNAcc = p2p_accum_pnacc(iPNAcc, iData,
                                 clight, j, nb,
                                 sharedJData);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return iPNAcc;
}


__kernel void p2p_pnacc_kernel(const uint ni,
                               __global const REAL8 *idata,
                               const uint nj,
                               __global const REAL8 *jdata,
                               const uint order,
                               const REAL cinv1,
                               const REAL cinv2,
                               const REAL cinv3,
                               const REAL cinv4,
                               const REAL cinv5,
                               const REAL cinv6,
                               const REAL cinv7,
                               __global REAL3 *ipnacc,
                               __local REAL8 *sharedJData
                              )
{
    uint gid = get_global_id(0);
    uint i = (gid < ni) ? (gid) : (ni-1);
    const CLIGHT clight = (CLIGHT){cinv1, cinv2, cinv3,
                                   cinv4, cinv5, cinv6,
                                   cinv7, order};
    REAL8 iData = idata[i];
    ipnacc[i] = p2p_pnacc_kernel_main_loop(iData,
                                           nj, jdata,
                                           clight,
                                           sharedJData);
}

#else
//
// C implementation
////////////////////////////////////////////////////////////////////////////////
static PyObject *
_p2p_pnacc_kernel(PyObject *_args)
{
    unsigned int ni, nj;
    CLIGHT clight;
    PyObject *_idata = NULL;
    PyObject *_jdata = NULL;
    PyObject *_output = NULL;

    int typenum;
    char *fmt = NULL;
    if (sizeof(REAL) == sizeof(double)) {
        fmt = "IOIOIdddddddO!";
        typenum = NPY_FLOAT64;
    } else if (sizeof(REAL) == sizeof(float)) {
        fmt = "IOIOIfffffffO!";
        typenum = NPY_FLOAT32;
    }

    if (!PyArg_ParseTuple(_args, fmt, &ni, &_idata,
                                      &nj, &_jdata,
                                      &clight.order, &clight.inv1,
                                      &clight.inv2, &clight.inv3,
                                      &clight.inv4, &clight.inv5,
                                      &clight.inv6, &clight.inv7,
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
        REAL3 ipnacc = (REAL3){0, 0, 0};
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
            ipnacc = p2p_pnacc_kernel_core(ipnacc, rmi, vei, rmj, vej, clight);
        }
        ik = i * k;
        ret_ptr[ik  ] = ipnacc.x;
        ret_ptr[ik+1] = ipnacc.y;
        ret_ptr[ik+2] = ipnacc.z;
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


#endif  // P2P_PNACC_KERNEL_H


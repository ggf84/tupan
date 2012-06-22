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
                      const REAL4 ri, const REAL4 vi,
                      const REAL4 rj, const REAL4 vj,
                      const CLIGHT clight)
{
    REAL3 r;
    r.x = ri.x - rj.x;                                               // 1 FLOPs
    r.y = ri.y - rj.y;                                               // 1 FLOPs
    r.z = ri.z - rj.z;                                               // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs

    REAL mi = ri.w;
    REAL mj = rj.w;

    REAL3 v;
    v.x = vi.x - vj.x;                                               // 1 FLOPs
    v.y = vi.y - vj.y;                                               // 1 FLOPs
    v.z = vi.z - vj.z;                                               // 1 FLOPs
    REAL v2 = v.x * v.x + v.y * v.y + v.z * v.z;                     // 5 FLOPs

    REAL vi2 = vi.w;
    REAL vj2 = vj.w;

    REAL3 vixyz = {vi.x, vi.y, vi.z};
    REAL3 vjxyz = {vj.x, vj.y, vj.z};

    REAL inv_r2 = 1 / r2;                                            // 1 FLOPs
    inv_r2 = (r2 > 0) ? (inv_r2):(0);
    REAL inv_r = sqrt(inv_r2);                                       // 1 FLOPs

    REAL3 n;
    n.x = r.x * inv_r;                                               // 1 FLOPs
    n.y = r.y * inv_r;                                               // 1 FLOPs
    n.z = r.z * inv_r;                                               // 1 FLOPs

    REAL2 pn = p2p_pnterms(mi, mj,
                           inv_r, inv_r2,
                           n, v, v2,
                           vi2, vixyz,
                           vj2, vjxyz,
                           clight);                                  // ? FLOPs

    pnacc.x += pn.x * n.x + pn.y * v.x;                              // 4 FLOPs
    pnacc.y += pn.x * n.y + pn.y * v.y;                              // 4 FLOPs
    pnacc.z += pn.x * n.z + pn.y * v.z;                              // 4 FLOPs

    return pnacc;
}
// Total flop count: 36+???


#ifdef __OPENCL_VERSION__
//
// OpenCL implementation
////////////////////////////////////////////////////////////////////////////////
inline REAL3
p2p_accum_pnacc(REAL3 myPNAcc,
                const REAL8 myData,
                const CLIGHT clight,
                uint j_begin,
                uint j_end,
                __local REAL8 *sharedJData
               )
{
    uint j;
    for (j = j_begin; j < j_end; ++j) {
        REAL8 jData = sharedJData[j];
        jData.s7 = jData.s4 * jData.s4 + jData.s5 * jData.s5 + jData.s6 * jData.s6;
        myPNAcc = p2p_pnacc_kernel_core(myPNAcc, myData.lo, myData.hi,
                                        jData.lo, jData.hi,
                                        clight);
    }
    return myPNAcc;
}


inline REAL4
p2p_pnacc_kernel_main_loop(const REAL8 myData,
                           const uint nj,
                           __global const REAL8 *jdata,
                           const CLIGHT clight,
                           __local REAL8 *sharedJData
                          )
{
    uint lsize = get_local_size(0);

    REAL3 myPNAcc = (REAL3){0, 0, 0};

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
            myPNAcc = p2p_accum_pnacc(myPNAcc, myData,
                                      clight, j, j + JUNROLL,
                                      sharedJData);
        }
        myPNAcc = p2p_accum_pnacc(myPNAcc, myData,
                                  clight, j, nb,
                                  sharedJData);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return (REAL4){myPNAcc.x, myPNAcc.y, myPNAcc.z, 0};
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
                               __global REAL4 *ipnacc,  // XXX: Bug!!! if we use __global REAL3
                               __local REAL8 *sharedJData
                              )
{
    uint gid = get_global_id(0);
    uint i = (gid < ni) ? (gid) : (ni-1);
    const CLIGHT clight = (CLIGHT){cinv1, cinv2, cinv3,
                                   cinv4, cinv5, cinv6,
                                   cinv7, order};
    REAL8 myData = idata[i];
    myData.s7 = myData.s4 * myData.s4 + myData.s5 * myData.s5 + myData.s6 * myData.s6;
    ipnacc[i] = p2p_pnacc_kernel_main_loop(myData,
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

    int typenum;
    char *fmt = NULL;
    if (sizeof(REAL) == sizeof(double)) {
        fmt = "IOIOIddddddd";
        typenum = NPY_FLOAT64;
    } else if (sizeof(REAL) == sizeof(float)) {
        fmt = "IOIOIfffffff";
        typenum = NPY_FLOAT32;
    }

    if (!PyArg_ParseTuple(_args, fmt, &ni, &_idata,
                                      &nj, &_jdata,
                                      &clight.order, &clight.inv1,
                                      &clight.inv2, &clight.inv3,
                                      &clight.inv4, &clight.inv5,
                                      &clight.inv6, &clight.inv7))
        return NULL;

    // i-data
    PyObject *_idata_arr = PyArray_FROM_OTF(_idata, typenum, NPY_IN_ARRAY);
    REAL *idata_ptr = (REAL *)PyArray_DATA(_idata_arr);

    // j-data
    PyObject *_jdata_arr = PyArray_FROM_OTF(_jdata, typenum, NPY_IN_ARRAY);
    REAL *jdata_ptr = (REAL *)PyArray_DATA(_jdata_arr);

    // allocate a PyArrayObject to be returned
    npy_intp dims[2] = {ni, 3};
    PyArrayObject *ret = (PyArrayObject *)PyArray_EMPTY(2, dims, typenum, 0);
    REAL *ret_ptr = (REAL *)PyArray_DATA(ret);

    // main calculation
    unsigned int i, i3, i8, j, j8;
    for (i = 0; i < ni; ++i) {
        i8 = 8*i;
        REAL3 ipnacc = (REAL3){0, 0, 0};
        REAL4 ri = {idata_ptr[i8  ], idata_ptr[i8+1],
                    idata_ptr[i8+2], idata_ptr[i8+3]};
        REAL4 vi = {idata_ptr[i8+4], idata_ptr[i8+5],
                    idata_ptr[i8+6], idata_ptr[i8+7]};
        vi.w = vi.x * vi.x + vi.y * vi.y + vi.z * vi.z;
        for (j = 0; j < nj; ++j) {
            j8 = 8*j;
            REAL4 rj = {jdata_ptr[j8  ], jdata_ptr[j8+1],
                        jdata_ptr[j8+2], jdata_ptr[j8+3]};
            REAL4 vj = {jdata_ptr[j8+4], jdata_ptr[j8+5],
                        jdata_ptr[j8+6], jdata_ptr[j8+7]};
            vj.w = vj.x * vj.x + vj.y * vj.y + vj.z * vj.z;
            ipnacc = p2p_pnacc_kernel_core(ipnacc, ri, vi, rj, vj, clight);
        }
        i3 = 3*i;
        ret_ptr[i3  ] = ipnacc.x;
        ret_ptr[i3+1] = ipnacc.y;
        ret_ptr[i3+2] = ipnacc.z;
    }

    // Decrement the reference counts for i-objects
    Py_DECREF(_idata_arr);

    // Decrement the reference counts for j-objects
    Py_DECREF(_jdata_arr);

    // returns a PyArrayObject
    return PyArray_Return(ret);
}

#endif  // __OPENCL_VERSION__


#endif  // P2P_PNACC_KERNEL_H


#ifndef P2P_TSTEP_KERNEL_H
#define P2P_TSTEP_KERNEL_H

#include"common.h"
#include"smoothing.h"

//
// p2p_tstep_kernel_core
////////////////////////////////////////////////////////////////////////////////
inline REAL
p2p_tstep_kernel_core(REAL iomega,
                      const REAL4 ri, const REAL4 vi,
                      const REAL4 rj, const REAL4 vj,
                      const REAL eta)
{
    REAL4 r;
    r.x = ri.x - rj.x;                                               // 1 FLOPs
    r.y = ri.y - rj.y;                                               // 1 FLOPs
    r.z = ri.z - rj.z;                                               // 1 FLOPs
    r.w = ri.w + rj.w;                                               // 1 FLOPs
    REAL4 v;
    v.x = vi.x - vj.x;                                               // 1 FLOPs
    v.y = vi.y - vj.y;                                               // 1 FLOPs
    v.z = vi.z - vj.z;                                               // 1 FLOPs
    v.w = vi.w + vj.w;                                               // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs
    REAL v2 = v.x * v.x + v.y * v.y + v.z * v.z;                     // 5 FLOPs
    REAL rv = r.x * v.x + r.y * v.y + r.z * v.z;                     // 5 FLOPs
    REAL inv_r2 = 1 / (r2 + v.w);                                    // 2 FLOPs
    inv_r2 = (r2 > 0) ? (inv_r2):(0);
    REAL inv_r = sqrt(inv_r2);                                       // 1 FLOPs
/*    REAL inv_r3 = inv_r * inv_r2;                                    // 1 FLOPs

    REAL omega2a = v2 * inv_r2;                                      // 1 FLOPs
    REAL omega2b = 2 * r.w * inv_r3;                                 // 2 FLOPs
    REAL omega2 = omega2a + omega2b;                                 // 1 FLOPs
    REAL omega2b_omega2 = omega2b / omega2;                          // 1 FLOPs
    omega2b_omega2 = (r2 > 0) ? (omega2b_omega2):(0);
    REAL weighting = 1 + omega2b_omega2;                             // 1 FLOPs
    REAL dln_omega = -weighting * rv * inv_r2;                       // 2 FLOPs
    REAL omega = sqrt(omega2);                                       // 1 FLOPs
    omega += eta * dln_omega;   // factor 1/2 included in 'eta'      // 2 FLOPs
*/

    REAL4 h;
    h.x = r.y * v.z - r.z * v.y;                                     // 3 FLOPs
    h.y = r.z * v.x - r.x * v.z;                                     // 3 FLOPs
    h.z = r.x * v.y - r.y * v.x;                                     // 3 FLOPs
    REAL h2 = h.x * h.x + h.y * h.y + h.z * h.z;                     // 5 FLOPs
    REAL e = (v2 - 2 * r.w * inv_r);                                 // 3 FLOPs

    REAL omega2_e = (e > 0) ? (e):(-e);
    REAL omega2_h = ((REAL)1.5) * h2 * inv_r2;                       // 2 FLOPs
    REAL omega2 = (omega2_e + omega2_h) * inv_r2;                    // 2 FLOPs
    REAL omega = sqrt(omega2);                                       // 1 FLOPs
    REAL weight = (1 + inv_r2 * omega2_h / omega2);                  // 3 FLOPs
    REAL dln_omega = -weight * (rv * inv_r2);                        // 2 FLOPs
    omega += eta * dln_omega;   // factor 1/2 included in 'eta'      // 2 FLOPs

    iomega = (omega > iomega) ? (omega):(iomega);
    return iomega;
}
// Total flop count: 38


#ifdef __OPENCL_VERSION__
//
// OpenCL implementation
////////////////////////////////////////////////////////////////////////////////
inline REAL
p2p_accum_tstep(REAL myOmega,
                const REAL8 myData,
                const REAL eta,
                uint j_begin,
                uint j_end,
                __local REAL8 *sharedJData
               )
{
    uint j;
    for (j = j_begin; j < j_end; ++j) {
        myOmega = p2p_tstep_kernel_core(myOmega, myData.lo, myData.hi,
                                        sharedJData[j].lo, sharedJData[j].hi,
                                        eta);
    }
    return myOmega;
}


inline REAL
p2p_tstep_kernel_main_loop(const REAL8 myData,
                           const uint nj,
                           __global const REAL8 *jdata,
                           const REAL eta,
                           __local REAL8 *sharedJData
                          )
{
    uint lsize = get_local_size(0);

    REAL myOmega = (REAL)0;

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
            myOmega = p2p_accum_tstep(myOmega, myData,
                                      eta, j, j + JUNROLL,
                                      sharedJData);
        }
        myOmega = p2p_accum_tstep(myOmega, myData,
                                  eta, j, nb,
                                  sharedJData);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return myOmega;
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
    itstep[i] = 2 * eta / iomega;
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

    int typenum;
    char *fmt = NULL;
    if (sizeof(REAL) == sizeof(double)) {
        fmt = "IOIOd";
        typenum = NPY_FLOAT64;
    } else if (sizeof(REAL) == sizeof(float)) {
        fmt = "IOIOf";
        typenum = NPY_FLOAT32;
    }

    if (!PyArg_ParseTuple(_args, fmt, &ni, &_idata,
                                      &nj, &_jdata,
                                      &eta))
        return NULL;

    // i-data
    PyObject *_idata_arr = PyArray_FROM_OTF(_idata, typenum, NPY_IN_ARRAY);
    REAL *idata_ptr = (REAL *)PyArray_DATA(_idata_arr);

    // j-data
    PyObject *_jdata_arr = PyArray_FROM_OTF(_jdata, typenum, NPY_IN_ARRAY);
    REAL *jdata_ptr = (REAL *)PyArray_DATA(_jdata_arr);

    // allocate a PyArrayObject to be returned
    npy_intp dims[1] = {ni};
    PyArrayObject *ret = (PyArrayObject *)PyArray_EMPTY(1, dims, typenum, 0);
    REAL *ret_ptr = (REAL *)PyArray_DATA(ret);

    // main calculation
    unsigned int i, i8, j, j8;
    for (i = 0; i < ni; ++i) {
        i8 = 8*i;
        REAL iomega = (REAL)0;
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
            iomega = p2p_tstep_kernel_core(iomega, ri, vi, rj, vj, eta);
        }
        ret_ptr[i] = 2 * eta / iomega;
    }

    // Decrement the reference counts for i-objects
    Py_DECREF(_idata_arr);

    // Decrement the reference counts for j-objects
    Py_DECREF(_jdata_arr);

    // returns a PyArrayObject
    return PyArray_Return(ret);
}

#endif  // __OPENCL_VERSION__


#endif  // P2P_TSTEP_KERNEL_H


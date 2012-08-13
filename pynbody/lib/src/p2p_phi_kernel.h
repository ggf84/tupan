#ifndef P2P_PHI_KERNEL_H
#define P2P_PHI_KERNEL_H

#include"common.h"
#include"smoothing.h"

//
// p2p_phi_kernel_core
////////////////////////////////////////////////////////////////////////////////
inline REAL
p2p_phi_kernel_core(REAL phi,
                    const REAL4 ri, const REAL hi2,
                    const REAL4 rj, const REAL hj2)
{
    REAL4 r;
    r.x = ri.x - rj.x;                                               // 1 FLOPs
    r.y = ri.y - rj.y;                                               // 1 FLOPs
    r.z = ri.z - rj.z;                                               // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs
    REAL inv_r = phi_smooth(r2, hi2 + hj2);                          // 4 FLOPs
    phi -= rj.w * inv_r;                                             // 2 FLOPs
    return phi;
}
// Total flop count: 14


#ifdef __OPENCL_VERSION__
//
// OpenCL implementation
////////////////////////////////////////////////////////////////////////////////
inline REAL
p2p_accum_phi(REAL myPhi,
              const REAL8 myData,
              uint j_begin,
              uint j_end,
              __local REAL8 *sharedJData
             )
{
    uint j;
    for (j = j_begin; j < j_end; ++j) {
        myPhi = p2p_phi_kernel_core(myPhi, myData.lo, myData.s7,
                                    sharedJData[j].lo, sharedJData[j].s7);
    }
    return myPhi;
}


inline REAL
p2p_phi_kernel_main_loop(const REAL8 myData,
                         const uint nj,
                         __global const REAL8 *jdata,
                         __local REAL8 *sharedJData
                        )
{
    uint lsize = get_local_size(0);

    REAL myPhi = (REAL)0;

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
            myPhi = p2p_accum_phi(myPhi, myData,
                                  j, j + JUNROLL,
                                  sharedJData);
        }
        myPhi = p2p_accum_phi(myPhi, myData,
                              j, nb,
                              sharedJData);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return myPhi;
}


__kernel void p2p_phi_kernel(const uint ni,
                             __global const REAL8 *idata,
                             const uint nj,
                             __global const REAL8 *jdata,
                             __global REAL *iphi,
                             __local REAL8 *sharedJData
                            )
{
    uint gid = get_global_id(0);
    uint i = (gid < ni) ? (gid) : (ni-1);
    iphi[i] = p2p_phi_kernel_main_loop(idata[i],
                                       nj, jdata,
                                       sharedJData);
}

#else
//
// C implementation
////////////////////////////////////////////////////////////////////////////////
static PyObject *
_p2p_phi_kernel(PyObject *_args)
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
        REAL iphi = (REAL)0;
        REAL4 ri = {idata_ptr[i8  ], idata_ptr[i8+1],
                    idata_ptr[i8+2], idata_ptr[i8+3]};
        REAL ieps2 = idata_ptr[i8+7];
        for (j = 0; j < nj; ++j) {
            j8 = 8*j;
            REAL4 rj = {jdata_ptr[j8  ], jdata_ptr[j8+1],
                        jdata_ptr[j8+2], jdata_ptr[j8+3]};
            REAL jeps2 = jdata_ptr[j8+7];
            iphi = p2p_phi_kernel_core(iphi, ri, ieps2, rj, jeps2);
        }
        ret_ptr[i] = iphi;
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


#endif  // P2P_PHI_KERNEL_H


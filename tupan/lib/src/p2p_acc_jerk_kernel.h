#ifndef P2P_ACC_JERK_KERNEL_H
#define P2P_ACC_JERK_KERNEL_H

#include"common.h"
#include"smoothing.h"

//
// p2p_acc_jerk_kernel_core
////////////////////////////////////////////////////////////////////////////////
inline REAL8
p2p_acc_jerk_kernel_core(REAL8 iaj,
                         const REAL4 irm, const REAL4 ive,
                         const REAL4 jrm, const REAL4 jve)
{
    REAL3 r;
    r.x = irm.x - jrm.x;                                             // 1 FLOPs
    r.y = irm.y - jrm.y;                                             // 1 FLOPs
    r.z = irm.z - jrm.z;                                             // 1 FLOPs
    REAL4 v;
    v.x = ive.x - jve.x;                                             // 1 FLOPs
    v.y = ive.y - jve.y;                                             // 1 FLOPs
    v.z = ive.z - jve.z;                                             // 1 FLOPs
    v.w = ive.w + jve.w;                                             // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs
    REAL rv = r.x * v.x + r.y * v.y + r.z * v.z;                     // 5 FLOPs
    REAL2 ret = smoothed_inv_r2r3(r2, v.w);                          // 5 FLOPs
    REAL inv_r2 = ret.x;
    REAL inv_r3 = ret.y;

    inv_r3 *= jrm.w;                                                 // 1 FLOPs
    rv *= 3 * inv_r2;                                                // 2 FLOPs

    iaj.s0 -= inv_r3 * r.x;                                          // 2 FLOPs
    iaj.s1 -= inv_r3 * r.y;                                          // 2 FLOPs
    iaj.s2 -= inv_r3 * r.z;                                          // 2 FLOPs
    iaj.s3  = 0;
    iaj.s4 -= inv_r3 * (v.x - rv * r.x);                             // 4 FLOPs
    iaj.s5 -= inv_r3 * (v.y - rv * r.y);                             // 4 FLOPs
    iaj.s6 -= inv_r3 * (v.z - rv * r.z);                             // 4 FLOPs
    iaj.s7  = 0;
    return iaj;
}
// Total flop count: 43


#ifdef __OPENCL_VERSION__
//
// OpenCL implementation
////////////////////////////////////////////////////////////////////////////////
inline REAL8
p2p_accum_acc_jerk(REAL8 iaj,
                   const REAL8 idata,
                   uint j_begin,
                   uint j_end,
                   __local REAL *shr_jrx,
                   __local REAL *shr_jry,
                   __local REAL *shr_jrz,
                   __local REAL *shr_jmass,
                   __local REAL *shr_jvx,
                   __local REAL *shr_jvy,
                   __local REAL *shr_jvz,
                   __local REAL *shr_jeps2
                  )
{
    uint j;
    for (j = j_begin; j < j_end; ++j) {
        REAL8 jdata = (REAL8){shr_jrx[j], shr_jry[j], shr_jrz[j], shr_jmass[j],
                              shr_jvx[j], shr_jvy[j], shr_jvz[j], shr_jeps2[j]};
        iaj = p2p_acc_jerk_kernel_core(iaj,
                                       idata.lo, idata.hi,
                                       jdata.lo, jdata.hi);
    }
    return iaj;
}


inline REAL8
p2p_acc_jerk_kernel_main_loop(const REAL8 idata,
                              const uint nj,
                              __global const REAL *inp_jrx,
                              __global const REAL *inp_jry,
                              __global const REAL *inp_jrz,
                              __global const REAL *inp_jmass,
                              __global const REAL *inp_jvx,
                              __global const REAL *inp_jvy,
                              __global const REAL *inp_jvz,
                              __global const REAL *inp_jeps2,
                              __local REAL *shr_jrx,
                              __local REAL *shr_jry,
                              __local REAL *shr_jrz,
                              __local REAL *shr_jmass,
                              __local REAL *shr_jvx,
                              __local REAL *shr_jvy,
                              __local REAL *shr_jvz,
                              __local REAL *shr_jeps2
                             )
{
    uint lsize = get_local_size(0);

    REAL8 iaj = (REAL8){0, 0, 0, 0, 0, 0, 0, 0};

    uint tile;
    uint numTiles = (nj - 1)/lsize + 1;
    for (tile = 0; tile < numTiles; ++tile) {
        uint nb = min(lsize, (nj - (tile * lsize)));

        event_t e[8];
        e[0] = async_work_group_copy(shr_jrx, inp_jrx + tile * lsize, nb, 0);
        e[1] = async_work_group_copy(shr_jry, inp_jry + tile * lsize, nb, 0);
        e[2] = async_work_group_copy(shr_jrz, inp_jrz + tile * lsize, nb, 0);
        e[3] = async_work_group_copy(shr_jmass, inp_jmass + tile * lsize, nb, 0);
        e[4] = async_work_group_copy(shr_jvx, inp_jvx + tile * lsize, nb, 0);
        e[5] = async_work_group_copy(shr_jvy, inp_jvy + tile * lsize, nb, 0);
        e[6] = async_work_group_copy(shr_jvz, inp_jvz + tile * lsize, nb, 0);
        e[7] = async_work_group_copy(shr_jeps2, inp_jeps2 + tile * lsize, nb, 0);
        wait_group_events(8, e);

        uint j = 0;
        uint j_max = (nb > (JUNROLL - 1)) ? (nb - (JUNROLL - 1)):(0);
        for (; j < j_max; j += JUNROLL) {
            iaj = p2p_accum_acc_jerk(iaj, idata,
                                     j, j + JUNROLL,
                                     shr_jrx, shr_jry, shr_jrz, shr_jmass,
                                     shr_jvx, shr_jvy, shr_jvz, shr_jeps2);
        }
        iaj = p2p_accum_acc_jerk(iaj, idata,
                                 j, nb,
                                 shr_jrx, shr_jry, shr_jrz, shr_jmass,
                                 shr_jvx, shr_jvy, shr_jvz, shr_jeps2);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return iaj;
}


__kernel void p2p_acc_jerk_kernel(const uint ni,
                                  __global const REAL *inp_irx,
                                  __global const REAL *inp_iry,
                                  __global const REAL *inp_irz,
                                  __global const REAL *inp_imass,
                                  __global const REAL *inp_ivx,
                                  __global const REAL *inp_ivy,
                                  __global const REAL *inp_ivz,
                                  __global const REAL *inp_ieps2,
                                  const uint nj,
                                  __global const REAL *inp_jrx,
                                  __global const REAL *inp_jry,
                                  __global const REAL *inp_jrz,
                                  __global const REAL *inp_jmass,
                                  __global const REAL *inp_jvx,
                                  __global const REAL *inp_jvy,
                                  __global const REAL *inp_jvz,
                                  __global const REAL *inp_jeps2,
                                  __global REAL *out_iax,
                                  __global REAL *out_iay,
                                  __global REAL *out_iaz,
                                  __global REAL *out_ijx,
                                  __global REAL *out_ijy,
                                  __global REAL *out_ijz,
                                  __local REAL *shr_jrx,
                                  __local REAL *shr_jry,
                                  __local REAL *shr_jrz,
                                  __local REAL *shr_jmass,
                                  __local REAL *shr_jvx,
                                  __local REAL *shr_jvy,
                                  __local REAL *shr_jvz,
                                  __local REAL *shr_jeps2
                                 )
{
    uint gid = get_global_id(0);
    uint i = (gid < ni) ? (gid) : (ni-1);

    REAL8 idata = (REAL8){inp_irx[i], inp_iry[i], inp_irz[i], inp_imass[i],
                          inp_ivx[i], inp_ivy[i], inp_ivz[i], inp_ieps2[i]};

    REAL8 iaj = p2p_acc_jerk_kernel_main_loop(idata,
                                              nj,
                                              inp_jrx, inp_jry, inp_jrz, inp_jmass,
                                              inp_jvx, inp_jvy, inp_jvz, inp_jeps2,
                                              shr_jrx, shr_jry, shr_jrz, shr_jmass,
                                              shr_jvx, shr_jvy, shr_jvz, shr_jeps2);
    out_iax[i] = iaj.s0;
    out_iay[i] = iaj.s1;
    out_iaz[i] = iaj.s2;
    out_ijx[i] = iaj.s4;
    out_ijy[i] = iaj.s5;
    out_ijz[i] = iaj.s6;
}

#else
//
// C implementation
////////////////////////////////////////////////////////////////////////////////

inline void
main_p2p_acc_jerk_kernel(const unsigned int ni,
                         const REAL *irx, const REAL *iry, const REAL *irz, const REAL *imass,
                         const REAL *ivx, const REAL *ivy, const REAL *ivz, const REAL *ieps2,
                         const unsigned int nj,
                         const REAL *jrx, const REAL *jry, const REAL *jrz, const REAL *jmass,
                         const REAL *jvx, const REAL *jvy, const REAL *jvz, const REAL *jeps2,
                         REAL *iax, REAL *iay, REAL *iaz,
                         REAL *ijx, REAL *ijy, REAL *ijz)
{
    unsigned int i, j;
    for (i = 0; i < ni; ++i) {
        REAL4 irm = {irx[i], iry[i], irz[i], imass[i]};
        REAL4 ive = {ivx[i], ivy[i], ivz[i], ieps2[i]};
        REAL8 iaj = (REAL8){0, 0, 0, 0, 0, 0, 0, 0};
        for (j = 0; j < nj; ++j) {
            REAL4 jrm = {jrx[j], jry[j], jrz[j], jmass[j]};
            REAL4 jve = {jvx[j], jvy[j], jvz[j], jeps2[j]};
            iaj = p2p_acc_jerk_kernel_core(iaj, irm, ive, jrm, jve);
        }
        iax[i] = iaj.s0;
        iay[i] = iaj.s1;
        iaz[i] = iaj.s2;
        ijx[i] = iaj.s4;
        ijy[i] = iaj.s5;
        ijz[i] = iaj.s6;
    }
}


#ifndef USE_CTYPES
static PyObject *
p2p_acc_jerk_kernel(PyObject *_self, PyObject *_args)
{
    unsigned int ni, nj;
    // i-objs
    PyObject *inp_irx = NULL;
    PyObject *inp_iry = NULL;
    PyObject *inp_irz = NULL;
    PyObject *inp_imass = NULL;
    PyObject *inp_ivx = NULL;
    PyObject *inp_ivy = NULL;
    PyObject *inp_ivz = NULL;
    PyObject *inp_ieps2 = NULL;
    // j-objs
    PyObject *inp_jrx = NULL;
    PyObject *inp_jry = NULL;
    PyObject *inp_jrz = NULL;
    PyObject *inp_jmass = NULL;
    PyObject *inp_jvx = NULL;
    PyObject *inp_jvy = NULL;
    PyObject *inp_jvz = NULL;
    PyObject *inp_jeps2 = NULL;
    // out-objs
    PyObject *out_iax = NULL;
    PyObject *out_iay = NULL;
    PyObject *out_iaz = NULL;
    PyObject *out_ijx = NULL;
    PyObject *out_ijy = NULL;
    PyObject *out_ijz = NULL;

    int typenum;
    char *fmt = NULL;
    if (sizeof(REAL) == sizeof(double)) {
        fmt = "IOOOOOOOOIOOOOOOOOO!O!O!O!O!O!";
        typenum = NPY_FLOAT64;
    } else if (sizeof(REAL) == sizeof(float)) {
        fmt = "IOOOOOOOOIOOOOOOOOO!O!O!O!O!O!";
        typenum = NPY_FLOAT32;
    }

    if (!PyArg_ParseTuple(_args, fmt, &ni,
                                      &inp_irx, &inp_iry, &inp_irz, &inp_imass,
                                      &inp_ivx, &inp_ivy, &inp_ivz, &inp_ieps2,
                                      &nj,
                                      &inp_jrx, &inp_jry, &inp_jrz, &inp_jmass,
                                      &inp_jvx, &inp_jvy, &inp_jvz, &inp_jeps2,
                                      &PyArray_Type, &out_iax,
                                      &PyArray_Type, &out_iay,
                                      &PyArray_Type, &out_iaz,
                                      &PyArray_Type, &out_ijx,
                                      &PyArray_Type, &out_ijy,
                                      &PyArray_Type, &out_ijz))
        return NULL;

    // i-objs
    PyObject *irx = PyArray_FROM_OTF(inp_irx, typenum, NPY_IN_ARRAY);
    REAL *irx_ptr = (REAL *)PyArray_DATA(irx);

    PyObject *iry = PyArray_FROM_OTF(inp_iry, typenum, NPY_IN_ARRAY);
    REAL *iry_ptr = (REAL *)PyArray_DATA(iry);

    PyObject *irz = PyArray_FROM_OTF(inp_irz, typenum, NPY_IN_ARRAY);
    REAL *irz_ptr = (REAL *)PyArray_DATA(irz);

    PyObject *imass = PyArray_FROM_OTF(inp_imass, typenum, NPY_IN_ARRAY);
    REAL *imass_ptr = (REAL *)PyArray_DATA(imass);

    PyObject *ivx = PyArray_FROM_OTF(inp_ivx, typenum, NPY_IN_ARRAY);
    REAL *ivx_ptr = (REAL *)PyArray_DATA(ivx);

    PyObject *ivy = PyArray_FROM_OTF(inp_ivy, typenum, NPY_IN_ARRAY);
    REAL *ivy_ptr = (REAL *)PyArray_DATA(ivy);

    PyObject *ivz = PyArray_FROM_OTF(inp_ivz, typenum, NPY_IN_ARRAY);
    REAL *ivz_ptr = (REAL *)PyArray_DATA(ivz);

    PyObject *ieps2 = PyArray_FROM_OTF(inp_ieps2, typenum, NPY_IN_ARRAY);
    REAL *ieps2_ptr = (REAL *)PyArray_DATA(ieps2);

    // j-objs
    PyObject *jrx = PyArray_FROM_OTF(inp_jrx, typenum, NPY_IN_ARRAY);
    REAL *jrx_ptr = (REAL *)PyArray_DATA(jrx);

    PyObject *jry = PyArray_FROM_OTF(inp_jry, typenum, NPY_IN_ARRAY);
    REAL *jry_ptr = (REAL *)PyArray_DATA(jry);

    PyObject *jrz = PyArray_FROM_OTF(inp_jrz, typenum, NPY_IN_ARRAY);
    REAL *jrz_ptr = (REAL *)PyArray_DATA(jrz);

    PyObject *jmass = PyArray_FROM_OTF(inp_jmass, typenum, NPY_IN_ARRAY);
    REAL *jmass_ptr = (REAL *)PyArray_DATA(jmass);

    PyObject *jvx = PyArray_FROM_OTF(inp_jvx, typenum, NPY_IN_ARRAY);
    REAL *jvx_ptr = (REAL *)PyArray_DATA(jvx);

    PyObject *jvy = PyArray_FROM_OTF(inp_jvy, typenum, NPY_IN_ARRAY);
    REAL *jvy_ptr = (REAL *)PyArray_DATA(jvy);

    PyObject *jvz = PyArray_FROM_OTF(inp_jvz, typenum, NPY_IN_ARRAY);
    REAL *jvz_ptr = (REAL *)PyArray_DATA(jvz);

    PyObject *jeps2 = PyArray_FROM_OTF(inp_jeps2, typenum, NPY_IN_ARRAY);
    REAL *jeps2_ptr = (REAL *)PyArray_DATA(jeps2);

    // out-objs
    PyObject *iax = PyArray_FROM_OTF(out_iax, typenum, NPY_INOUT_ARRAY);
    REAL *iax_ptr = (REAL *)PyArray_DATA(iax);

    PyObject *iay = PyArray_FROM_OTF(out_iay, typenum, NPY_INOUT_ARRAY);
    REAL *iay_ptr = (REAL *)PyArray_DATA(iay);

    PyObject *iaz = PyArray_FROM_OTF(out_iaz, typenum, NPY_INOUT_ARRAY);
    REAL *iaz_ptr = (REAL *)PyArray_DATA(iaz);

    PyObject *ijx = PyArray_FROM_OTF(out_ijx, typenum, NPY_INOUT_ARRAY);
    REAL *ijx_ptr = (REAL *)PyArray_DATA(ijx);

    PyObject *ijy = PyArray_FROM_OTF(out_ijy, typenum, NPY_INOUT_ARRAY);
    REAL *ijy_ptr = (REAL *)PyArray_DATA(ijy);

    PyObject *ijz = PyArray_FROM_OTF(out_ijz, typenum, NPY_INOUT_ARRAY);
    REAL *ijz_ptr = (REAL *)PyArray_DATA(ijz);

    // main calculation
    main_p2p_acc_jerk_kernel(ni,
                             irx_ptr, iry_ptr, irz_ptr, imass_ptr,
                             ivx_ptr, ivy_ptr, ivz_ptr, ieps2_ptr,
                             nj,
                             jrx_ptr, jry_ptr, jrz_ptr, jmass_ptr,
                             jvx_ptr, jvy_ptr, jvz_ptr, jeps2_ptr,
                             iax_ptr, iay_ptr, iaz_ptr,
                             ijx_ptr, ijy_ptr, ijz_ptr);

    // Decrement the reference counts for auxiliary i-objs
    Py_DECREF(irx);
    Py_DECREF(iry);
    Py_DECREF(irz);
    Py_DECREF(imass);
    Py_DECREF(ivx);
    Py_DECREF(ivy);
    Py_DECREF(ivz);
    Py_DECREF(ieps2);

    // Decrement the reference counts for auxiliary j-objs
    Py_DECREF(jrx);
    Py_DECREF(jry);
    Py_DECREF(jrz);
    Py_DECREF(jmass);
    Py_DECREF(jvx);
    Py_DECREF(jvy);
    Py_DECREF(jvz);
    Py_DECREF(jeps2);

    // Decrement the reference counts for auxiliary out-objs
    Py_DECREF(iax);
    Py_DECREF(iay);
    Py_DECREF(iaz);
    Py_DECREF(ijx);
    Py_DECREF(ijy);
    Py_DECREF(ijz);

    // Returns None
    Py_INCREF(Py_None);
    return Py_None;
}
#endif  // USE_CTYPES

#endif  // __OPENCL_VERSION__


#endif  // P2P_ACC_JERK_KERNEL_H


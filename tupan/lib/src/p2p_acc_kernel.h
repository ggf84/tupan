#ifndef P2P_ACC_KERNEL_H
#define P2P_ACC_KERNEL_H

#include"common.h"
#include"smoothing.h"

//
// p2p_acc_kernel_core
////////////////////////////////////////////////////////////////////////////////
inline REAL3
p2p_acc_kernel_core(REAL3 ia,
                    const REAL4 irm, const REAL4 ive,
                    const REAL4 jrm, const REAL4 jve)
{
    REAL3 r;
    r.x = irm.x - jrm.x;                                             // 1 FLOPs
    r.y = irm.y - jrm.y;                                             // 1 FLOPs
    r.z = irm.z - jrm.z;                                             // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs
    REAL inv_r3 = smoothed_inv_r3(r2, ive.w + jve.w);                // 5+1 FLOPs

    inv_r3 *= jrm.w;                                                 // 1 FLOPs

    ia.x -= inv_r3 * r.x;                                            // 2 FLOPs
    ia.y -= inv_r3 * r.y;                                            // 2 FLOPs
    ia.z -= inv_r3 * r.z;                                            // 2 FLOPs
    return ia;
}
// Total flop count: 21


#ifdef __OPENCL_VERSION__
//
// OpenCL implementation
////////////////////////////////////////////////////////////////////////////////
inline REAL3
p2p_accum_acc(REAL3 ia,
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
        ia = p2p_acc_kernel_core(ia,
                                 idata.lo, idata.hi,
                                 jdata.lo, jdata.hi);
    }
    return ia;
}


inline REAL3
p2p_acc_kernel_main_loop(const REAL8 idata,
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

    REAL3 ia = (REAL3){0, 0, 0};

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
            ia = p2p_accum_acc(ia, idata,
                               j, j + JUNROLL,
                               shr_jrx, shr_jry, shr_jrz, shr_jmass,
                               shr_jvx, shr_jvy, shr_jvz, shr_jeps2);
        }
        ia = p2p_accum_acc(ia, idata,
                           j, nb,
                           shr_jrx, shr_jry, shr_jrz, shr_jmass,
                           shr_jvx, shr_jvy, shr_jvz, shr_jeps2);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return ia;
}


__kernel void p2p_acc_kernel(const uint ni,
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

    REAL3 ia = p2p_acc_kernel_main_loop(idata,
                                        nj,
                                        inp_jrx, inp_jry, inp_jrz, inp_jmass,
                                        inp_jvx, inp_jvy, inp_jvz, inp_jeps2,
                                        shr_jrx, shr_jry, shr_jrz, shr_jmass,
                                        shr_jvx, shr_jvy, shr_jvz, shr_jeps2);
    out_iax[i] = ia.x;
    out_iay[i] = ia.y;
    out_iaz[i] = ia.z;
}

#else
//
// C implementation
////////////////////////////////////////////////////////////////////////////////
static PyObject *
p2p_acc_kernel(PyObject *_self, PyObject *_args)
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

    int typenum;
    char *fmt = NULL;
    if (sizeof(REAL) == sizeof(double)) {
        fmt = "IOOOOOOOOIOOOOOOOOO!O!O!";
        typenum = NPY_FLOAT64;
    } else if (sizeof(REAL) == sizeof(float)) {
        fmt = "IOOOOOOOOIOOOOOOOOO!O!O!";
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
                                      &PyArray_Type, &out_iaz))
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

    // main calculation
    unsigned int i, j;
    for (i = 0; i < ni; ++i) {
        REAL4 irm = {irx_ptr[i], iry_ptr[i],
                     irz_ptr[i], imass_ptr[i]};
        REAL4 ive = {ivx_ptr[i], ivy_ptr[i],
                     ivz_ptr[i], ieps2_ptr[i]};
        REAL3 ia = (REAL3){0, 0, 0};
        for (j = 0; j < nj; ++j) {
            REAL4 jrm = {jrx_ptr[j], jry_ptr[j],
                         jrz_ptr[j], jmass_ptr[j]};
            REAL4 jve = {jvx_ptr[j], jvy_ptr[j],
                         jvz_ptr[j], jeps2_ptr[j]};
            ia = p2p_acc_kernel_core(ia, irm, ive, jrm, jve);
        }
        iax_ptr[i] = ia.x;
        iay_ptr[i] = ia.y;
        iaz_ptr[i] = ia.z;
    }

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

    // Returns None
    Py_INCREF(Py_None);
    return Py_None;
}

#endif  // __OPENCL_VERSION__


#endif  // P2P_ACC_KERNEL_H


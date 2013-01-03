#ifndef NREG_KERNELS_H
#define NREG_KERNELS_H

#include"common.h"
#include"smoothing.h"

//
// nreg_Xkernel_core
////////////////////////////////////////////////////////////////////////////////
inline REAL8
nreg_Xkernel_core(REAL8 ira,
                  const REAL4 irm, const REAL4 ive,
                  const REAL4 jrm, const REAL4 jve,
                  const REAL dt)
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

    r.x += dt * v.x;                                                 // 2 FLOPs
    r.y += dt * v.y;                                                 // 2 FLOPs
    r.z += dt * v.z;                                                 // 2 FLOPs

    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs

    REAL2 ret = smoothed_inv_r1r3(r2, v.w);                          // 5 FLOPs
    REAL inv_r1 = ret.x;
    REAL inv_r3 = ret.y;

    r.x *= jrm.w;                                                    // 1 FLOPs
    r.y *= jrm.w;                                                    // 1 FLOPs
    r.z *= jrm.w;                                                    // 1 FLOPs

    ira.s0 += r.x;                                                   // 1 FLOPs
    ira.s1 += r.y;                                                   // 1 FLOPs
    ira.s2 += r.z;                                                   // 1 FLOPs
    ira.s3  = 0;
    ira.s4 -= inv_r3 * r.x;                                          // 2 FLOPs
    ira.s5 -= inv_r3 * r.y;                                          // 2 FLOPs
    ira.s6 -= inv_r3 * r.z;                                          // 2 FLOPs
    ira.s7 += inv_r1 * jrm.w;                                        // 2 FLOPs
    return ira;
}
// Total flop count: 37


//
// nreg_Vkernel_core
////////////////////////////////////////////////////////////////////////////////
inline REAL4
nreg_Vkernel_core(REAL4 ivk,
                  const REAL4 ivm, const REAL3 ia,
                  const REAL4 jvm, const REAL3 ja,
                  const REAL dt)
{
    REAL3 a;
    a.x = ia.x - ja.x;                                               // 1 FLOPs
    a.y = ia.y - ja.y;                                               // 1 FLOPs
    a.z = ia.z - ja.z;                                               // 1 FLOPs
    REAL3 v;
    v.x = ivm.x - jvm.x;                                             // 1 FLOPs
    v.y = ivm.y - jvm.y;                                             // 1 FLOPs
    v.z = ivm.z - jvm.z;                                             // 1 FLOPs

    v.x += dt * a.x;                                                 // 2 FLOPs
    v.y += dt * a.y;                                                 // 2 FLOPs
    v.z += dt * a.z;                                                 // 2 FLOPs

    REAL v2 = v.x * v.x + v.y * v.y + v.z * v.z;                     // 5 FLOPs

    v.x *= jvm.w;                                                    // 1 FLOPs
    v.y *= jvm.w;                                                    // 1 FLOPs
    v.z *= jvm.w;                                                    // 1 FLOPs
    v2 *= jvm.w;                                                     // 1 FLOPs

    ivk.x += v.x;                                                    // 1 FLOPs
    ivk.y += v.y;                                                    // 1 FLOPs
    ivk.z += v.z;                                                    // 1 FLOPs
    ivk.w += v2;                                                     // 1 FLOPs
    return ivk;
}
// Total flop count: 25


#ifdef __OPENCL_VERSION__
//
// OpenCL implementation
////////////////////////////////////////////////////////////////////////////////
inline REAL
nreg_X_accum(REAL iOmega,
                const REAL8 iData,
                const REAL eta,
                uint j_begin,
                uint j_end,
                __local REAL *sharedJObj_x,
                __local REAL *sharedJObj_y,
                __local REAL *sharedJObj_z,
                __local REAL *sharedJObj_mass,
                __local REAL *sharedJObj_vx,
                __local REAL *sharedJObj_vy,
                __local REAL *sharedJObj_vz,
                __local REAL *sharedJObj_eps2
               )
{
    uint j;
    for (j = j_begin; j < j_end; ++j) {
        REAL8 jData = (REAL8){sharedJObj_x[j],
                              sharedJObj_y[j],
                              sharedJObj_z[j],
                              sharedJObj_mass[j],
                              sharedJObj_vx[j],
                              sharedJObj_vy[j],
                              sharedJObj_vz[j],
                              sharedJObj_eps2[j]};
        iOmega = p2p_tstep_kernel_core(iOmega,
                                       iData.lo, iData.hi,
                                       jData.lo, jData.hi,
                                       eta);
    }
    return iOmega;
}


inline REAL
nreg_Xkernel_main_loop(const REAL8 iData,
                           const uint nj,
                           __global const REAL *jobj_x,
                           __global const REAL *jobj_y,
                           __global const REAL *jobj_z,
                           __global const REAL *jobj_mass,
                           __global const REAL *jobj_vx,
                           __global const REAL *jobj_vy,
                           __global const REAL *jobj_vz,
                           __global const REAL *jobj_eps2,
                           const REAL eta,
                           __local REAL *sharedJObj_x,
                           __local REAL *sharedJObj_y,
                           __local REAL *sharedJObj_z,
                           __local REAL *sharedJObj_mass,
                           __local REAL *sharedJObj_vx,
                           __local REAL *sharedJObj_vy,
                           __local REAL *sharedJObj_vz,
                           __local REAL *sharedJObj_eps2
                          )
{
    uint lsize = get_local_size(0);

    REAL iOmega = (REAL)0;

    uint tile;
    uint numTiles = (nj - 1)/lsize + 1;
    for (tile = 0; tile < numTiles; ++tile) {
        uint nb = min(lsize, (nj - (tile * lsize)));

        event_t e[8];
        e[0] = async_work_group_copy(sharedJObj_x, jobj_x + tile * lsize, nb, 0);
        e[1] = async_work_group_copy(sharedJObj_y, jobj_y + tile * lsize, nb, 0);
        e[2] = async_work_group_copy(sharedJObj_z, jobj_z + tile * lsize, nb, 0);
        e[3] = async_work_group_copy(sharedJObj_mass, jobj_mass + tile * lsize, nb, 0);
        e[4] = async_work_group_copy(sharedJObj_vx, jobj_vx + tile * lsize, nb, 0);
        e[5] = async_work_group_copy(sharedJObj_vy, jobj_vy + tile * lsize, nb, 0);
        e[6] = async_work_group_copy(sharedJObj_vz, jobj_vz + tile * lsize, nb, 0);
        e[7] = async_work_group_copy(sharedJObj_eps2, jobj_eps2 + tile * lsize, nb, 0);
        wait_group_events(8, e);

        uint j = 0;
        uint j_max = (nb > (JUNROLL - 1)) ? (nb - (JUNROLL - 1)):(0);
        for (; j < j_max; j += JUNROLL) {
            iOmega = p2p_accum_tstep(iOmega, iData,
                                     eta, j, j + JUNROLL,
                                     sharedJObj_x, sharedJObj_y, sharedJObj_z, sharedJObj_mass,
                                     sharedJObj_vx, sharedJObj_vy, sharedJObj_vz, sharedJObj_eps2);
        }
        iOmega = p2p_accum_tstep(iOmega, iData,
                                 eta, j, nb,
                                 sharedJObj_x, sharedJObj_y, sharedJObj_z, sharedJObj_mass,
                                 sharedJObj_vx, sharedJObj_vy, sharedJObj_vz, sharedJObj_eps2);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return iOmega;
}


__kernel void nreg_Xkernel(const uint ni,
                               __global const REAL *iobj_x,
                               __global const REAL *iobj_y,
                               __global const REAL *iobj_z,
                               __global const REAL *iobj_mass,
                               __global const REAL *iobj_vx,
                               __global const REAL *iobj_vy,
                               __global const REAL *iobj_vz,
                               __global const REAL *iobj_eps2,
                               const uint nj,
                               __global const REAL *jobj_x,
                               __global const REAL *jobj_y,
                               __global const REAL *jobj_z,
                               __global const REAL *jobj_mass,
                               __global const REAL *jobj_vx,
                               __global const REAL *jobj_vy,
                               __global const REAL *jobj_vz,
                               __global const REAL *jobj_eps2,
                               const REAL eta,
                               __global REAL *itstep,
                               __local REAL *sharedJObj_x,
                               __local REAL *sharedJObj_y,
                               __local REAL *sharedJObj_z,
                               __local REAL *sharedJObj_mass,
                               __local REAL *sharedJObj_vx,
                               __local REAL *sharedJObj_vy,
                               __local REAL *sharedJObj_vz,
                               __local REAL *sharedJObj_eps2
                              )
{
    uint gid = get_global_id(0);
    uint i = (gid < ni) ? (gid) : (ni-1);

    const REAL8 iData = (REAL8){iobj_x[i],
                                iobj_y[i],
                                iobj_z[i],
                                iobj_mass[i],
                                iobj_vx[i],
                                iobj_vy[i],
                                iobj_vz[i],
                                iobj_eps2[i]};

    REAL iomega = p2p_tstep_kernel_main_loop(iData,
                                             nj,
                                             jobj_x,
                                             jobj_y,
                                             jobj_z,
                                             jobj_mass,
                                             jobj_vx,
                                             jobj_vy,
                                             jobj_vz,
                                             jobj_eps2,
                                             eta,
                                             sharedJObj_x,
                                             sharedJObj_y,
                                             sharedJObj_z,
                                             sharedJObj_mass,
                                             sharedJObj_vx,
                                             sharedJObj_vy,
                                             sharedJObj_vz,
                                             sharedJObj_eps2);
//    itstep[i] = 2 * eta / iomega;
    itstep[i] = 2 * eta / sqrt(iomega);
}


__kernel void nreg_Vkernel(const uint ni,
                               __global const REAL *iobj_x,
                               __global const REAL *iobj_y,
                               __global const REAL *iobj_z,
                               __global const REAL *iobj_mass,
                               __global const REAL *iobj_vx,
                               __global const REAL *iobj_vy,
                               __global const REAL *iobj_vz,
                               __global const REAL *iobj_eps2,
                               const uint nj,
                               __global const REAL *jobj_x,
                               __global const REAL *jobj_y,
                               __global const REAL *jobj_z,
                               __global const REAL *jobj_mass,
                               __global const REAL *jobj_vx,
                               __global const REAL *jobj_vy,
                               __global const REAL *jobj_vz,
                               __global const REAL *jobj_eps2,
                               const REAL eta,
                               __global REAL *itstep,
                               __local REAL *sharedJObj_x,
                               __local REAL *sharedJObj_y,
                               __local REAL *sharedJObj_z,
                               __local REAL *sharedJObj_mass,
                               __local REAL *sharedJObj_vx,
                               __local REAL *sharedJObj_vy,
                               __local REAL *sharedJObj_vz,
                               __local REAL *sharedJObj_eps2
                              )
{
    uint gid = get_global_id(0);
    uint i = (gid < ni) ? (gid) : (ni-1);

    const REAL8 iData = (REAL8){iobj_x[i],
                                iobj_y[i],
                                iobj_z[i],
                                iobj_mass[i],
                                iobj_vx[i],
                                iobj_vy[i],
                                iobj_vz[i],
                                iobj_eps2[i]};

    REAL iomega = p2p_tstep_kernel_main_loop(iData,
                                             nj,
                                             jobj_x,
                                             jobj_y,
                                             jobj_z,
                                             jobj_mass,
                                             jobj_vx,
                                             jobj_vy,
                                             jobj_vz,
                                             jobj_eps2,
                                             eta,
                                             sharedJObj_x,
                                             sharedJObj_y,
                                             sharedJObj_z,
                                             sharedJObj_mass,
                                             sharedJObj_vx,
                                             sharedJObj_vy,
                                             sharedJObj_vz,
                                             sharedJObj_eps2);
//    itstep[i] = 2 * eta / iomega;
    itstep[i] = 2 * eta / sqrt(iomega);
}

#else
//
// C implementation
////////////////////////////////////////////////////////////////////////////////

//
// nreg_Xkernel
////////////////////////////////////////////////////////////////////////////////
static PyObject *
nreg_Xkernel(PyObject *_self, PyObject *_args)
{
    unsigned int ni, nj;
    REAL dt;
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
    PyObject *out_irx = NULL;
    PyObject *out_iry = NULL;
    PyObject *out_irz = NULL;
    PyObject *out_iax = NULL;
    PyObject *out_iay = NULL;
    PyObject *out_iaz = NULL;
    PyObject *out_iu = NULL;

    int typenum;
    char *fmt = NULL;
    if (sizeof(REAL) == sizeof(double)) {
        fmt = "IOOOOOOOOIOOOOOOOOdO!O!O!O!O!O!O!";
        typenum = NPY_FLOAT64;
    } else if (sizeof(REAL) == sizeof(float)) {
        fmt = "IOOOOOOOOIOOOOOOOOfO!O!O!O!O!O!O!";
        typenum = NPY_FLOAT32;
    }

    if (!PyArg_ParseTuple(_args, fmt, &ni,
                                      &inp_irx, &inp_iry, &inp_irz, &inp_imass,
                                      &inp_ivx, &inp_ivy, &inp_ivz, &inp_ieps2,
                                      &nj,
                                      &inp_jrx, &inp_jry, &inp_jrz, &inp_jmass,
                                      &inp_jvx, &inp_jvy, &inp_jvz, &inp_jeps2,
                                      &dt,
                                      &PyArray_Type, &out_irx,
                                      &PyArray_Type, &out_iry,
                                      &PyArray_Type, &out_irz,
                                      &PyArray_Type, &out_iax,
                                      &PyArray_Type, &out_iay,
                                      &PyArray_Type, &out_iaz,
                                      &PyArray_Type, &out_iu))
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
    PyObject *new_irx = PyArray_FROM_OTF(out_irx, typenum, NPY_INOUT_ARRAY);
    REAL *new_irx_ptr = (REAL *)PyArray_DATA(new_irx);

    PyObject *new_iry = PyArray_FROM_OTF(out_iry, typenum, NPY_INOUT_ARRAY);
    REAL *new_iry_ptr = (REAL *)PyArray_DATA(new_iry);

    PyObject *new_irz = PyArray_FROM_OTF(out_irz, typenum, NPY_INOUT_ARRAY);
    REAL *new_irz_ptr = (REAL *)PyArray_DATA(new_irz);

    PyObject *iax = PyArray_FROM_OTF(out_iax, typenum, NPY_INOUT_ARRAY);
    REAL *iax_ptr = (REAL *)PyArray_DATA(iax);

    PyObject *iay = PyArray_FROM_OTF(out_iay, typenum, NPY_INOUT_ARRAY);
    REAL *iay_ptr = (REAL *)PyArray_DATA(iay);

    PyObject *iaz = PyArray_FROM_OTF(out_iaz, typenum, NPY_INOUT_ARRAY);
    REAL *iaz_ptr = (REAL *)PyArray_DATA(iaz);

    PyObject *iu = PyArray_FROM_OTF(out_iu, typenum, NPY_INOUT_ARRAY);
    REAL *iu_ptr = (REAL *)PyArray_DATA(iu);

    // main calculation
    unsigned int i, j;
    for (i = 0; i < ni; ++i) {
        REAL4 irm = {irx_ptr[i], iry_ptr[i],
                     irz_ptr[i], imass_ptr[i]};
        REAL4 ive = {ivx_ptr[i], ivy_ptr[i],
                     ivz_ptr[i], ieps2_ptr[i]};
        REAL8 ira = (REAL8){0, 0, 0, 0, 0, 0, 0, 0};
        for (j = 0; j < nj; ++j) {
            REAL4 jrm = {jrx_ptr[j], jry_ptr[j],
                         jrz_ptr[j], jmass_ptr[j]};
            REAL4 jve = {jvx_ptr[j], jvy_ptr[j],
                         jvz_ptr[j], jeps2_ptr[j]};
            ira = nreg_Xkernel_core(ira, irm, ive, jrm, jve, dt);
        }
        new_irx_ptr[i] = ira.s0;
        new_iry_ptr[i] = ira.s1;
        new_irz_ptr[i] = ira.s2;
        iax_ptr[i] = ira.s4;
        iay_ptr[i] = ira.s5;
        iaz_ptr[i] = ira.s6;
        iu_ptr[i] = ira.s7 * irm.w;
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
    Py_DECREF(new_irx);
    Py_DECREF(new_iry);
    Py_DECREF(new_irz);
    Py_DECREF(iax);
    Py_DECREF(iay);
    Py_DECREF(iaz);
    Py_DECREF(iu);

    // Returns None
    Py_INCREF(Py_None);
    return Py_None;
}

//
// nreg_Vkernel
////////////////////////////////////////////////////////////////////////////////
static PyObject *
nreg_Vkernel(PyObject *_self, PyObject *_args)
{
    unsigned int ni, nj;
    REAL dt;
    // i-objs
    PyObject *inp_ivx = NULL;
    PyObject *inp_ivy = NULL;
    PyObject *inp_ivz = NULL;
    PyObject *inp_imass = NULL;
    PyObject *inp_iax = NULL;
    PyObject *inp_iay = NULL;
    PyObject *inp_iaz = NULL;
    // j-objs
    PyObject *inp_jvx = NULL;
    PyObject *inp_jvy = NULL;
    PyObject *inp_jvz = NULL;
    PyObject *inp_jmass = NULL;
    PyObject *inp_jax = NULL;
    PyObject *inp_jay = NULL;
    PyObject *inp_jaz = NULL;
    // out-objs
    PyObject *out_ivx = NULL;
    PyObject *out_ivy = NULL;
    PyObject *out_ivz = NULL;
    PyObject *out_ik = NULL;

    int typenum;
    char *fmt = NULL;
    if (sizeof(REAL) == sizeof(double)) {
        fmt = "IOOOOOOOIOOOOOOOdO!O!O!O!";
        typenum = NPY_FLOAT64;
    } else if (sizeof(REAL) == sizeof(float)) {
        fmt = "IOOOOOOOIOOOOOOOfO!O!O!O!";
        typenum = NPY_FLOAT32;
    }

    if (!PyArg_ParseTuple(_args, fmt, &ni,
                                      &inp_ivx, &inp_ivy, &inp_ivz, &inp_imass,
                                      &inp_iax, &inp_iay, &inp_iaz,
                                      &nj,
                                      &inp_jvx, &inp_jvy, &inp_jvz, &inp_jmass,
                                      &inp_jax, &inp_jay, &inp_jaz,
                                      &dt,
                                      &PyArray_Type, &out_ivx,
                                      &PyArray_Type, &out_ivy,
                                      &PyArray_Type, &out_ivz,
                                      &PyArray_Type, &out_ik))
        return NULL;

    // i-objs
    PyObject *ivx = PyArray_FROM_OTF(inp_ivx, typenum, NPY_IN_ARRAY);
    REAL *ivx_ptr = (REAL *)PyArray_DATA(ivx);

    PyObject *ivy = PyArray_FROM_OTF(inp_ivy, typenum, NPY_IN_ARRAY);
    REAL *ivy_ptr = (REAL *)PyArray_DATA(ivy);

    PyObject *ivz = PyArray_FROM_OTF(inp_ivz, typenum, NPY_IN_ARRAY);
    REAL *ivz_ptr = (REAL *)PyArray_DATA(ivz);

    PyObject *imass = PyArray_FROM_OTF(inp_imass, typenum, NPY_IN_ARRAY);
    REAL *imass_ptr = (REAL *)PyArray_DATA(imass);

    PyObject *iax = PyArray_FROM_OTF(inp_iax, typenum, NPY_IN_ARRAY);
    REAL *iax_ptr = (REAL *)PyArray_DATA(iax);

    PyObject *iay = PyArray_FROM_OTF(inp_iay, typenum, NPY_IN_ARRAY);
    REAL *iay_ptr = (REAL *)PyArray_DATA(iay);

    PyObject *iaz = PyArray_FROM_OTF(inp_iaz, typenum, NPY_IN_ARRAY);
    REAL *iaz_ptr = (REAL *)PyArray_DATA(iaz);

    // j-objs
    PyObject *jvx = PyArray_FROM_OTF(inp_jvx, typenum, NPY_IN_ARRAY);
    REAL *jvx_ptr = (REAL *)PyArray_DATA(jvx);

    PyObject *jvy = PyArray_FROM_OTF(inp_jvy, typenum, NPY_IN_ARRAY);
    REAL *jvy_ptr = (REAL *)PyArray_DATA(jvy);

    PyObject *jvz = PyArray_FROM_OTF(inp_jvz, typenum, NPY_IN_ARRAY);
    REAL *jvz_ptr = (REAL *)PyArray_DATA(jvz);

    PyObject *jmass = PyArray_FROM_OTF(inp_jmass, typenum, NPY_IN_ARRAY);
    REAL *jmass_ptr = (REAL *)PyArray_DATA(jmass);

    PyObject *jax = PyArray_FROM_OTF(inp_jax, typenum, NPY_IN_ARRAY);
    REAL *jax_ptr = (REAL *)PyArray_DATA(jax);

    PyObject *jay = PyArray_FROM_OTF(inp_jay, typenum, NPY_IN_ARRAY);
    REAL *jay_ptr = (REAL *)PyArray_DATA(jay);

    PyObject *jaz = PyArray_FROM_OTF(inp_jaz, typenum, NPY_IN_ARRAY);
    REAL *jaz_ptr = (REAL *)PyArray_DATA(jaz);

    // out-objs
    PyObject *new_ivx = PyArray_FROM_OTF(out_ivx, typenum, NPY_INOUT_ARRAY);
    REAL *new_ivx_ptr = (REAL *)PyArray_DATA(new_ivx);

    PyObject *new_ivy = PyArray_FROM_OTF(out_ivy, typenum, NPY_INOUT_ARRAY);
    REAL *new_ivy_ptr = (REAL *)PyArray_DATA(new_ivy);

    PyObject *new_ivz = PyArray_FROM_OTF(out_ivz, typenum, NPY_INOUT_ARRAY);
    REAL *new_ivz_ptr = (REAL *)PyArray_DATA(new_ivz);

    PyObject *ik = PyArray_FROM_OTF(out_ik, typenum, NPY_INOUT_ARRAY);
    REAL *ik_ptr = (REAL *)PyArray_DATA(ik);

    // main calculation
    unsigned int i, j;
    for (i = 0; i < ni; ++i) {
        REAL4 ivm = {ivx_ptr[i], ivy_ptr[i],
                     ivz_ptr[i], imass_ptr[i]};
        REAL3 ia = {iax_ptr[i], iay_ptr[i],
                    iaz_ptr[i]};
        REAL4 ivk = (REAL4){0, 0, 0, 0};
        for (j = 0; j < nj; ++j) {
            REAL4 jvm = {jvx_ptr[j], jvy_ptr[j],
                         jvz_ptr[j], jmass_ptr[j]};
            REAL3 ja = {jax_ptr[j], jay_ptr[j],
                        jaz_ptr[j]};
            ivk = nreg_Vkernel_core(ivk, ivm, ia, jvm, ja, dt);
        }
        new_ivx_ptr[i] = ivk.x;
        new_ivy_ptr[i] = ivk.y;
        new_ivz_ptr[i] = ivk.z;
        ik_ptr[i] = ivm.w * ivk.w / 2;
    }

    // Decrement the reference counts for auxiliary i-objs
    Py_DECREF(ivx);
    Py_DECREF(ivy);
    Py_DECREF(ivz);
    Py_DECREF(imass);
    Py_DECREF(iax);
    Py_DECREF(iay);
    Py_DECREF(iaz);

    // Decrement the reference counts for auxiliary j-objs
    Py_DECREF(jvx);
    Py_DECREF(jvy);
    Py_DECREF(jvz);
    Py_DECREF(jmass);
    Py_DECREF(jax);
    Py_DECREF(jay);
    Py_DECREF(jaz);

    // Decrement the reference counts for auxiliary out-objs
    Py_DECREF(new_ivx);
    Py_DECREF(new_ivy);
    Py_DECREF(new_ivz);
    Py_DECREF(ik);

    // Returns None
    Py_INCREF(Py_None);
    return Py_None;
}

#endif  // __OPENCL_VERSION__


#endif  // NREG_KERNELS_H


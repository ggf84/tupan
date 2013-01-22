#ifndef BIOS_KERNEL_H
#define BIOS_KERNEL_H

#include"common.h"
#include"smoothing.h"
#include"universal_kepler_solver.h"


/*
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * *  BIOS - BInary-based n-bOdy Solver  * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 */


inline REAL
get_phi(const REAL m,
        const REAL x,
        const REAL y,
        const REAL z,
        const REAL eps2)
{
    REAL r2 = x * x + y * y + z * z;                                 // 5 FLOPs
    REAL inv_r = smoothed_inv_r1(r2, eps2);                          // 3 FLOPs
    return m * inv_r;                                                // 1 FLOPs
}


inline REAL3
get_acc(const REAL m,
        const REAL x,
        const REAL y,
        const REAL z,
        const REAL eps2)
{
    REAL r2 = x * x + y * y + z * z;                                 // 5 FLOPs
    REAL inv_r3 = smoothed_inv_r3(r2, eps2);                         // 4 FLOPs
    REAL m_r3 = m * inv_r3;                                          // 1 FLOPs
    REAL3 a;
    a.x = -m_r3 * x;                                                 // 1 FLOPs
    a.y = -m_r3 * y;                                                 // 1 FLOPs
    a.z = -m_r3 * z;                                                 // 1 FLOPs
    return a;
}


inline void
leapfrog(const REAL dt,
         const REAL4 r0,
         const REAL4 v0,
         REAL4 *r1,
         REAL4 *v1)
{
    REAL4 r = r0;
    REAL4 v = v0;
    REAL dt_2 = dt / 2;

    r.x += v.x * dt_2;
    r.y += v.y * dt_2;
    r.z += v.z * dt_2;

    REAL3 a = get_acc(r.w, r.x, r.y, r.z, v.w);

    v.x += a.x * dt;
    v.y += a.y * dt;
    v.z += a.z * dt;

    r.x += v.x * dt_2;
    r.y += v.y * dt_2;
    r.z += v.z * dt_2;

    *r1 = r;
    *v1 = v;
}


inline void
TTL_core(const REAL h,
         const REAL u0,
         const REAL k0,
         REAL4 *r0,
         REAL4 *v0,
         REAL *t)
{
    REAL4 r = *r0;
    REAL4 v = *v0;

    REAL dt0 = h / (2*u0);
    *t += dt0;
    r.x += v.x * dt0;
    r.y += v.y * dt0;
    r.z += v.z * dt0;

    REAL3 a12 = get_acc(r.w, r.x, r.y, r.z, v.w);
    REAL u12 = get_phi(r.w, r.x, r.y, r.z, v.w);
    REAL dt12 = h / u12;

    v.x += a12.x * dt12;
    v.y += a12.y * dt12;
    v.z += a12.z * dt12;

    REAL k1 = (v.x * v.x + v.y * v.y + v.z * v.z)/2;
    REAL u1 = u0 + (k1 - k0);

    REAL dt1 = h / (2*u1);
    *t += dt1;
    r.x += v.x * dt1;
    r.y += v.y * dt1;
    r.z += v.z * dt1;

    *r0 = r;
    *v0 = v;
}

#ifdef DOUBLE
    #define TOLERANCE 2.3283064365386962891E-10     // sqrt(2^-64)
#else
    #define TOLERANCE 1.52587890625E-5              // sqrt(2^-32)
#endif
inline void
TTL(const REAL dt0,
    const REAL4 r0,
    const REAL4 v0,
    REAL4 *r1,
    REAL4 *v1)
{
    int i;
    int err = 0;
    int nsteps = 1;
    REAL t = 0;
    REAL4 r = r0;
    REAL4 v = v0;

    do {
        err = 0;
        REAL dt = dt0 / nsteps;
        for (i = 0; i < nsteps; ++i) {

            REAL mij = r.w;
            REAL r2 = (r.x * r.x + r.y * r.y + r.z * r.z);
            REAL rv = (r.x * v.x + r.y * v.y + r.z * v.z);
            REAL v2 = (v.x * v.x + v.y * v.y + v.z * v.z);

            REAL2 ret = smoothed_inv_r1r3(r2, v.w);
            REAL inv_r = ret.x;
            REAL inv_r3 = ret.y;

            REAL mij_r = mij * inv_r;
            REAL mij_r3 = -mij * inv_r3;

            REAL k0 = v2/2;

            REAL u0 = mij_r;
            REAL u0dot = mij_r3 * rv;

            REAL h = dt * (u0 + (dt / 2) * (u0dot));

            TTL_core(h, u0, k0, &r, &v, &t);

            REAL dts = (i+1) * dt;
            if (fabs(fabs(t) - dts)/dts > TOLERANCE) {
                err = -1;
                t = 0;
                r = r0;
                v = v0;
                nsteps *= 2;
                i = nsteps;
            }
            if (r2 == 0) {
                err = 0;
                r = r0;
                v = v0;
            }
        }
    } while (err != 0);

    *r1 = r;
    *v1 = v;
}


inline void
twobody_solver(const REAL dt,
               const REAL4 r0,
               const REAL4 v0,
               REAL4 *r1,
               REAL4 *v1)
{
//    leapfrog(dt, r0, v0, r1, v1);
//    TTL(dt, r0, v0, r1, v1);
    universal_kepler_solver(dt, r0, v0, r1, v1);
}


//
// bios_kernel_core
////////////////////////////////////////////////////////////////////////////////
inline REAL8
bios_kernel_core(REAL8 idrdv,
                 const REAL4 irm, const REAL4 ive,
                 const REAL4 jrm, const REAL4 jve,
                 const REAL dt)
{
    REAL4 r0;
    r0.x = irm.x - jrm.x;                                            // 1 FLOPs
    r0.y = irm.y - jrm.y;                                            // 1 FLOPs
    r0.z = irm.z - jrm.z;                                            // 1 FLOPs
    r0.w = irm.w + jrm.w;                                            // 1 FLOPs
    REAL4 v0;
    v0.x = ive.x - jve.x;                                            // 1 FLOPs
    v0.y = ive.y - jve.y;                                            // 1 FLOPs
    v0.z = ive.z - jve.z;                                            // 1 FLOPs
    v0.w = ive.w + jve.w;                                            // 1 FLOPs

    REAL4 r1, v1;
//    twobody_solver(dt, r0, v0, &r1, &v1);                            // ? FLOPS


    REAL r2 = r0.x * r0.x + r0.y * r0.y + r0.z * r0.z;
    REAL v2 = v0.x * v0.x + v0.y * v0.y + v0.z * v0.z;
    REAL gamma = sqrt(r2 / (v2 + 2 * r0.w / sqrt(r2)));
    if (64 * dt < gamma) {
        leapfrog(dt, r0, v0, &r1, &v1);
    } else {
        universal_kepler_solver(dt, r0, v0, &r1, &v1);
    }


    REAL muj = jrm.w / r0.w;                                         // 1 FLOPs
    REAL dt_2 = dt / 2;                                              // 1 FLOPs

    r0.x += v0.x * dt_2;                                             // 2 FLOPs
    r0.y += v0.y * dt_2;                                             // 2 FLOPs
    r0.z += v0.z * dt_2;                                             // 2 FLOPs
    r1.x -= v1.x * dt_2;                                             // 2 FLOPs
    r1.y -= v1.y * dt_2;                                             // 2 FLOPs
    r1.z -= v1.z * dt_2;                                             // 2 FLOPs

    idrdv.s0 += muj * (r1.x - r0.x);                                 // 3 FLOPs
    idrdv.s1 += muj * (r1.y - r0.y);                                 // 3 FLOPs
    idrdv.s2 += muj * (r1.z - r0.z);                                 // 3 FLOPs
    idrdv.s3  = 0;
    idrdv.s4 += muj * (v1.x - v0.x);                                 // 3 FLOPs
    idrdv.s5 += muj * (v1.y - v0.y);                                 // 3 FLOPs
    idrdv.s6 += muj * (v1.z - v0.z);                                 // 3 FLOPs
    idrdv.s7  = 0;

    return idrdv;
}
// Total flop count: 36 + ?




#ifdef __OPENCL_VERSION__
//
// OpenCL implementation
////////////////////////////////////////////////////////////////////////////////
inline REAL8
bios_kernel_accum(REAL8 idrdv,
                  const REAL8 idata,
                  const REAL dt,
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
        idrdv = bios_kernel_core(idrdv,
                                 idata.lo, idata.hi,
                                 jdata.lo, jdata.hi,
                                 dt);
    }
    return idrdv;
}


inline REAL8
bios_kernel_main_loop(const REAL8 idata,
                      const uint nj,
                      __global const REAL *inp_jrx,
                      __global const REAL *inp_jry,
                      __global const REAL *inp_jrz,
                      __global const REAL *inp_jmass,
                      __global const REAL *inp_jvx,
                      __global const REAL *inp_jvy,
                      __global const REAL *inp_jvz,
                      __global const REAL *inp_jeps2,
                      const REAL dt,
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

    REAL8 idrdv = (REAL8){0, 0, 0, 0, 0, 0, 0, 0};

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
            idrdv = bios_kernel_accum(idrdv, idata,
                                      dt, j, j + JUNROLL,
                                      shr_jrx, shr_jry, shr_jrz, shr_jmass,
                                      shr_jvx, shr_jvy, shr_jvz, shr_jeps2);
        }
        idrdv = bios_kernel_accum(idrdv, idata,
                                  dt, j, nb,
                                  shr_jrx, shr_jry, shr_jrz, shr_jmass,
                                  shr_jvx, shr_jvy, shr_jvz, shr_jeps2);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return idrdv;
}


__kernel void bios_kernel(const uint ni,
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
                          const REAL dt,
                          __global REAL *out_idrx,
                          __global REAL *out_idry,
                          __global REAL *out_idrz,
                          __global REAL *out_idvx,
                          __global REAL *out_idvy,
                          __global REAL *out_idvz,
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

    REAL8 idrdv = bios_kernel_main_loop(idata,
                                        nj,
                                        inp_jrx, inp_jry, inp_jrz, inp_jmass,
                                        inp_jvx, inp_jvy, inp_jvz, inp_jeps2,
                                        dt,
                                        shr_jrx, shr_jry, shr_jrz, shr_jmass,
                                        shr_jvx, shr_jvy, shr_jvz, shr_jeps2);
    out_idrx[i] = idrdv.s0;
    out_idry[i] = idrdv.s1;
    out_idrz[i] = idrdv.s2;
    out_idvx[i] = idrdv.s4;
    out_idvy[i] = idrdv.s5;
    out_idvz[i] = idrdv.s6;
}

#else
//
// C implementation
////////////////////////////////////////////////////////////////////////////////

inline void
main_bios_kernel(const unsigned int ni,
                 const REAL *irx, const REAL *iry, const REAL *irz, const REAL *imass,
                 const REAL *ivx, const REAL *ivy, const REAL *ivz, const REAL *ieps2,
                 const unsigned int nj,
                 const REAL *jrx, const REAL *jry, const REAL *jrz, const REAL *jmass,
                 const REAL *jvx, const REAL *jvy, const REAL *jvz, const REAL *jeps2,
                 const REAL dt,
                 REAL *idrx, REAL *idry, REAL *idrz,
                 REAL *idvx, REAL *idvy, REAL *idvz)
{
    unsigned int i, j;
    for (i = 0; i < ni; ++i) {
        REAL4 irm = {irx[i], iry[i], irz[i], imass[i]};
        REAL4 ive = {ivx[i], ivy[i], ivz[i], ieps2[i]};
        REAL8 idrdv = (REAL8){0, 0, 0, 0, 0, 0, 0, 0};
        for (j = 0; j < nj; ++j) {
            REAL4 jrm = {jrx[j], jry[j], jrz[j], jmass[j]};
            REAL4 jve = {jvx[j], jvy[j], jvz[j], jeps2[j]};
            idrdv = bios_kernel_core(idrdv, irm, ive, jrm, jve, dt);
        }
        idrx[i] = idrdv.s0;
        idry[i] = idrdv.s1;
        idrz[i] = idrdv.s2;
        idvx[i] = idrdv.s4;
        idvy[i] = idrdv.s5;
        idvz[i] = idrdv.s6;
    }
}


#ifndef __USE_CTYPES__
static PyObject *
bios_kernel(PyObject *_self, PyObject *_args)
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
    PyObject *out_idrx = NULL;
    PyObject *out_idry = NULL;
    PyObject *out_idrz = NULL;
    PyObject *out_idvx = NULL;
    PyObject *out_idvy = NULL;
    PyObject *out_idvz = NULL;

    int typenum;
    char *fmt = NULL;
    if (sizeof(REAL) == sizeof(double)) {
        fmt = "IOOOOOOOOIOOOOOOOOdO!O!O!O!O!O!";
        typenum = NPY_FLOAT64;
    } else if (sizeof(REAL) == sizeof(float)) {
        fmt = "IOOOOOOOOIOOOOOOOOfO!O!O!O!O!O!";
        typenum = NPY_FLOAT32;
    }

    if (!PyArg_ParseTuple(_args, fmt, &ni,
                                      &inp_irx, &inp_iry, &inp_irz, &inp_imass,
                                      &inp_ivx, &inp_ivy, &inp_ivz, &inp_ieps2,
                                      &nj,
                                      &inp_jrx, &inp_jry, &inp_jrz, &inp_jmass,
                                      &inp_jvx, &inp_jvy, &inp_jvz, &inp_jeps2,
                                      &dt,
                                      &PyArray_Type, &out_idrx,
                                      &PyArray_Type, &out_idry,
                                      &PyArray_Type, &out_idrz,
                                      &PyArray_Type, &out_idvx,
                                      &PyArray_Type, &out_idvy,
                                      &PyArray_Type, &out_idvz))
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
    PyObject *idrx = PyArray_FROM_OTF(out_idrx, typenum, NPY_INOUT_ARRAY);
    REAL *idrx_ptr = (REAL *)PyArray_DATA(idrx);

    PyObject *idry = PyArray_FROM_OTF(out_idry, typenum, NPY_INOUT_ARRAY);
    REAL *idry_ptr = (REAL *)PyArray_DATA(idry);

    PyObject *idrz = PyArray_FROM_OTF(out_idrz, typenum, NPY_INOUT_ARRAY);
    REAL *idrz_ptr = (REAL *)PyArray_DATA(idrz);

    PyObject *idvx = PyArray_FROM_OTF(out_idvx, typenum, NPY_INOUT_ARRAY);
    REAL *idvx_ptr = (REAL *)PyArray_DATA(idvx);

    PyObject *idvy = PyArray_FROM_OTF(out_idvy, typenum, NPY_INOUT_ARRAY);
    REAL *idvy_ptr = (REAL *)PyArray_DATA(idvy);

    PyObject *idvz = PyArray_FROM_OTF(out_idvz, typenum, NPY_INOUT_ARRAY);
    REAL *idvz_ptr = (REAL *)PyArray_DATA(idvz);

    // main calculation
    main_bios_kernel(ni,
                     irx_ptr, iry_ptr, irz_ptr, imass_ptr,
                     ivx_ptr, ivy_ptr, ivz_ptr, ieps2_ptr,
                     nj,
                     jrx_ptr, jry_ptr, jrz_ptr, jmass_ptr,
                     jvx_ptr, jvy_ptr, jvz_ptr, jeps2_ptr,
                     dt,
                     idrx_ptr, idry_ptr, idrz_ptr,
                     idvx_ptr, idvy_ptr, idvz_ptr);

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
    Py_DECREF(idrx);
    Py_DECREF(idry);
    Py_DECREF(idrz);
    Py_DECREF(idvx);
    Py_DECREF(idvy);
    Py_DECREF(idvz);

    // Returns None
    Py_INCREF(Py_None);
    return Py_None;
}
#endif  // __USE_CTYPES__

#endif  // __OPENCL_VERSION__


#endif  // BIOS_KERNEL_H


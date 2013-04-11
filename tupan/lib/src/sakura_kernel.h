#ifndef SAKURA_KERNEL_H
#define SAKURA_KERNEL_H

#include"common.h"
#include"smoothing.h"
#include"universal_kepler_solver.h"


/*
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * SAKURA  * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
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
twobody_solver(const REAL dt,
               const REAL4 r0,
               const REAL4 v0,
               REAL4 *r1,
               REAL4 *v1)
{
//    leapfrog(dt, r0, v0, r1, v1);
    universal_kepler_solver(dt, r0, v0, r1, v1);
}


//
// sakura_kernel_core
////////////////////////////////////////////////////////////////////////////////
inline REAL8
sakura_kernel_core(REAL8 idrdv,
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

    REAL dt_2 = dt / 2;                                              // 1 FLOPs

    REAL4 rr0 = r0;
    rr0.x -= v0.x * dt_2;                                            // 2 FLOPs
    rr0.y -= v0.y * dt_2;                                            // 2 FLOPs
    rr0.z -= v0.z * dt_2;                                            // 2 FLOPs
    REAL4 vv0 = v0;

    REAL4 rr1, vv1;
    twobody_solver(dt, rr0, vv0, &rr1, &vv1);                        // ? FLOPS

    REAL4 r1 = rr1;
    r1.x -= vv1.x * dt_2;                                            // 2 FLOPs
    r1.y -= vv1.y * dt_2;                                            // 2 FLOPs
    r1.z -= vv1.z * dt_2;                                            // 2 FLOPs
    REAL4 v1 = vv1;

    REAL muj = jrm.w / r0.w;                                         // 1 FLOPs

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
sakura_kernel_accum(REAL8 idrdv,
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
        idrdv = sakura_kernel_core(idrdv,
                                   idata.lo, idata.hi,
                                   jdata.lo, jdata.hi,
                                   dt);
    }
    return idrdv;
}


inline REAL8
sakura_kernel_main_loop(const REAL8 idata,
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
            idrdv = sakura_kernel_accum(idrdv, idata,
                                        dt, j, j + JUNROLL,
                                        shr_jrx, shr_jry, shr_jrz, shr_jmass,
                                        shr_jvx, shr_jvy, shr_jvz, shr_jeps2);
        }
        idrdv = sakura_kernel_accum(idrdv, idata,
                                    dt, j, nb,
                                    shr_jrx, shr_jry, shr_jrz, shr_jmass,
                                    shr_jvx, shr_jvy, shr_jvz, shr_jeps2);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return idrdv;
}


__kernel void sakura_kernel(const uint ni,
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

    REAL8 idrdv = sakura_kernel_main_loop(idata,
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
sakura_kernel(const unsigned int ni,
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
            idrdv = sakura_kernel_core(idrdv, irm, ive, jrm, jve, dt);
        }
        idrx[i] = idrdv.s0;
        idry[i] = idrdv.s1;
        idrz[i] = idrdv.s2;
        idvx[i] = idrdv.s4;
        idvy[i] = idrdv.s5;
        idvz[i] = idrdv.s6;
    }
}

#endif  // __OPENCL_VERSION__
#endif  // SAKURA_KERNEL_H


#include"common.h"
#include"smoothing.h"


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


inline void
nreg_Xkernel(const unsigned int ni,
             const REAL *irx, const REAL *iry, const REAL *irz, const REAL *imass,
             const REAL *ivx, const REAL *ivy, const REAL *ivz, const REAL *ieps2,
             const unsigned int nj,
             const REAL *jrx, const REAL *jry, const REAL *jrz, const REAL *jmass,
             const REAL *jvx, const REAL *jvy, const REAL *jvz, const REAL *jeps2,
             const REAL dt,
             REAL *new_irx, REAL *new_iry, REAL *new_irz,
             REAL *iax, REAL *iay, REAL *iaz,
             REAL *iu)
{
    unsigned int i, j;
    for (i = 0; i < ni; ++i) {
        REAL4 irm = {irx[i], iry[i], irz[i], imass[i]};
        REAL4 ive = {ivx[i], ivy[i], ivz[i], ieps2[i]};
        REAL8 ira = (REAL8){0, 0, 0, 0, 0, 0, 0, 0};
        for (j = 0; j < nj; ++j) {
            REAL4 jrm = {jrx[j], jry[j], jrz[j], jmass[j]};
            REAL4 jve = {jvx[j], jvy[j], jvz[j], jeps2[j]};
            ira = nreg_Xkernel_core(ira, irm, ive, jrm, jve, dt);
        }
        new_irx[i] = ira.s0;
        new_iry[i] = ira.s1;
        new_irz[i] = ira.s2;
        iax[i] = ira.s4;
        iay[i] = ira.s5;
        iaz[i] = ira.s6;
        iu[i] = ira.s7 * irm.w;
    }
}


inline void
nreg_Vkernel(const unsigned int ni,
             const REAL *ivx, const REAL *ivy, const REAL *ivz, const REAL *imass,
             const REAL *iax, const REAL *iay, const REAL *iaz,
             const unsigned int nj,
             const REAL *jvx, const REAL *jvy, const REAL *jvz, const REAL *jmass,
             const REAL *jax, const REAL *jay, const REAL *jaz,
             const REAL dt,
             REAL *new_ivx, REAL *new_ivy, REAL *new_ivz,
             REAL *ik)
{
    unsigned int i, j;
    for (i = 0; i < ni; ++i) {
        REAL4 ivm = {ivx[i], ivy[i], ivz[i], imass[i]};
        REAL3 ia = {iax[i], iay[i], iaz[i]};
        REAL4 ivk = (REAL4){0, 0, 0, 0};
        for (j = 0; j < nj; ++j) {
            REAL4 jvm = {jvx[j], jvy[j], jvz[j], jmass[j]};
            REAL3 ja = {jax[j], jay[j], jaz[j]};
            ivk = nreg_Vkernel_core(ivk, ivm, ia, jvm, ja, dt);
        }
        new_ivx[i] = ivk.x;
        new_ivy[i] = ivk.y;
        new_ivz[i] = ivk.z;
        ik[i] = ivm.w * ivk.w / 2;
    }
}


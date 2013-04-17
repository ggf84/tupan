#include "nreg_kernels_common.h"


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


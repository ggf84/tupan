#include "nreg_kernels_common.h"


__attribute__((reqd_work_group_size(LSIZE, 1, 1)))
kernel void
nreg_Xkernel(
    UINT const ni,
    global REALn const __im[restrict],
    global REALn const __irx[restrict],
    global REALn const __iry[restrict],
    global REALn const __irz[restrict],
    global REALn const __ie2[restrict],
    global REALn const __ivx[restrict],
    global REALn const __ivy[restrict],
    global REALn const __ivz[restrict],
    UINT const nj,
    global REAL const __jm[restrict],
    global REAL const __jrx[restrict],
    global REAL const __jry[restrict],
    global REAL const __jrz[restrict],
    global REAL const __je2[restrict],
    global REAL const __jvx[restrict],
    global REAL const __jvy[restrict],
    global REAL const __jvz[restrict],
    REAL const dt,
    global REALn __idrx[restrict],
    global REALn __idry[restrict],
    global REALn __idrz[restrict],
    global REALn __iax[restrict],
    global REALn __iay[restrict],
    global REALn __iaz[restrict],
    global REALn __iu[restrict])
{
    for (UINT i = LSIZE * get_group_id(0) + get_global_offset(0);
              i < ni;
              i += LSIZE * get_num_groups(0)) {
        UINT lid = get_local_id(0);
        UINT gid = ((i + lid) < ni) ? (i + lid):(0);

        REALn im = __im[gid];
        REALn irx = __irx[gid];
        REALn iry = __iry[gid];
        REALn irz = __irz[gid];
        REALn ie2 = __ie2[gid];
        REALn ivx = __ivx[gid];
        REALn ivy = __ivy[gid];
        REALn ivz = __ivz[gid];

        REALn idrx = (REALn)(0);
        REALn idry = (REALn)(0);
        REALn idrz = (REALn)(0);
        REALn iax = (REALn)(0);
        REALn iay = (REALn)(0);
        REALn iaz = (REALn)(0);
        REALn iu = (REALn)(0);

        UINT j = 0;

        #ifdef FAST_LOCAL_MEM
        local REAL _jm[LSIZE];
        local REAL _jrx[LSIZE];
        local REAL _jry[LSIZE];
        local REAL _jrz[LSIZE];
        local REAL _je2[LSIZE];
        local REAL _jvx[LSIZE];
        local REAL _jvy[LSIZE];
        local REAL _jvz[LSIZE];
        for (; (j + LSIZE - 1) < nj; j += LSIZE) {
            REAL jm = __jm[j + lid];
            REAL jrx = __jrx[j + lid];
            REAL jry = __jry[j + lid];
            REAL jrz = __jrz[j + lid];
            REAL je2 = __je2[j + lid];
            REAL jvx = __jvx[j + lid];
            REAL jvy = __jvy[j + lid];
            REAL jvz = __jvz[j + lid];
            barrier(CLK_LOCAL_MEM_FENCE);
            _jm[lid] = jm;
            _jrx[lid] = jrx;
            _jry[lid] = jry;
            _jrz[lid] = jrz;
            _je2[lid] = je2;
            _jvx[lid] = jvx;
            _jvy[lid] = jvy;
            _jvz[lid] = jvz;
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll UNROLL
            for (UINT k = 0; k < LSIZE; ++k) {
                jm = _jm[k];
                jrx = _jrx[k];
                jry = _jry[k];
                jrz = _jrz[k];
                je2 = _je2[k];
                jvx = _jvx[k];
                jvy = _jvy[k];
                jvz = _jvz[k];
                nreg_Xkernel_core(
                    dt,
                    im, irx, iry, irz, ie2, ivx, ivy, ivz,
                    jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                    &idrx, &idry, &idrz, &iax, &iay, &iaz, &iu);
            }
        }
        #endif

        for (; j < nj; ++j) {
            REAL jm = __jm[j];
            REAL jrx = __jrx[j];
            REAL jry = __jry[j];
            REAL jrz = __jrz[j];
            REAL je2 = __je2[j];
            REAL jvx = __jvx[j];
            REAL jvy = __jvy[j];
            REAL jvz = __jvz[j];
            nreg_Xkernel_core(
                dt,
                im, irx, iry, irz, ie2, ivx, ivy, ivz,
                jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                &idrx, &idry, &idrz, &iax, &iay, &iaz, &iu);
        }

        __idrx[gid] = idrx;
        __idry[gid] = idry;
        __idrz[gid] = idrz;
        __iax[gid] = iax;
        __iay[gid] = iay;
        __iaz[gid] = iaz;
        __iu[gid] = im * iu;
    }
}


__attribute__((reqd_work_group_size(LSIZE, 1, 1)))
kernel void
nreg_Vkernel(
    UINT const ni,
    global REALn const __im[restrict],
    global REALn const __ivx[restrict],
    global REALn const __ivy[restrict],
    global REALn const __ivz[restrict],
    global REALn const __iax[restrict],
    global REALn const __iay[restrict],
    global REALn const __iaz[restrict],
    UINT const nj,
    global REAL const __jm[restrict],
    global REAL const __jvx[restrict],
    global REAL const __jvy[restrict],
    global REAL const __jvz[restrict],
    global REAL const __jax[restrict],
    global REAL const __jay[restrict],
    global REAL const __jaz[restrict],
    REAL const dt,
    global REALn __idvx[restrict],
    global REALn __idvy[restrict],
    global REALn __idvz[restrict],
    global REALn __ik[restrict])
{
    for (UINT i = LSIZE * get_group_id(0) + get_global_offset(0);
              i < ni;
              i += LSIZE * get_num_groups(0)) {
        UINT lid = get_local_id(0);
        UINT gid = ((i + lid) < ni) ? (i + lid):(0);

        REALn im = __im[gid];
        REALn ivx = __ivx[gid];
        REALn ivy = __ivy[gid];
        REALn ivz = __ivz[gid];
        REALn iax = __iax[gid];
        REALn iay = __iay[gid];
        REALn iaz = __iaz[gid];

        REALn idvx = (REALn)(0);
        REALn idvy = (REALn)(0);
        REALn idvz = (REALn)(0);
        REALn ik = (REALn)(0);

        UINT j = 0;

        #ifdef FAST_LOCAL_MEM
        local REAL _jm[LSIZE];
        local REAL _jvx[LSIZE];
        local REAL _jvy[LSIZE];
        local REAL _jvz[LSIZE];
        local REAL _jax[LSIZE];
        local REAL _jay[LSIZE];
        local REAL _jaz[LSIZE];
        for (; (j + LSIZE - 1) < nj; j += LSIZE) {
            REAL jm = __jm[j + lid];
            REAL jvx = __jvx[j + lid];
            REAL jvy = __jvy[j + lid];
            REAL jvz = __jvz[j + lid];
            REAL jax = __jax[j + lid];
            REAL jay = __jay[j + lid];
            REAL jaz = __jaz[j + lid];
            barrier(CLK_LOCAL_MEM_FENCE);
            _jm[lid] = jm;
            _jvx[lid] = jvx;
            _jvy[lid] = jvy;
            _jvz[lid] = jvz;
            _jax[lid] = jax;
            _jay[lid] = jay;
            _jaz[lid] = jaz;
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll UNROLL
            for (UINT k = 0; k < LSIZE; ++k) {
                jm = _jm[k];
                jvx = _jvx[k];
                jvy = _jvy[k];
                jvz = _jvz[k];
                jax = _jax[k];
                jay = _jay[k];
                jaz = _jaz[k];
                nreg_Vkernel_core(
                    dt,
                    im, ivx, ivy, ivz, iax, iay, iaz,
                    jm, jvx, jvy, jvz, jax, jay, jaz,
                    &idvx, &idvy, &idvz, &ik);
            }
        }
        #endif

        for (; j < nj; ++j) {
            REAL jm = __jm[j];
            REAL jvx = __jvx[j];
            REAL jvy = __jvy[j];
            REAL jvz = __jvz[j];
            REAL jax = __jax[j];
            REAL jay = __jay[j];
            REAL jaz = __jaz[j];
            nreg_Vkernel_core(
                dt,
                im, ivx, ivy, ivz, iax, iay, iaz,
                jm, jvx, jvy, jvz, jax, jay, jaz,
                &idvx, &idvy, &idvz, &ik);
        }

        __idvx[gid] = idvx;
        __idvy[gid] = idvy;
        __idvz[gid] = idvz;
        __ik[gid] = im * ik;
    }
}


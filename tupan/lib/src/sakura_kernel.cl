#include "sakura_kernel_common.h"


__attribute__((reqd_work_group_size(LSIZE, 1, 1)))
kernel void
sakura_kernel(
    UINT const ni,
    global REAL const __im[restrict],
    global REAL const __irx[restrict],
    global REAL const __iry[restrict],
    global REAL const __irz[restrict],
    global REAL const __ie2[restrict],
    global REAL const __ivx[restrict],
    global REAL const __ivy[restrict],
    global REAL const __ivz[restrict],
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
    INT const flag,
    global REAL __idrx[restrict],
    global REAL __idry[restrict],
    global REAL __idrz[restrict],
    global REAL __idvx[restrict],
    global REAL __idvy[restrict],
    global REAL __idvz[restrict])
{
    for (UINT i = LSIZE * get_group_id(0) + get_global_offset(0);
              i < ni;
              i += LSIZE * get_num_groups(0)) {
        UINT lid = get_local_id(0);
        UINT gid = ((i + lid) < ni) ? (i + lid):(0);

        REAL im = __im[gid];
        REAL irx = __irx[gid];
        REAL iry = __iry[gid];
        REAL irz = __irz[gid];
        REAL ie2 = __ie2[gid];
        REAL ivx = __ivx[gid];
        REAL ivy = __ivy[gid];
        REAL ivz = __ivz[gid];

        REAL idrx = (REAL)(0);
        REAL idry = (REAL)(0);
        REAL idrz = (REAL)(0);
        REAL idvx = (REAL)(0);
        REAL idvy = (REAL)(0);
        REAL idvz = (REAL)(0);

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
                sakura_kernel_core(
                    dt, flag,
                    im, irx, iry, irz, ie2, ivx, ivy, ivz,
                    jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                    &idrx, &idry, &idrz, &idvx, &idvy, &idvz);
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
            sakura_kernel_core(
                dt, flag,
                im, irx, iry, irz, ie2, ivx, ivy, ivz,
                jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                &idrx, &idry, &idrz, &idvx, &idvy, &idvz);
        }

        __idrx[gid] = idrx;
        __idry[gid] = idry;
        __idrz[gid] = idrz;
        __idvx[gid] = idvx;
        __idvy[gid] = idvy;
        __idvz[gid] = idvz;
    }
}


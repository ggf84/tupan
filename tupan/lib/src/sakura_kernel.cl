#include "sakura_kernel_common.h"


kernel
__attribute__((reqd_work_group_size(LSIZE, 1, 1)))
void sakura_kernel(
    const UINT ni,
    global const REAL * restrict __im,
    global const REAL * restrict __irx,
    global const REAL * restrict __iry,
    global const REAL * restrict __irz,
    global const REAL * restrict __ie2,
    global const REAL * restrict __ivx,
    global const REAL * restrict __ivy,
    global const REAL * restrict __ivz,
    const UINT nj,
    global const REAL * restrict __jm,
    global const REAL * restrict __jrx,
    global const REAL * restrict __jry,
    global const REAL * restrict __jrz,
    global const REAL * restrict __je2,
    global const REAL * restrict __jvx,
    global const REAL * restrict __jvy,
    global const REAL * restrict __jvz,
    const REAL dt,
    const INT flag,
    global REAL * restrict __idrx,
    global REAL * restrict __idry,
    global REAL * restrict __idrz,
    global REAL * restrict __idvx,
    global REAL * restrict __idvy,
    global REAL * restrict __idvz)
{
    for (UINT i = LSIZE * get_group_id(0);
         1 * i < ni; i += LSIZE * get_num_groups(0)) {
        UINT lid = get_local_id(0);
        UINT gid = i + lid;
        gid = ((1 * gid) < ni) ? (gid):(0);

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
        for (; (j + LSIZE - 1) < nj; j += LSIZE) {
            barrier(CLK_LOCAL_MEM_FENCE);
            local REAL _jm[LSIZE];
            local REAL _jrx[LSIZE];
            local REAL _jry[LSIZE];
            local REAL _jrz[LSIZE];
            local REAL _je2[LSIZE];
            local REAL _jvx[LSIZE];
            local REAL _jvy[LSIZE];
            local REAL _jvz[LSIZE];
            _jm[lid] = __jm[j + lid];
            _jrx[lid] = __jrx[j + lid];
            _jry[lid] = __jry[j + lid];
            _jrz[lid] = __jrz[j + lid];
            _je2[lid] = __je2[j + lid];
            _jvx[lid] = __jvx[j + lid];
            _jvy[lid] = __jvy[j + lid];
            _jvz[lid] = __jvz[j + lid];
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll UNROLL
            for (UINT k = 0; k < LSIZE; ++k) {
                sakura_kernel_core(
                    dt, flag,
                    im, irx, iry, irz,
                    ie2, ivx, ivy, ivz,
                    _jm[k], _jrx[k], _jry[k], _jrz[k],
                    _je2[k], _jvx[k], _jvy[k], _jvz[k],
                    &idrx, &idry, &idrz,
                    &idvx, &idvy, &idvz);
            }
        }
        #endif

        #pragma unroll UNROLL
        for (; j < nj; ++j) {
            sakura_kernel_core(
                dt, flag,
                im, irx, iry, irz,
                ie2, ivx, ivy, ivz,
                __jm[j], __jrx[j], __jry[j], __jrz[j],
                __je2[j], __jvx[j], __jvy[j], __jvz[j],
                &idrx, &idry, &idrz,
                &idvx, &idvy, &idvz);
        }

        __idrx[gid] = idrx;
        __idry[gid] = idry;
        __idrz[gid] = idrz;
        __idvx[gid] = idvx;
        __idvy[gid] = idvy;
        __idvz[gid] = idvz;
    }
}


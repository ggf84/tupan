#include "sakura_kernel_common.h"


__kernel
__attribute__((reqd_work_group_size(LSIZE, 1, 1)))
void sakura_kernel(
    const UINT ni,
    __global const REAL * restrict _im,
    __global const REAL * restrict _irx,
    __global const REAL * restrict _iry,
    __global const REAL * restrict _irz,
    __global const REAL * restrict _ie2,
    __global const REAL * restrict _ivx,
    __global const REAL * restrict _ivy,
    __global const REAL * restrict _ivz,
    const UINT nj,
    __global const REAL * restrict _jm,
    __global const REAL * restrict _jrx,
    __global const REAL * restrict _jry,
    __global const REAL * restrict _jrz,
    __global const REAL * restrict _je2,
    __global const REAL * restrict _jvx,
    __global const REAL * restrict _jvy,
    __global const REAL * restrict _jvz,
    const REAL dt,
    const INT flag,
    __global REAL * restrict _idrx,
    __global REAL * restrict _idry,
    __global REAL * restrict _idrz,
    __global REAL * restrict _idvx,
    __global REAL * restrict _idvy,
    __global REAL * restrict _idvz)
{
    for (UINT i = LSIZE * get_group_id(0);
         1 * i < ni; i += LSIZE * get_num_groups(0)) {
        UINT lid = get_local_id(0);
        UINT gid = i + lid;
        gid = ((1 * gid) < ni) ? (gid):(0);

        REAL im = _im[gid];
        REAL irx = _irx[gid];
        REAL iry = _iry[gid];
        REAL irz = _irz[gid];
        REAL ie2 = _ie2[gid];
        REAL ivx = _ivx[gid];
        REAL ivy = _ivy[gid];
        REAL ivz = _ivz[gid];

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
            __local REAL __jm[LSIZE];
            __local REAL __jrx[LSIZE];
            __local REAL __jry[LSIZE];
            __local REAL __jrz[LSIZE];
            __local REAL __je2[LSIZE];
            __local REAL __jvx[LSIZE];
            __local REAL __jvy[LSIZE];
            __local REAL __jvz[LSIZE];
            __jm[lid] = _jm[j + lid];
            __jrx[lid] = _jrx[j + lid];
            __jry[lid] = _jry[j + lid];
            __jrz[lid] = _jrz[j + lid];
            __je2[lid] = _je2[j + lid];
            __jvx[lid] = _jvx[j + lid];
            __jvy[lid] = _jvy[j + lid];
            __jvz[lid] = _jvz[j + lid];
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll UNROLL
            for (UINT k = 0; k < LSIZE; ++k) {
                sakura_kernel_core(
                    dt, flag,
                    im, irx, iry, irz,
                    ie2, ivx, ivy, ivz,
                    __jm[k], __jrx[k], __jry[k], __jrz[k],
                    __je2[k], __jvx[k], __jvy[k], __jvz[k],
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
                _jm[j], _jrx[j], _jry[j], _jrz[j],
                _je2[j], _jvx[j], _jvy[j], _jvz[j],
                &idrx, &idry, &idrz,
                &idvx, &idvy, &idvz);
        }

        _idrx[gid] = idrx;
        _idry[gid] = idry;
        _idrz[gid] = idrz;
        _idvx[gid] = idvx;
        _idvy[gid] = idvy;
        _idvz[gid] = idvz;
    }
}


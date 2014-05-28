#include "sakura_kernel_common.h"


__kernel void sakura_kernel(
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
    UINT lid = get_local_id(0);
    UINT lsize = get_local_size(0);
//    UINT i = VW * lsize * get_group_id(0);
    UINT i = 1 * lsize * get_group_id(0);

//    UINT mask = (i + VW * lid) < ni;
    UINT mask = (i + 1 * lid) < ni;
    mask *= lid;

    REAL im = vload1(mask, _im + i);
    REAL irx = vload1(mask, _irx + i);
    REAL iry = vload1(mask, _iry + i);
    REAL irz = vload1(mask, _irz + i);
    REAL ie2 = vload1(mask, _ie2 + i);
    REAL ivx = vload1(mask, _ivx + i);
    REAL ivy = vload1(mask, _ivy + i);
    REAL ivz = vload1(mask, _ivz + i);

    REAL idrx = (REAL)(0);
    REAL idry = (REAL)(0);
    REAL idrz = (REAL)(0);
    REAL idvx = (REAL)(0);
    REAL idvy = (REAL)(0);
    REAL idvz = (REAL)(0);

    UINT j = 0;

    #ifdef FAST_LOCAL_MEM
        __local REAL __jm[LSIZE];
        __local REAL __jrx[LSIZE];
        __local REAL __jry[LSIZE];
        __local REAL __jrz[LSIZE];
        __local REAL __je2[LSIZE];
        __local REAL __jvx[LSIZE];
        __local REAL __jvy[LSIZE];
        __local REAL __jvz[LSIZE];
        for (; (j + lsize - 1) < nj; j += lsize) {
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
            for (UINT k = 0; k < lsize; ++k) {
                sakura_kernel_core(
                    dt, flag,
                    im, irx, iry, irz,
                    ie2, ivx, ivy, ivz,
                    __jm[k], __jrx[k], __jry[k], __jrz[k],
                    __je2[k], __jvx[k], __jvy[k], __jvz[k],
                    &idrx, &idry, &idrz,
                    &idvx, &idvy, &idvz);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
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

    vstore1(idrx, mask, _idrx + i);
    vstore1(idry, mask, _idry + i);
    vstore1(idrz, mask, _idrz + i);
    vstore1(idvx, mask, _idvx + i);
    vstore1(idvy, mask, _idvy + i);
    vstore1(idvz, mask, _idvz + i);
}


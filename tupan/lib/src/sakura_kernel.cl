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
//    UINT gid = get_global_id(0) * WPT * VW;
    UINT gid = get_global_id(0) * WPT * 1;

    UINT imask[WPT];

    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
//        imask[i] = (VW * i + gid) < ni;
        imask[i] = (1 * i + gid) < ni;

    REAL im[WPT], irx[WPT], iry[WPT], irz[WPT],
         ie2[WPT], ivx[WPT], ivy[WPT], ivz[WPT];

    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            im[i] = vload1(i, _im + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            irx[i] = vload1(i, _irx + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            iry[i] = vload1(i, _iry + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            irz[i] = vload1(i, _irz + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            ie2[i] = vload1(i, _ie2 + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            ivx[i] = vload1(i, _ivx + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            ivy[i] = vload1(i, _ivy + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            ivz[i] = vload1(i, _ivz + gid);

    REAL idrx[WPT], idry[WPT], idrz[WPT],
         idvx[WPT], idvy[WPT], idvz[WPT];

    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            idrx[i] = (REAL)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            idry[i] = (REAL)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            idrz[i] = (REAL)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            idvx[i] = (REAL)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            idvy[i] = (REAL)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            idvz[i] = (REAL)(0);

#ifdef FAST_LOCAL_MEM
    __local REAL __jm[LSIZE];
    __local REAL __jrx[LSIZE];
    __local REAL __jry[LSIZE];
    __local REAL __jrz[LSIZE];
    __local REAL __je2[LSIZE];
    __local REAL __jvx[LSIZE];
    __local REAL __jvy[LSIZE];
    __local REAL __jvz[LSIZE];
    UINT j = 0;
    UINT lid = get_local_id(0);
    for (UINT stride = get_local_size(0); stride > 0; stride /= 2) {
        INT mask = lid < stride;
        for (; (j + stride - 1) < nj; j += stride) {
            if (mask) {
                __jm[lid] = _jm[j + lid];
                __jrx[lid] = _jrx[j + lid];
                __jry[lid] = _jry[j + lid];
                __jrz[lid] = _jrz[j + lid];
                __je2[lid] = _je2[j + lid];
                __jvx[lid] = _jvx[j + lid];
                __jvy[lid] = _jvy[j + lid];
                __jvz[lid] = _jvz[j + lid];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll UNROLL
            for (UINT k = 0; k < stride; ++k) {
                #pragma unroll
                for (UINT i = 0; i < WPT; ++i) {
                    sakura_kernel_core(dt, flag,
                                       im[i], irx[i], iry[i], irz[i],
                                       ie2[i], ivx[i], ivy[i], ivz[i],
                                       __jm[k], __jrx[k], __jry[k], __jrz[k],
                                       __je2[k], __jvx[k], __jvy[k], __jvz[k],
                                       &idrx[i], &idry[i], &idrz[i],
                                       &idvx[i], &idvy[i], &idvz[i]);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
#else
    #pragma unroll UNROLL
    for (UINT j = 0; j < nj; ++j) {
        #pragma unroll
        for (UINT i = 0; i < WPT; ++i) {
            sakura_kernel_core(dt, flag,
                               im[i], irx[i], iry[i], irz[i],
                               ie2[i], ivx[i], ivy[i], ivz[i],
                               _jm[j], _jrx[j], _jry[j], _jrz[j],
                               _je2[j], _jvx[j], _jvy[j], _jvz[j],
                               &idrx[i], &idry[i], &idrz[i],
                               &idvx[i], &idvy[i], &idvz[i]);
        }
    }
#endif

    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            vstore1(idrx[i], i, _idrx + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            vstore1(idry[i], i, _idry + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            vstore1(idrz[i], i, _idrz + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            vstore1(idvx[i], i, _idvx + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            vstore1(idvy[i], i, _idvy + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            vstore1(idvz[i], i, _idvz + gid);
}


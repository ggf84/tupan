#include "snap_crackle_kernel_common.h"


__kernel void snap_crackle_kernel(
    const UINT ni,
    __global const REAL * restrict _im,
    __global const REAL * restrict _irx,
    __global const REAL * restrict _iry,
    __global const REAL * restrict _irz,
    __global const REAL * restrict _ie2,
    __global const REAL * restrict _ivx,
    __global const REAL * restrict _ivy,
    __global const REAL * restrict _ivz,
    __global const REAL * restrict _iax,
    __global const REAL * restrict _iay,
    __global const REAL * restrict _iaz,
    __global const REAL * restrict _ijx,
    __global const REAL * restrict _ijy,
    __global const REAL * restrict _ijz,
    const UINT nj,
    __global const REAL * restrict _jm,
    __global const REAL * restrict _jrx,
    __global const REAL * restrict _jry,
    __global const REAL * restrict _jrz,
    __global const REAL * restrict _je2,
    __global const REAL * restrict _jvx,
    __global const REAL * restrict _jvy,
    __global const REAL * restrict _jvz,
    __global const REAL * restrict _jax,
    __global const REAL * restrict _jay,
    __global const REAL * restrict _jaz,
    __global const REAL * restrict _jjx,
    __global const REAL * restrict _jjy,
    __global const REAL * restrict _jjz,
    __global REAL * restrict _isx,
    __global REAL * restrict _isy,
    __global REAL * restrict _isz,
    __global REAL * restrict _icx,
    __global REAL * restrict _icy,
    __global REAL * restrict _icz)
{
    UINT gid = get_global_id(0);
    gid = min(WPT * VW * gid, (ni - WPT * VW));

    REALn im[WPT], irx[WPT], iry[WPT], irz[WPT],
          ie2[WPT], ivx[WPT], ivy[WPT], ivz[WPT],
          iax[WPT], iay[WPT], iaz[WPT],
          ijx[WPT], ijy[WPT], ijz[WPT];

    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) im[i] = vloadn(i, _im + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) irx[i] = vloadn(i, _irx + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) iry[i] = vloadn(i, _iry + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) irz[i] = vloadn(i, _irz + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) ie2[i] = vloadn(i, _ie2 + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) ivx[i] = vloadn(i, _ivx + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) ivy[i] = vloadn(i, _ivy + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) ivz[i] = vloadn(i, _ivz + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) iax[i] = vloadn(i, _iax + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) iay[i] = vloadn(i, _iay + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) iaz[i] = vloadn(i, _iaz + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) ijx[i] = vloadn(i, _ijx + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) ijy[i] = vloadn(i, _ijy + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) ijz[i] = vloadn(i, _ijz + gid);

    REALn isx[WPT], isy[WPT], isz[WPT],
          icx[WPT], icy[WPT], icz[WPT];

    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) isx[i] = (REALn)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) isy[i] = (REALn)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) isz[i] = (REALn)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) icx[i] = (REALn)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) icy[i] = (REALn)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) icz[i] = (REALn)(0);

#ifdef FAST_LOCAL_MEM
    __local REAL __jm[LSIZE];
    __local REAL __jrx[LSIZE];
    __local REAL __jry[LSIZE];
    __local REAL __jrz[LSIZE];
    __local REAL __je2[LSIZE];
    __local REAL __jvx[LSIZE];
    __local REAL __jvy[LSIZE];
    __local REAL __jvz[LSIZE];
    __local REAL __jax[LSIZE];
    __local REAL __jay[LSIZE];
    __local REAL __jaz[LSIZE];
    __local REAL __jjx[LSIZE];
    __local REAL __jjy[LSIZE];
    __local REAL __jjz[LSIZE];
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
                __jax[lid] = _jax[j + lid];
                __jay[lid] = _jay[j + lid];
                __jaz[lid] = _jaz[j + lid];
                __jjx[lid] = _jjx[j + lid];
                __jjy[lid] = _jjy[j + lid];
                __jjz[lid] = _jjz[j + lid];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll UNROLL
            for (UINT k = 0; k < stride; ++k) {
                #pragma unroll
                for (UINT i = 0; i < WPT; ++i) {
                    snap_crackle_kernel_core(im[i], irx[i], iry[i], irz[i],
                                             ie2[i], ivx[i], ivy[i], ivz[i],
                                             iax[i], iay[i], iaz[i],
                                             ijx[i], ijy[i], ijz[i],
                                             __jm[k], __jrx[k], __jry[k], __jrz[k],
                                             __je2[k], __jvx[k], __jvy[k], __jvz[k],
                                             __jax[k], __jay[k], __jaz[k],
                                             __jjx[k], __jjy[k], __jjz[k],
                                             &isx[i], &isy[i], &isz[i],
                                             &icx[i], &icy[i], &icz[i]);
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
            snap_crackle_kernel_core(im[i], irx[i], iry[i], irz[i],
                                     ie2[i], ivx[i], ivy[i], ivz[i],
                                     iax[i], iay[i], iaz[i],
                                     ijx[i], ijy[i], ijz[i],
                                     _jm[j], _jrx[j], _jry[j], _jrz[j],
                                     _je2[j], _jvx[j], _jvy[j], _jvz[j],
                                     _jax[j], _jay[j], _jaz[j],
                                     _jjx[j], _jjy[j], _jjz[j],
                                     &isx[i], &isy[i], &isz[i],
                                     &icx[i], &icy[i], &icz[i]);
        }
    }
#endif

    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) vstoren(isx[i], i, _isx + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) vstoren(isy[i], i, _isy + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) vstoren(isz[i], i, _isz + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) vstoren(icx[i], i, _icx + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) vstoren(icy[i], i, _icy + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) vstoren(icz[i], i, _icz + gid);
}


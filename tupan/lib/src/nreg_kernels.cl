#include "nreg_kernels_common.h"


__kernel void nreg_Xkernel(
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
    __global REAL * restrict _idrx,
    __global REAL * restrict _idry,
    __global REAL * restrict _idrz,
    __global REAL * restrict _iax,
    __global REAL * restrict _iay,
    __global REAL * restrict _iaz,
    __global REAL * restrict _iu)
{
    UINT gid = get_global_id(0);
    gid = min(WPT * VW * gid, (ni - WPT * VW));

    REALn im[WPT], irx[WPT], iry[WPT], irz[WPT],
          ie2[WPT], ivx[WPT], ivy[WPT], ivz[WPT];

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

    REALn idrx[WPT], idry[WPT], idrz[WPT],
          iax[WPT], iay[WPT], iaz[WPT], iu[WPT];

    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) idrx[i] = (REALn)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) idry[i] = (REALn)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) idrz[i] = (REALn)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) iax[i] = (REALn)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) iay[i] = (REALn)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) iaz[i] = (REALn)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) iu[i] = (REALn)(0);

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
    UINT stride = min((UINT)(get_local_size(0)), (UINT)(LSIZE));
    #pragma unroll 4
    for (; stride > 0; stride /= 2) {
        UINT lid = get_local_id(0) % stride;
        for (; (j + stride - 1) < nj; j += stride) {
            barrier(CLK_LOCAL_MEM_FENCE);
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
            for (UINT k = 0; k < stride; ++k) {
                #pragma unroll
                for (UINT i = 0; i < WPT; ++i) {
                    nreg_Xkernel_core(dt,
                                      im[i], irx[i], iry[i], irz[i],
                                      ie2[i], ivx[i], ivy[i], ivz[i],
                                      __jm[k], __jrx[k], __jry[k], __jrz[k],
                                      __je2[k], __jvx[k], __jvy[k], __jvz[k],
                                      &idrx[i], &idry[i], &idrz[i],
                                      &iax[i], &iay[i], &iaz[i], &iu[i]);
                }
            }
        }
    }
#else
    for (UINT j = 0; j < nj; ++j) {
        #pragma unroll
        for (UINT i = 0; i < WPT; ++i) {
            nreg_Xkernel_core(dt,
                              im[i], irx[i], iry[i], irz[i],
                              ie2[i], ivx[i], ivy[i], ivz[i],
                              _jm[j], _jrx[j], _jry[j], _jrz[j],
                              _je2[j], _jvx[j], _jvy[j], _jvz[j],
                              &idrx[i], &idry[i], &idrz[i],
                              &iax[i], &iay[i], &iaz[i], &iu[i]);
        }
    }
#endif

    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) vstoren(idrx[i], i, _idrx + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) vstoren(idry[i], i, _idry + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) vstoren(idrz[i], i, _idrz + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) vstoren(iax[i], i, _iax + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) vstoren(iay[i], i, _iay + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) vstoren(iaz[i], i, _iaz + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) vstoren(im[i] * iu[i], i, _iu + gid);
}


__kernel void nreg_Vkernel(
    const UINT ni,
    __global const REAL * restrict _im,
    __global const REAL * restrict _ivx,
    __global const REAL * restrict _ivy,
    __global const REAL * restrict _ivz,
    __global const REAL * restrict _iax,
    __global const REAL * restrict _iay,
    __global const REAL * restrict _iaz,
    const UINT nj,
    __global const REAL * restrict _jm,
    __global const REAL * restrict _jvx,
    __global const REAL * restrict _jvy,
    __global const REAL * restrict _jvz,
    __global const REAL * restrict _jax,
    __global const REAL * restrict _jay,
    __global const REAL * restrict _jaz,
    const REAL dt,
    __global REAL * restrict _idvx,
    __global REAL * restrict _idvy,
    __global REAL * restrict _idvz,
    __global REAL * restrict _ik)
{
    UINT gid = get_global_id(0);
    gid = min(WPT * VW * gid, (ni - WPT * VW));

    REALn im[WPT], ivx[WPT], ivy[WPT], ivz[WPT],
          iax[WPT], iay[WPT], iaz[WPT];

    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) im[i] = vloadn(i, _im + gid);
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

    REALn idvx[WPT], idvy[WPT], idvz[WPT], ik[WPT];

    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) idvx[i] = (REALn)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) idvy[i] = (REALn)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) idvz[i] = (REALn)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) ik[i] = (REALn)(0);

#ifdef FAST_LOCAL_MEM
    __local REAL __jm[LSIZE];
    __local REAL __jvx[LSIZE];
    __local REAL __jvy[LSIZE];
    __local REAL __jvz[LSIZE];
    __local REAL __jax[LSIZE];
    __local REAL __jay[LSIZE];
    __local REAL __jaz[LSIZE];
    UINT j = 0;
    UINT stride = min((UINT)(get_local_size(0)), (UINT)(LSIZE));
    #pragma unroll 4
    for (; stride > 0; stride /= 2) {
        UINT lid = get_local_id(0) % stride;
        for (; (j + stride - 1) < nj; j += stride) {
            barrier(CLK_LOCAL_MEM_FENCE);
            __jm[lid] = _jm[j + lid];
            __jvx[lid] = _jvx[j + lid];
            __jvy[lid] = _jvy[j + lid];
            __jvz[lid] = _jvz[j + lid];
            __jax[lid] = _jax[j + lid];
            __jay[lid] = _jay[j + lid];
            __jaz[lid] = _jaz[j + lid];
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll UNROLL
            for (UINT k = 0; k < stride; ++k) {
                #pragma unroll
                for (UINT i = 0; i < WPT; ++i) {
                    nreg_Vkernel_core(dt,
                                      im[i], ivx[i], ivy[i], ivz[i],
                                      iax[i], iay[i], iaz[i],
                                      __jm[k], __jvx[k], __jvy[k], __jvz[k],
                                      __jax[k], __jay[k], __jaz[k],
                                      &idvx[i], &idvy[i], &idvz[i], &ik[i]);
                }
            }
        }
    }
#else
    for (UINT j = 0; j < nj; ++j) {
        #pragma unroll
        for (UINT i = 0; i < WPT; ++i) {
            nreg_Vkernel_core(dt,
                              im[i], ivx[i], ivy[i], ivz[i],
                              iax[i], iay[i], iaz[i],
                              _jm[j], _jvx[j], _jvy[j], _jvz[j],
                              _jax[j], _jay[j], _jaz[j],
                              &idvx[i], &idvy[i], &idvz[i], &ik[i]);
        }
    }
#endif

    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) vstoren(idvx[i], i, _idvx + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) vstoren(idvy[i], i, _idvy + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) vstoren(idvz[i], i, _idvz + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) vstoren(im[i] * ik[i], i, _ik + gid);
}


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
    UINT lsize = get_local_size(0);
    UINT lid = get_local_id(0);
//    UINT gid = VECTOR_WIDTH * get_global_id(0);
//    gid = min(gid, (ni - VECTOR_WIDTH));
    UINT gid = get_global_id(0);
    gid = min(gid, (ni - 1));

    REAL im = vload1(0, _im + gid);
    REAL irx = vload1(0, _irx + gid);
    REAL iry = vload1(0, _iry + gid);
    REAL irz = vload1(0, _irz + gid);
    REAL ie2 = vload1(0, _ie2 + gid);
    REAL ivx = vload1(0, _ivx + gid);
    REAL ivy = vload1(0, _ivy + gid);
    REAL ivz = vload1(0, _ivz + gid);
    REAL idrx = (REAL)(0);
    REAL idry = (REAL)(0);
    REAL idrz = (REAL)(0);
    REAL idvx = (REAL)(0);
    REAL idvy = (REAL)(0);
    REAL idvz = (REAL)(0);

#ifdef FAST_LOCAL_MEM
    __local REAL __jm[LSIZE];
    __local REAL __jrx[LSIZE];
    __local REAL __jry[LSIZE];
    __local REAL __jrz[LSIZE];
    __local REAL __je2[LSIZE];
    __local REAL __jvx[LSIZE];
    __local REAL __jvy[LSIZE];
    __local REAL __jvz[LSIZE];
    for (UINT j = 0; j < nj; j += lsize) {
        lid = min(lid, (nj - j) - 1);
        lsize = min(lsize, (nj - j));
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
        for (UINT k = 0; k < lsize; ++k) {
            sakura_kernel_core(dt, flag,
                               im, irx, iry, irz,
                               ie2, ivx, ivy, ivz,
                               __jm[k], __jrx[k], __jry[k], __jrz[k],
                               __je2[k], __jvx[k], __jvy[k], __jvz[k],
                               &idrx, &idry, &idrz,
                               &idvx, &idvy, &idvz);
        }
    }
#else
    for (UINT j = 0; j < nj; ++j) {
        sakura_kernel_core(dt, flag,
                           im, irx, iry, irz,
                           ie2, ivx, ivy, ivz,
                           _jm[j], _jrx[j], _jry[j], _jrz[j],
                           _je2[j], _jvx[j], _jvy[j], _jvz[j],
                           &idrx, &idry, &idrz,
                           &idvx, &idvy, &idvz);
    }
#endif

    vstore1(idrx, 0, _idrx + gid);
    vstore1(idry, 0, _idry + gid);
    vstore1(idrz, 0, _idrz + gid);
    vstore1(idvx, 0, _idvx + gid);
    vstore1(idvy, 0, _idvy + gid);
    vstore1(idvz, 0, _idvz + gid);
}


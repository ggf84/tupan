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
    UINT lsize = (get_local_size(0) + UNROLL - 1) / UNROLL;
    lsize = min(lsize, (UINT)(LSIZE));
    UINT lid = get_local_id(0) % lsize;
    UINT gid = get_global_id(0);

    REAL im = vload1(gid, _im);
    REAL irx = vload1(gid, _irx);
    REAL iry = vload1(gid, _iry);
    REAL irz = vload1(gid, _irz);
    REAL ie2 = vload1(gid, _ie2);
    REAL ivx = vload1(gid, _ivx);
    REAL ivy = vload1(gid, _ivy);
    REAL ivz = vload1(gid, _ivz);
    REAL idrx = (REAL)(0);
    REAL idry = (REAL)(0);
    REAL idrz = (REAL)(0);
    REAL idvx = (REAL)(0);
    REAL idvy = (REAL)(0);
    REAL idvz = (REAL)(0);

    UINT j = 0;
    __local concat(REAL, UNROLL) __jm[LSIZE];
    __local concat(REAL, UNROLL) __jrx[LSIZE];
    __local concat(REAL, UNROLL) __jry[LSIZE];
    __local concat(REAL, UNROLL) __jrz[LSIZE];
    __local concat(REAL, UNROLL) __je2[LSIZE];
    __local concat(REAL, UNROLL) __jvx[LSIZE];
    __local concat(REAL, UNROLL) __jvy[LSIZE];
    __local concat(REAL, UNROLL) __jvz[LSIZE];
    for (; (j + UNROLL * lsize) < nj; j += UNROLL * lsize) {
        concat(REAL, UNROLL) jm = concat(vload, UNROLL)(lid, _jm + j);
        concat(REAL, UNROLL) jrx = concat(vload, UNROLL)(lid, _jrx + j);
        concat(REAL, UNROLL) jry = concat(vload, UNROLL)(lid, _jry + j);
        concat(REAL, UNROLL) jrz = concat(vload, UNROLL)(lid, _jrz + j);
        concat(REAL, UNROLL) je2 = concat(vload, UNROLL)(lid, _je2 + j);
        concat(REAL, UNROLL) jvx = concat(vload, UNROLL)(lid, _jvx + j);
        concat(REAL, UNROLL) jvy = concat(vload, UNROLL)(lid, _jvy + j);
        concat(REAL, UNROLL) jvz = concat(vload, UNROLL)(lid, _jvz + j);
        barrier(CLK_LOCAL_MEM_FENCE);
        __jm[lid] = jm;
        __jrx[lid] = jrx;
        __jry[lid] = jry;
        __jrz[lid] = jrz;
        __je2[lid] = je2;
        __jvx[lid] = jvx;
        __jvy[lid] = jvy;
        __jvz[lid] = jvz;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (UINT k = 0; k < lsize; ++k) {
            jm = __jm[k];
            jrx = __jrx[k];
            jry = __jry[k];
            jrz = __jrz[k];
            je2 = __je2[k];
            jvx = __jvx[k];
            jvy = __jvy[k];
            jvz = __jvz[k];
            #if UNROLL == 1
                sakura_kernel_core(dt, flag,
                                   im, irx, iry, irz,
                                   ie2, ivx, ivy, ivz,
                                   jm, jrx, jry, jrz,
                                   je2, jvx, jvy, jvz,
                                   &idrx, &idry, &idrz,
                                   &idvx, &idvy, &idvz);
            #else
                sakura_kernel_core(dt, flag,
                                   im, irx, iry, irz,
                                   ie2, ivx, ivy, ivz,
                                   jm.s0, jrx.s0, jry.s0, jrz.s0,
                                   je2.s0, jvx.s0, jvy.s0, jvz.s0,
                                   &idrx, &idry, &idrz,
                                   &idvx, &idvy, &idvz);
                #pragma unroll
                for (UINT l = 1; l < UNROLL; ++l) {
                    jm = shuffle(jm, MASK);
                    jrx = shuffle(jrx, MASK);
                    jry = shuffle(jry, MASK);
                    jrz = shuffle(jrz, MASK);
                    je2 = shuffle(je2, MASK);
                    jvx = shuffle(jvx, MASK);
                    jvy = shuffle(jvy, MASK);
                    jvz = shuffle(jvz, MASK);
                    sakura_kernel_core(dt, flag,
                                       im, irx, iry, irz,
                                       ie2, ivx, ivy, ivz,
                                       jm.s0, jrx.s0, jry.s0, jrz.s0,
                                       je2.s0, jvx.s0, jvy.s0, jvz.s0,
                                       &idrx, &idry, &idrz,
                                       &idvx, &idvy, &idvz);
                }
            #endif
        }
    }
    for (; j < nj; ++j) {
        sakura_kernel_core(dt, flag,
                           im, irx, iry, irz,
                           ie2, ivx, ivy, ivz,
                           _jm[j], _jrx[j], _jry[j], _jrz[j],
                           _je2[j], _jvx[j], _jvy[j], _jvz[j],
                           &idrx, &idry, &idrz,
                           &idvx, &idvy, &idvz);
    }

    vstore1(idrx, gid, _idrx);
    vstore1(idry, gid, _idry);
    vstore1(idrz, gid, _idrz);
    vstore1(idvx, gid, _idvx);
    vstore1(idvy, gid, _idvy);
    vstore1(idvz, gid, _idvz);
}


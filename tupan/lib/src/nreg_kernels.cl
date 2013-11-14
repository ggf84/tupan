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
    gid = min(VECTOR_WIDTH * gid, (ni - VECTOR_WIDTH));

    REALn im = vloadn(0, _im + gid);
    REALn irx = vloadn(0, _irx + gid);
    REALn iry = vloadn(0, _iry + gid);
    REALn irz = vloadn(0, _irz + gid);
    REALn ie2 = vloadn(0, _ie2 + gid);
    REALn ivx = vloadn(0, _ivx + gid);
    REALn ivy = vloadn(0, _ivy + gid);
    REALn ivz = vloadn(0, _ivz + gid);
    REALn idrx = (REALn)(0);
    REALn idry = (REALn)(0);
    REALn idrz = (REALn)(0);
    REALn iax = (REALn)(0);
    REALn iay = (REALn)(0);
    REALn iaz = (REALn)(0);
    REALn iu = (REALn)(0);

    UINT j = 0;
#ifdef FAST_LOCAL_MEM
    __local REAL4 __jm[LSIZE];
    __local REAL4 __jrx[LSIZE];
    __local REAL4 __jry[LSIZE];
    __local REAL4 __jrz[LSIZE];
    __local REAL4 __je2[LSIZE];
    __local REAL4 __jvx[LSIZE];
    __local REAL4 __jvy[LSIZE];
    __local REAL4 __jvz[LSIZE];
    UINT stride = min((UINT)(get_local_size(0)), (UINT)(LSIZE));
    #pragma unroll 4
    for (; stride > 0; stride /= 2) {
        UINT lid = get_local_id(0) % stride;
        for (; (j + 4 * stride - 1) < nj; j += 4 * stride) {
            REAL4 jm = vload4(lid, _jm + j);
            REAL4 jrx = vload4(lid, _jrx + j);
            REAL4 jry = vload4(lid, _jry + j);
            REAL4 jrz = vload4(lid, _jrz + j);
            REAL4 je2 = vload4(lid, _je2 + j);
            REAL4 jvx = vload4(lid, _jvx + j);
            REAL4 jvy = vload4(lid, _jvy + j);
            REAL4 jvz = vload4(lid, _jvz + j);
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
            #pragma unroll UNROLL
            for (UINT k = 0; k < stride; ++k) {
                jm = __jm[k];
                jrx = __jrx[k];
                jry = __jry[k];
                jrz = __jrz[k];
                je2 = __je2[k];
                jvx = __jvx[k];
                jvy = __jvy[k];
                jvz = __jvz[k];
                nreg_Xkernel_core(dt,
                                  im, irx, iry, irz,
                                  ie2, ivx, ivy, ivz,
                                  jm.s0, jrx.s0, jry.s0, jrz.s0,
                                  je2.s0, jvx.s0, jvy.s0, jvz.s0,
                                  &idrx, &idry, &idrz,
                                  &iax, &iay, &iaz, &iu);
                #pragma unroll
                for (UINT l = 1; l < 4; ++l) {
                    jm = jm.s1230;
                    jrx = jrx.s1230;
                    jry = jry.s1230;
                    jrz = jrz.s1230;
                    je2 = je2.s1230;
                    jvx = jvx.s1230;
                    jvy = jvy.s1230;
                    jvz = jvz.s1230;
                    nreg_Xkernel_core(dt,
                                      im, irx, iry, irz,
                                      ie2, ivx, ivy, ivz,
                                      jm.s0, jrx.s0, jry.s0, jrz.s0,
                                      je2.s0, jvx.s0, jvy.s0, jvz.s0,
                                      &idrx, &idry, &idrz,
                                      &iax, &iay, &iaz, &iu);
                }
            }
        }
    }
#endif
    for (; j < nj; ++j) {
        nreg_Xkernel_core(dt,
                          im, irx, iry, irz,
                          ie2, ivx, ivy, ivz,
                          _jm[j], _jrx[j], _jry[j], _jrz[j],
                          _je2[j], _jvx[j], _jvy[j], _jvz[j],
                          &idrx, &idry, &idrz,
                          &iax, &iay, &iaz, &iu);
    }

    vstoren(idrx, 0, _idrx + gid);
    vstoren(idry, 0, _idry + gid);
    vstoren(idrz, 0, _idrz + gid);
    vstoren(iax, 0, _iax + gid);
    vstoren(iay, 0, _iay + gid);
    vstoren(iaz, 0, _iaz + gid);
    vstoren(im * iu, 0, _iu + gid);
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
    gid = min(VECTOR_WIDTH * gid, (ni - VECTOR_WIDTH));

    REALn im = vloadn(0, _im + gid);
    REALn ivx = vloadn(0, _ivx + gid);
    REALn ivy = vloadn(0, _ivy + gid);
    REALn ivz = vloadn(0, _ivz + gid);
    REALn iax = vloadn(0, _iax + gid);
    REALn iay = vloadn(0, _iay + gid);
    REALn iaz = vloadn(0, _iaz + gid);
    REALn idvx = (REALn)(0);
    REALn idvy = (REALn)(0);
    REALn idvz = (REALn)(0);
    REALn ik = (REALn)(0);

    UINT j = 0;
#ifdef FAST_LOCAL_MEM
    __local REAL4 __jm[LSIZE];
    __local REAL4 __jvx[LSIZE];
    __local REAL4 __jvy[LSIZE];
    __local REAL4 __jvz[LSIZE];
    __local REAL4 __jax[LSIZE];
    __local REAL4 __jay[LSIZE];
    __local REAL4 __jaz[LSIZE];
    UINT stride = min((UINT)(get_local_size(0)), (UINT)(LSIZE));
    #pragma unroll 4
    for (; stride > 0; stride /= 2) {
        UINT lid = get_local_id(0) % stride;
        for (; (j + 4 * stride - 1) < nj; j += 4 * stride) {
            REAL4 jm = vload4(lid, _jm + j);
            REAL4 jvx = vload4(lid, _jvx + j);
            REAL4 jvy = vload4(lid, _jvy + j);
            REAL4 jvz = vload4(lid, _jvz + j);
            REAL4 jax = vload4(lid, _jax + j);
            REAL4 jay = vload4(lid, _jay + j);
            REAL4 jaz = vload4(lid, _jaz + j);
            barrier(CLK_LOCAL_MEM_FENCE);
            __jm[lid] = jm;
            __jvx[lid] = jvx;
            __jvy[lid] = jvy;
            __jvz[lid] = jvz;
            __jax[lid] = jax;
            __jay[lid] = jay;
            __jaz[lid] = jaz;
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll UNROLL
            for (UINT k = 0; k < stride; ++k) {
                jm = __jm[k];
                jvx = __jvx[k];
                jvy = __jvy[k];
                jvz = __jvz[k];
                jax = __jax[k];
                jay = __jay[k];
                jaz = __jaz[k];
                nreg_Vkernel_core(dt,
                                  im, ivx, ivy, ivz,
                                  iax, iay, iaz,
                                  jm.s0, jvx.s0, jvy.s0, jvz.s0,
                                  jax.s0, jay.s0, jaz.s0,
                                  &idvx, &idvy, &idvz, &ik);
                #pragma unroll
                for (UINT l = 1; l < 4; ++l) {
                    jm = jm.s1230;
                    jvx = jvx.s1230;
                    jvy = jvy.s1230;
                    jvz = jvz.s1230;
                    jax = jax.s1230;
                    jay = jay.s1230;
                    jaz = jaz.s1230;
                    nreg_Vkernel_core(dt,
                                      im, ivx, ivy, ivz,
                                      iax, iay, iaz,
                                      jm.s0, jvx.s0, jvy.s0, jvz.s0,
                                      jax.s0, jay.s0, jaz.s0,
                                      &idvx, &idvy, &idvz, &ik);
                }
            }
        }
    }
#endif
    for (; j < nj; ++j) {
        nreg_Vkernel_core(dt,
                          im, ivx, ivy, ivz,
                          iax, iay, iaz,
                          _jm[j], _jvx[j], _jvy[j], _jvz[j],
                          _jax[j], _jay[j], _jaz[j],
                          &idvx, &idvy, &idvz, &ik);
    }

    vstoren(idvx, 0, _idvx + gid);
    vstoren(idvy, 0, _idvy + gid);
    vstoren(idvz, 0, _idvz + gid);
    vstoren(im * ik, 0, _ik + gid);
}


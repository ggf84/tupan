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
    UINT lsize = (get_local_size(0) + UNROLL - 1) / UNROLL;
    lsize = min(lsize, (UINT)(LSIZE));
    UINT lid = get_local_id(0) % lsize;
    UINT gid = get_global_id(0);

    REALn im = vloadn(gid, _im);
    REALn irx = vloadn(gid, _irx);
    REALn iry = vloadn(gid, _iry);
    REALn irz = vloadn(gid, _irz);
    REALn ie2 = vloadn(gid, _ie2);
    REALn ivx = vloadn(gid, _ivx);
    REALn ivy = vloadn(gid, _ivy);
    REALn ivz = vloadn(gid, _ivz);
    REALn idrx = (REALn)(0);
    REALn idry = (REALn)(0);
    REALn idrz = (REALn)(0);
    REALn iax = (REALn)(0);
    REALn iay = (REALn)(0);
    REALn iaz = (REALn)(0);
    REALn iu = (REALn)(0);

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
                nreg_Xkernel_core(dt,
                                  im, irx, iry, irz,
                                  ie2, ivx, ivy, ivz,
                                  jm, jrx, jry, jrz,
                                  je2, jvx, jvy, jvz,
                                  &idrx, &idry, &idrz,
                                  &iax, &iay, &iaz, &iu);
            #else
                nreg_Xkernel_core(dt,
                                  im, irx, iry, irz,
                                  ie2, ivx, ivy, ivz,
                                  jm.s0, jrx.s0, jry.s0, jrz.s0,
                                  je2.s0, jvx.s0, jvy.s0, jvz.s0,
                                  &idrx, &idry, &idrz,
                                  &iax, &iay, &iaz, &iu);
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
                    nreg_Xkernel_core(dt,
                                      im, irx, iry, irz,
                                      ie2, ivx, ivy, ivz,
                                      jm.s0, jrx.s0, jry.s0, jrz.s0,
                                      je2.s0, jvx.s0, jvy.s0, jvz.s0,
                                      &idrx, &idry, &idrz,
                                      &iax, &iay, &iaz, &iu);
                }
            #endif
        }
    }
    for (; j < nj; ++j) {
        nreg_Xkernel_core(dt,
                          im, irx, iry, irz,
                          ie2, ivx, ivy, ivz,
                          _jm[j], _jrx[j], _jry[j], _jrz[j],
                          _je2[j], _jvx[j], _jvy[j], _jvz[j],
                          &idrx, &idry, &idrz,
                          &iax, &iay, &iaz, &iu);
    }

    vstoren(idrx, gid, _idrx);
    vstoren(idry, gid, _idry);
    vstoren(idrz, gid, _idrz);
    vstoren(iax, gid, _iax);
    vstoren(iay, gid, _iay);
    vstoren(iaz, gid, _iaz);
    vstoren(im * iu, gid, _iu);
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
    UINT lsize = (get_local_size(0) + UNROLL - 1) / UNROLL;
    lsize = min(lsize, (UINT)(LSIZE));
    UINT lid = get_local_id(0) % lsize;
    UINT gid = get_global_id(0);

    REALn im = vloadn(gid, _im);
    REALn ivx = vloadn(gid, _ivx);
    REALn ivy = vloadn(gid, _ivy);
    REALn ivz = vloadn(gid, _ivz);
    REALn iax = vloadn(gid, _iax);
    REALn iay = vloadn(gid, _iay);
    REALn iaz = vloadn(gid, _iaz);
    REALn idvx = (REALn)(0);
    REALn idvy = (REALn)(0);
    REALn idvz = (REALn)(0);
    REALn ik = (REALn)(0);

    UINT j = 0;
    __local concat(REAL, UNROLL) __jm[LSIZE];
    __local concat(REAL, UNROLL) __jvx[LSIZE];
    __local concat(REAL, UNROLL) __jvy[LSIZE];
    __local concat(REAL, UNROLL) __jvz[LSIZE];
    __local concat(REAL, UNROLL) __jax[LSIZE];
    __local concat(REAL, UNROLL) __jay[LSIZE];
    __local concat(REAL, UNROLL) __jaz[LSIZE];
    for (; (j + UNROLL * lsize) < nj; j += UNROLL * lsize) {
        concat(REAL, UNROLL) jm = concat(vload, UNROLL)(lid, _jm + j);
        concat(REAL, UNROLL) jvx = concat(vload, UNROLL)(lid, _jvx + j);
        concat(REAL, UNROLL) jvy = concat(vload, UNROLL)(lid, _jvy + j);
        concat(REAL, UNROLL) jvz = concat(vload, UNROLL)(lid, _jvz + j);
        concat(REAL, UNROLL) jax = concat(vload, UNROLL)(lid, _jax + j);
        concat(REAL, UNROLL) jay = concat(vload, UNROLL)(lid, _jay + j);
        concat(REAL, UNROLL) jaz = concat(vload, UNROLL)(lid, _jaz + j);
        barrier(CLK_LOCAL_MEM_FENCE);
        __jm[lid] = jm;
        __jvx[lid] = jvx;
        __jvy[lid] = jvy;
        __jvz[lid] = jvz;
        __jax[lid] = jax;
        __jay[lid] = jay;
        __jaz[lid] = jaz;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (UINT k = 0; k < lsize; ++k) {
            jm = __jm[k];
            jvx = __jvx[k];
            jvy = __jvy[k];
            jvz = __jvz[k];
            jax = __jax[k];
            jay = __jay[k];
            jaz = __jaz[k];
            #if UNROLL == 1
                nreg_Vkernel_core(dt,
                                  im, ivx, ivy, ivz,
                                  iax, iay, iaz,
                                  jm, jvx, jvy, jvz,
                                  jax, jay, jaz,
                                  &idvx, &idvy, &idvz, &ik);
            #else
                nreg_Vkernel_core(dt,
                                  im, ivx, ivy, ivz,
                                  iax, iay, iaz,
                                  jm.s0, jvx.s0, jvy.s0, jvz.s0,
                                  jax.s0, jay.s0, jaz.s0,
                                  &idvx, &idvy, &idvz, &ik);
                #pragma unroll
                for (UINT l = 1; l < UNROLL; ++l) {
                    jm = shuffle(jm, MASK);
                    jvx = shuffle(jvx, MASK);
                    jvy = shuffle(jvy, MASK);
                    jvz = shuffle(jvz, MASK);
                    jax = shuffle(jax, MASK);
                    jay = shuffle(jay, MASK);
                    jaz = shuffle(jaz, MASK);
                    nreg_Vkernel_core(dt,
                                      im, ivx, ivy, ivz,
                                      iax, iay, iaz,
                                      jm.s0, jvx.s0, jvy.s0, jvz.s0,
                                      jax.s0, jay.s0, jaz.s0,
                                      &idvx, &idvy, &idvz, &ik);
                }
            #endif
        }
    }
    for (; j < nj; ++j) {
        nreg_Vkernel_core(dt,
                          im, ivx, ivy, ivz,
                          iax, iay, iaz,
                          _jm[j], _jvx[j], _jvy[j], _jvz[j],
                          _jax[j], _jay[j], _jaz[j],
                          &idvx, &idvy, &idvz, &ik);
    }

    vstoren(idvx, gid, _idvx);
    vstoren(idvy, gid, _idvy);
    vstoren(idvz, gid, _idvz);
    vstoren(im * ik, gid, _ik);
}


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
    REALn iax = vloadn(gid, _iax);
    REALn iay = vloadn(gid, _iay);
    REALn iaz = vloadn(gid, _iaz);
    REALn ijx = vloadn(gid, _ijx);
    REALn ijy = vloadn(gid, _ijy);
    REALn ijz = vloadn(gid, _ijz);
    REALn isx = (REALn)(0);
    REALn isy = (REALn)(0);
    REALn isz = (REALn)(0);
    REALn icx = (REALn)(0);
    REALn icy = (REALn)(0);
    REALn icz = (REALn)(0);

    UINT j = 0;
    __local concat(REAL, UNROLL) __jm[LSIZE];
    __local concat(REAL, UNROLL) __jrx[LSIZE];
    __local concat(REAL, UNROLL) __jry[LSIZE];
    __local concat(REAL, UNROLL) __jrz[LSIZE];
    __local concat(REAL, UNROLL) __je2[LSIZE];
    __local concat(REAL, UNROLL) __jvx[LSIZE];
    __local concat(REAL, UNROLL) __jvy[LSIZE];
    __local concat(REAL, UNROLL) __jvz[LSIZE];
    __local concat(REAL, UNROLL) __jax[LSIZE];
    __local concat(REAL, UNROLL) __jay[LSIZE];
    __local concat(REAL, UNROLL) __jaz[LSIZE];
    __local concat(REAL, UNROLL) __jjx[LSIZE];
    __local concat(REAL, UNROLL) __jjy[LSIZE];
    __local concat(REAL, UNROLL) __jjz[LSIZE];
    for (; (j + UNROLL * lsize) < nj; j += UNROLL * lsize) {
        concat(REAL, UNROLL) jm = concat(vload, UNROLL)(lid, _jm + j);
        concat(REAL, UNROLL) jrx = concat(vload, UNROLL)(lid, _jrx + j);
        concat(REAL, UNROLL) jry = concat(vload, UNROLL)(lid, _jry + j);
        concat(REAL, UNROLL) jrz = concat(vload, UNROLL)(lid, _jrz + j);
        concat(REAL, UNROLL) je2 = concat(vload, UNROLL)(lid, _je2 + j);
        concat(REAL, UNROLL) jvx = concat(vload, UNROLL)(lid, _jvx + j);
        concat(REAL, UNROLL) jvy = concat(vload, UNROLL)(lid, _jvy + j);
        concat(REAL, UNROLL) jvz = concat(vload, UNROLL)(lid, _jvz + j);
        concat(REAL, UNROLL) jax = concat(vload, UNROLL)(lid, _jax + j);
        concat(REAL, UNROLL) jay = concat(vload, UNROLL)(lid, _jay + j);
        concat(REAL, UNROLL) jaz = concat(vload, UNROLL)(lid, _jaz + j);
        concat(REAL, UNROLL) jjx = concat(vload, UNROLL)(lid, _jjx + j);
        concat(REAL, UNROLL) jjy = concat(vload, UNROLL)(lid, _jjy + j);
        concat(REAL, UNROLL) jjz = concat(vload, UNROLL)(lid, _jjz + j);
        barrier(CLK_LOCAL_MEM_FENCE);
        __jm[lid] = jm;
        __jrx[lid] = jrx;
        __jry[lid] = jry;
        __jrz[lid] = jrz;
        __je2[lid] = je2;
        __jvx[lid] = jvx;
        __jvy[lid] = jvy;
        __jvz[lid] = jvz;
        __jax[lid] = jax;
        __jay[lid] = jay;
        __jaz[lid] = jaz;
        __jjx[lid] = jjx;
        __jjy[lid] = jjy;
        __jjz[lid] = jjz;
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
            jax = __jax[k];
            jay = __jay[k];
            jaz = __jaz[k];
            jjx = __jjx[k];
            jjy = __jjy[k];
            jjz = __jjz[k];
            #if UNROLL == 1
                snap_crackle_kernel_core(im, irx, iry, irz,
                                         ie2, ivx, ivy, ivz,
                                         iax, iay, iaz,
                                         ijx, ijy, ijz,
                                         jm, jrx, jry, jrz,
                                         je2, jvx, jvy, jvz,
                                         jax, jay, jaz,
                                         jjx, jjy, jjz,
                                         &isx, &isy, &isz,
                                         &icx, &icy, &icz);
            #else
                snap_crackle_kernel_core(im, irx, iry, irz,
                                         ie2, ivx, ivy, ivz,
                                         iax, iay, iaz,
                                         ijx, ijy, ijz,
                                         jm.s0, jrx.s0, jry.s0, jrz.s0,
                                         je2.s0, jvx.s0, jvy.s0, jvz.s0,
                                         jax.s0, jay.s0, jaz.s0,
                                         jjx.s0, jjy.s0, jjz.s0,
                                         &isx, &isy, &isz,
                                         &icx, &icy, &icz);
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
                    jax = shuffle(jax, MASK);
                    jay = shuffle(jay, MASK);
                    jaz = shuffle(jaz, MASK);
                    jjx = shuffle(jjx, MASK);
                    jjy = shuffle(jjy, MASK);
                    jjz = shuffle(jjz, MASK);
                    snap_crackle_kernel_core(im, irx, iry, irz,
                                             ie2, ivx, ivy, ivz,
                                             iax, iay, iaz,
                                             ijx, ijy, ijz,
                                             jm.s0, jrx.s0, jry.s0, jrz.s0,
                                             je2.s0, jvx.s0, jvy.s0, jvz.s0,
                                             jax.s0, jay.s0, jaz.s0,
                                             jjx.s0, jjy.s0, jjz.s0,
                                             &isx, &isy, &isz,
                                             &icx, &icy, &icz);
                }
            #endif
        }
    }
    for (; j < nj; ++j) {
        snap_crackle_kernel_core(im, irx, iry, irz,
                                 ie2, ivx, ivy, ivz,
                                 iax, iay, iaz,
                                 ijx, ijy, ijz,
                                 _jm[j], _jrx[j], _jry[j], _jrz[j],
                                 _je2[j], _jvx[j], _jvy[j], _jvz[j],
                                 _jax[j], _jay[j], _jaz[j],
                                 _jjx[j], _jjy[j], _jjz[j],
                                 &isx, &isy, &isz,
                                 &icx, &icy, &icz);
    }

    vstoren(isx, gid, _isx);
    vstoren(isy, gid, _isy);
    vstoren(isz, gid, _isz);
    vstoren(icx, gid, _icx);
    vstoren(icy, gid, _icy);
    vstoren(icz, gid, _icz);
}


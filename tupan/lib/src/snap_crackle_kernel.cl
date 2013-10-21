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
    UINT i = get_global_id(0);

    REALn im = vloadn(i, _im);
    REALn irx = vloadn(i, _irx);
    REALn iry = vloadn(i, _iry);
    REALn irz = vloadn(i, _irz);
    REALn ie2 = vloadn(i, _ie2);
    REALn ivx = vloadn(i, _ivx);
    REALn ivy = vloadn(i, _ivy);
    REALn ivz = vloadn(i, _ivz);
    REALn iax = vloadn(i, _iax);
    REALn iay = vloadn(i, _iay);
    REALn iaz = vloadn(i, _iaz);
    REALn ijx = vloadn(i, _ijx);
    REALn ijy = vloadn(i, _ijy);
    REALn ijz = vloadn(i, _ijz);
    REALn isx = (REALn)(0);
    REALn isy = (REALn)(0);
    REALn isz = (REALn)(0);
    REALn icx = (REALn)(0);
    REALn icy = (REALn)(0);
    REALn icz = (REALn)(0);

    UINT j = 0;
    UINT lsize = min((UINT)(LSIZE), (UINT)(get_local_size(0) + WIDTH - 1)) / WIDTH;
    UINT lid = get_local_id(0) % lsize;
    __local concat(REAL, WIDTH) __jm[LSIZE / WIDTH];
    __local concat(REAL, WIDTH) __jrx[LSIZE / WIDTH];
    __local concat(REAL, WIDTH) __jry[LSIZE / WIDTH];
    __local concat(REAL, WIDTH) __jrz[LSIZE / WIDTH];
    __local concat(REAL, WIDTH) __je2[LSIZE / WIDTH];
    __local concat(REAL, WIDTH) __jvx[LSIZE / WIDTH];
    __local concat(REAL, WIDTH) __jvy[LSIZE / WIDTH];
    __local concat(REAL, WIDTH) __jvz[LSIZE / WIDTH];
    __local concat(REAL, WIDTH) __jax[LSIZE / WIDTH];
    __local concat(REAL, WIDTH) __jay[LSIZE / WIDTH];
    __local concat(REAL, WIDTH) __jaz[LSIZE / WIDTH];
    __local concat(REAL, WIDTH) __jjx[LSIZE / WIDTH];
    __local concat(REAL, WIDTH) __jjy[LSIZE / WIDTH];
    __local concat(REAL, WIDTH) __jjz[LSIZE / WIDTH];
    for (; (j + WIDTH * lsize) < nj; j += WIDTH * lsize) {
        concat(REAL, WIDTH) jm = concat(vload, WIDTH)(lid, _jm + j);
        concat(REAL, WIDTH) jrx = concat(vload, WIDTH)(lid, _jrx + j);
        concat(REAL, WIDTH) jry = concat(vload, WIDTH)(lid, _jry + j);
        concat(REAL, WIDTH) jrz = concat(vload, WIDTH)(lid, _jrz + j);
        concat(REAL, WIDTH) je2 = concat(vload, WIDTH)(lid, _je2 + j);
        concat(REAL, WIDTH) jvx = concat(vload, WIDTH)(lid, _jvx + j);
        concat(REAL, WIDTH) jvy = concat(vload, WIDTH)(lid, _jvy + j);
        concat(REAL, WIDTH) jvz = concat(vload, WIDTH)(lid, _jvz + j);
        concat(REAL, WIDTH) jax = concat(vload, WIDTH)(lid, _jax + j);
        concat(REAL, WIDTH) jay = concat(vload, WIDTH)(lid, _jay + j);
        concat(REAL, WIDTH) jaz = concat(vload, WIDTH)(lid, _jaz + j);
        concat(REAL, WIDTH) jjx = concat(vload, WIDTH)(lid, _jjx + j);
        concat(REAL, WIDTH) jjy = concat(vload, WIDTH)(lid, _jjy + j);
        concat(REAL, WIDTH) jjz = concat(vload, WIDTH)(lid, _jjz + j);
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
            #if WIDTH == 1
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
                #pragma unroll
                for (UINT l = 0; l < UNROLL; ++l) {
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
                }
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

    vstoren(isx, i, _isx);
    vstoren(isy, i, _isy);
    vstoren(isz, i, _isz);
    vstoren(icx, i, _icx);
    vstoren(icy, i, _icy);
    vstoren(icz, i, _icz);
}


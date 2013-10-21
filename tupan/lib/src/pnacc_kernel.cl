#include "pnacc_kernel_common.h"


__kernel void pnacc_kernel(
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
    const UINT order,
    const REAL inv1,
    const REAL inv2,
    const REAL inv3,
    const REAL inv4,
    const REAL inv5,
    const REAL inv6,
    const REAL inv7,
    __global REAL * restrict _ipnax,
    __global REAL * restrict _ipnay,
    __global REAL * restrict _ipnaz)
{
    UINT i = get_global_id(0);

    CLIGHT clight = CLIGHT_Init(order, inv1, inv2, inv3, inv4, inv5, inv6, inv7);
    REALn im = vloadn(i, _im);
    REALn irx = vloadn(i, _irx);
    REALn iry = vloadn(i, _iry);
    REALn irz = vloadn(i, _irz);
    REALn ie2 = vloadn(i, _ie2);
    REALn ivx = vloadn(i, _ivx);
    REALn ivy = vloadn(i, _ivy);
    REALn ivz = vloadn(i, _ivz);
    REALn ipnax = (REALn)(0);
    REALn ipnay = (REALn)(0);
    REALn ipnaz = (REALn)(0);

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
    for (; (j + WIDTH * lsize) < nj; j += WIDTH * lsize) {
        concat(REAL, WIDTH) jm = concat(vload, WIDTH)(lid, _jm + j);
        concat(REAL, WIDTH) jrx = concat(vload, WIDTH)(lid, _jrx + j);
        concat(REAL, WIDTH) jry = concat(vload, WIDTH)(lid, _jry + j);
        concat(REAL, WIDTH) jrz = concat(vload, WIDTH)(lid, _jrz + j);
        concat(REAL, WIDTH) je2 = concat(vload, WIDTH)(lid, _je2 + j);
        concat(REAL, WIDTH) jvx = concat(vload, WIDTH)(lid, _jvx + j);
        concat(REAL, WIDTH) jvy = concat(vload, WIDTH)(lid, _jvy + j);
        concat(REAL, WIDTH) jvz = concat(vload, WIDTH)(lid, _jvz + j);
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
            #if WIDTH == 1
                pnacc_kernel_core(im, irx, iry, irz,
                                  ie2, ivx, ivy, ivz,
                                  jm, jrx, jry, jrz,
                                  je2, jvx, jvy, jvz,
                                  clight,
                                  &ipnax, &ipnay, &ipnaz);
            #else
                #pragma unroll
                for (UINT l = 0; l < UNROLL; ++l) {
                    pnacc_kernel_core(im, irx, iry, irz,
                                      ie2, ivx, ivy, ivz,
                                      jm.s0, jrx.s0, jry.s0, jrz.s0,
                                      je2.s0, jvx.s0, jvy.s0, jvz.s0,
                                      clight,
                                      &ipnax, &ipnay, &ipnaz);
                    jm = shuffle(jm, MASK);
                    jrx = shuffle(jrx, MASK);
                    jry = shuffle(jry, MASK);
                    jrz = shuffle(jrz, MASK);
                    je2 = shuffle(je2, MASK);
                    jvx = shuffle(jvx, MASK);
                    jvy = shuffle(jvy, MASK);
                    jvz = shuffle(jvz, MASK);
                }
                pnacc_kernel_core(im, irx, iry, irz,
                                  ie2, ivx, ivy, ivz,
                                  jm.s0, jrx.s0, jry.s0, jrz.s0,
                                  je2.s0, jvx.s0, jvy.s0, jvz.s0,
                                  clight,
                                  &ipnax, &ipnay, &ipnaz);
            #endif
        }
    }
    for (; j < nj; ++j) {
        pnacc_kernel_core(im, irx, iry, irz,
                          ie2, ivx, ivy, ivz,
                          _jm[j], _jrx[j], _jry[j], _jrz[j],
                          _je2[j], _jvx[j], _jvy[j], _jvz[j],
                          clight,
                          &ipnax, &ipnay, &ipnaz);
    }

    vstoren(ipnax, i, _ipnax);
    vstoren(ipnay, i, _ipnay);
    vstoren(ipnaz, i, _ipnaz);
}


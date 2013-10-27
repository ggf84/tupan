#include "tstep_kernel_common.h"


__kernel void tstep_kernel(
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
    const REAL eta,
    __global REAL * restrict _idt_a,
    __global REAL * restrict _idt_b)
{
    UINT lsize = get_local_size(0);
    UINT lid = get_local_id(0);
    UINT gid = get_global_id(0);
    gid = min(gid, ((ni - 1) + VECTOR_WIDTH - 1) / VECTOR_WIDTH);

    REALn im = vloadn(gid, _im);
    REALn irx = vloadn(gid, _irx);
    REALn iry = vloadn(gid, _iry);
    REALn irz = vloadn(gid, _irz);
    REALn ie2 = vloadn(gid, _ie2);
    REALn ivx = vloadn(gid, _ivx);
    REALn ivy = vloadn(gid, _ivy);
    REALn ivz = vloadn(gid, _ivz);
    REALn iw2_a = (REALn)(0);
    REALn iw2_b = (REALn)(0);

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
                tstep_kernel_core(eta,
                                  im, irx, iry, irz,
                                  ie2, ivx, ivy, ivz,
                                  jm, jrx, jry, jrz,
                                  je2, jvx, jvy, jvz,
                                  &iw2_a, &iw2_b);
            #else
                tstep_kernel_core(eta,
                                  im, irx, iry, irz,
                                  ie2, ivx, ivy, ivz,
                                  jm.s0, jrx.s0, jry.s0, jrz.s0,
                                  je2.s0, jvx.s0, jvy.s0, jvz.s0,
                                  &iw2_a, &iw2_b);
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
                    tstep_kernel_core(eta,
                                      im, irx, iry, irz,
                                      ie2, ivx, ivy, ivz,
                                      jm.s0, jrx.s0, jry.s0, jrz.s0,
                                      je2.s0, jvx.s0, jvy.s0, jvz.s0,
                                      &iw2_a, &iw2_b);
                }
            #endif
        }
    }
    for (; j < nj; ++j) {
        tstep_kernel_core(eta,
                          im, irx, iry, irz,
                          ie2, ivx, ivy, ivz,
                          _jm[j], _jrx[j], _jry[j], _jrz[j],
                          _je2[j], _jvx[j], _jvy[j], _jvz[j],
                          &iw2_a, &iw2_b);
    }

    REALn idt_a = eta / sqrt(1 + iw2_a);
    REALn idt_b = eta / sqrt(1 + iw2_b);
    vstoren(idt_a, gid, _idt_a);
    vstoren(idt_b, gid, _idt_b);
}


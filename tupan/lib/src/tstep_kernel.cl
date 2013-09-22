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
    __global REAL * restrict _idt_b,
    __local REAL *__jm,
    __local REAL *__jrx,
    __local REAL *__jry,
    __local REAL *__jrz,
    __local REAL *__je2,
    __local REAL *__jvx,
    __local REAL *__jvy,
    __local REAL *__jvz)
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

    REALn veta = (REALn)(eta);

    REALn iw2_a = (REALn)(0);
    REALn iw2_b = (REALn)(0);

    for (UINT j = 0; j < nj; ++j) {
        REALn jm = (REALn)(_jm[j]);
        REALn jrx = (REALn)(_jrx[j]);
        REALn jry = (REALn)(_jry[j]);
        REALn jrz = (REALn)(_jrz[j]);
        REALn je2 = (REALn)(_je2[j]);
        REALn jvx = (REALn)(_jvx[j]);
        REALn jvy = (REALn)(_jvy[j]);
        REALn jvz = (REALn)(_jvz[j]);
        tstep_kernel_core(veta,
                          im, irx, iry, irz, ie2, ivx, ivy, ivz,
                          jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                          &iw2_a, &iw2_b);
    }

    REALn idt_a = veta / sqrt(1 + iw2_a);
    REALn idt_b = veta / sqrt(1 + iw2_b);
    vstoren(idt_a, i, _idt_a);
    vstoren(idt_b, i, _idt_b);
}


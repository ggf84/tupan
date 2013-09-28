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

    UINT j = 0;
    for (; (j + LSIZE) < nj; j += LSIZE) {
        event_t e[8];
        e[0] = async_work_group_copy(__jm,  _jm  + j, LSIZE, 0);
        e[1] = async_work_group_copy(__jrx,  _jrx  + j, LSIZE, 0);
        e[2] = async_work_group_copy(__jry,  _jry  + j, LSIZE, 0);
        e[3] = async_work_group_copy(__jrz,  _jrz  + j, LSIZE, 0);
        e[4] = async_work_group_copy(__je2,  _je2  + j, LSIZE, 0);
        e[5] = async_work_group_copy(__jvx,  _jvx  + j, LSIZE, 0);
        e[6] = async_work_group_copy(__jvy,  _jvy  + j, LSIZE, 0);
        e[7] = async_work_group_copy(__jvz,  _jvz  + j, LSIZE, 0);
        wait_group_events(8, e);
        for (UINT k = 0; k < LSIZE; ++k) {
            REALn jm = (REALn)(_jm[k]);
            REALn jrx = (REALn)(_jrx[k]);
            REALn jry = (REALn)(_jry[k]);
            REALn jrz = (REALn)(_jrz[k]);
            REALn je2 = (REALn)(_je2[k]);
            REALn jvx = (REALn)(_jvx[k]);
            REALn jvy = (REALn)(_jvy[k]);
            REALn jvz = (REALn)(_jvz[k]);
            tstep_kernel_core(veta,
                              im, irx, iry, irz, ie2, ivx, ivy, ivz,
                              jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                              &iw2_a, &iw2_b);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (; j < nj; ++j) {
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


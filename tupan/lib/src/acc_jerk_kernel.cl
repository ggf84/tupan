#include "acc_jerk_kernel_common.h"


__kernel void acc_jerk_kernel(
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
    __global REAL * restrict _iax,
    __global REAL * restrict _iay,
    __global REAL * restrict _iaz,
    __global REAL * restrict _ijx,
    __global REAL * restrict _ijy,
    __global REAL * restrict _ijz,
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

    REALn iax = (REALn)(0);
    REALn iay = (REALn)(0);
    REALn iaz = (REALn)(0);
    REALn ijx = (REALn)(0);
    REALn ijy = (REALn)(0);
    REALn ijz = (REALn)(0);

    UINT j = 0;
    for (; (j + LSIZE) < nj; j += LSIZE) {
        event_t e[8];
        e[0] = async_work_group_copy(__jm, _jm + j, LSIZE, 0);
        e[1] = async_work_group_copy(__jrx, _jrx + j, LSIZE, 0);
        e[2] = async_work_group_copy(__jry, _jry + j, LSIZE, 0);
        e[3] = async_work_group_copy(__jrz, _jrz + j, LSIZE, 0);
        e[4] = async_work_group_copy(__je2, _je2 + j, LSIZE, 0);
        e[5] = async_work_group_copy(__jvx, _jvx + j, LSIZE, 0);
        e[6] = async_work_group_copy(__jvy, _jvy + j, LSIZE, 0);
        e[7] = async_work_group_copy(__jvz, _jvz + j, LSIZE, 0);
        wait_group_events(8, e);
        for (UINT k = 0; k < LSIZE; ++k) {
            REALn jm = (REALn)(__jm[k]);
            REALn jrx = (REALn)(__jrx[k]);
            REALn jry = (REALn)(__jry[k]);
            REALn jrz = (REALn)(__jrz[k]);
            REALn je2 = (REALn)(__je2[k]);
            REALn jvx = (REALn)(__jvx[k]);
            REALn jvy = (REALn)(__jvy[k]);
            REALn jvz = (REALn)(__jvz[k]);
            acc_jerk_kernel_core(im, irx, iry, irz, ie2, ivx, ivy, ivz,
                                 jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                                 &iax, &iay, &iaz,
                                 &ijx, &ijy, &ijz);
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
        acc_jerk_kernel_core(im, irx, iry, irz, ie2, ivx, ivy, ivz,
                             jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                             &iax, &iay, &iaz,
                             &ijx, &ijy, &ijz);
    }

    vstoren(iax, i, _iax);
    vstoren(iay, i, _iay);
    vstoren(iaz, i, _iaz);
    vstoren(ijx, i, _ijx);
    vstoren(ijy, i, _ijy);
    vstoren(ijz, i, _ijz);
}


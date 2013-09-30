#include "acc_kernel_common.h"


__kernel void acc_kernel(
    const UINT ni,
    __global const REAL * restrict _im,
    __global const REAL * restrict _irx,
    __global const REAL * restrict _iry,
    __global const REAL * restrict _irz,
    __global const REAL * restrict _ie2,
    const UINT nj,
    __global const REAL * restrict _jm,
    __global const REAL * restrict _jrx,
    __global const REAL * restrict _jry,
    __global const REAL * restrict _jrz,
    __global const REAL * restrict _je2,
    __global REAL * restrict _iax,
    __global REAL * restrict _iay,
    __global REAL * restrict _iaz,
    __local REAL * __jm,
    __local REAL * __jrx,
    __local REAL * __jry,
    __local REAL * __jrz,
    __local REAL * __je2)
{
    UINT i = get_global_id(0);

    REALn im = vloadn(i, _im);
    REALn irx = vloadn(i, _irx);
    REALn iry = vloadn(i, _iry);
    REALn irz = vloadn(i, _irz);
    REALn ie2 = vloadn(i, _ie2);

    REALn iax = (REALn)(0);
    REALn iay = (REALn)(0);
    REALn iaz = (REALn)(0);

    UINT j = 0;

    // async copy: global -> local
    for (; (j + LSIZE) < nj; j += LSIZE) {
        event_t e[5];
        e[0] = async_work_group_copy(__jm, _jm + j, LSIZE, 0);
        e[1] = async_work_group_copy(__jrx, _jrx + j, LSIZE, 0);
        e[2] = async_work_group_copy(__jry, _jry + j, LSIZE, 0);
        e[3] = async_work_group_copy(__jrz, _jrz + j, LSIZE, 0);
        e[4] = async_work_group_copy(__je2, _je2 + j, LSIZE, 0);
        wait_group_events(5, e);
        for (UINT k = 0; k < LSIZE; ++k) {
            REALn jm = (REALn)(__jm[k]);
            REALn jrx = (REALn)(__jrx[k]);
            REALn jry = (REALn)(__jry[k]);
            REALn jrz = (REALn)(__jrz[k]);
            REALn je2 = (REALn)(__je2[k]);
            acc_kernel_core(im, irx, iry, irz, ie2,
                            jm, jrx, jry, jrz, je2,
                            &iax, &iay, &iaz);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

/*
    // for decreasing lsize, async copy: global -> local
    for (UINT ls = LSIZE; ls > 0; ls /= 2) {
        for (; (j + ls) < nj; j += ls) {
            event_t e[5];
            e[0] = async_work_group_copy(__jm, _jm + j, ls, 0);
            e[1] = async_work_group_copy(__jrx, _jrx + j, ls, 0);
            e[2] = async_work_group_copy(__jry, _jry + j, ls, 0);
            e[3] = async_work_group_copy(__jrz, _jrz + j, ls, 0);
            e[4] = async_work_group_copy(__je2, _je2 + j, ls, 0);
            wait_group_events(5, e);
            for (UINT k = 0; k < ls; ++k) {
                REALn jm = (REALn)(__jm[k]);
                REALn jrx = (REALn)(__jrx[k]);
                REALn jry = (REALn)(__jry[k]);
                REALn jrz = (REALn)(__jrz[k]);
                REALn je2 = (REALn)(__je2[k]);
                acc_kernel_core(im, irx, iry, irz, ie2,
                                jm, jrx, jry, jrz, je2,
                                &iax, &iay, &iaz);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
*/
/*
    // async copy: global -> local. Then, vload2: local -> private
    for (; (j + LSIZE) < nj; j += LSIZE) {
        event_t e[5];
        e[0] = async_work_group_copy(__jm, _jm + j, LSIZE, 0);
        e[1] = async_work_group_copy(__jrx, _jrx + j, LSIZE, 0);
        e[2] = async_work_group_copy(__jry, _jry + j, LSIZE, 0);
        e[3] = async_work_group_copy(__jrz, _jrz + j, LSIZE, 0);
        e[4] = async_work_group_copy(__je2, _je2 + j, LSIZE, 0);
        wait_group_events(5, e);
        for (UINT k = 0; k < LSIZE; k += 2) {
            REAL2 jm = vload2(0, __jm + k);
            REAL2 jrx = vload2(0, __jrx + k);
            REAL2 jry = vload2(0, __jry + k);
            REAL2 jrz = vload2(0, __jrz + k);
            REAL2 je2 = vload2(0, __je2 + k);
            acc_kernel_core(im, irx, iry, irz, ie2,
                            jm.s0, jrx.s0, jry.s0, jrz.s0, je2.s0,
                            &iax, &iay, &iaz);
            acc_kernel_core(im, irx, iry, irz, ie2,
                            jm.s1, jrx.s1, jry.s1, jrz.s1, je2.s1,
                            &iax, &iay, &iaz);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
*/
/*
    // async copy: global -> local. Then, vload4: local -> private
    for (; (j + LSIZE) < nj; j += LSIZE) {
        event_t e[5];
        e[0] = async_work_group_copy(__jm, _jm + j, LSIZE, 0);
        e[1] = async_work_group_copy(__jrx, _jrx + j, LSIZE, 0);
        e[2] = async_work_group_copy(__jry, _jry + j, LSIZE, 0);
        e[3] = async_work_group_copy(__jrz, _jrz + j, LSIZE, 0);
        e[4] = async_work_group_copy(__je2, _je2 + j, LSIZE, 0);
        wait_group_events(5, e);
        for (UINT k = 0; k < LSIZE; k += 4) {
            REAL4 jm = vload4(0, __jm + k);
            REAL4 jrx = vload4(0, __jrx + k);
            REAL4 jry = vload4(0, __jry + k);
            REAL4 jrz = vload4(0, __jrz + k);
            REAL4 je2 = vload4(0, __je2 + k);
            acc_kernel_core(im, irx, iry, irz, ie2,
                            jm.s0, jrx.s0, jry.s0, jrz.s0, je2.s0,
                            &iax, &iay, &iaz);
            acc_kernel_core(im, irx, iry, irz, ie2,
                            jm.s1, jrx.s1, jry.s1, jrz.s1, je2.s1,
                            &iax, &iay, &iaz);
            acc_kernel_core(im, irx, iry, irz, ie2,
                            jm.s2, jrx.s2, jry.s2, jrz.s2, je2.s2,
                            &iax, &iay, &iaz);
            acc_kernel_core(im, irx, iry, irz, ie2,
                            jm.s3, jrx.s3, jry.s3, jrz.s3, je2.s3,
                            &iax, &iay, &iaz);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
*/
/*
    // vload16: global -> private
    for (; (j + 16) < nj; j += 16) {
        REAL16 jm = vload16(0, _jm + j);
        REAL16 jrx = vload16(0, _jrx + j);
        REAL16 jry = vload16(0, _jry + j);
        REAL16 jrz = vload16(0, _jrz + j);
        REAL16 je2 = vload16(0, _je2 + j);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.s0, jrx.s0, jry.s0, jrz.s0, je2.s0,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.s1, jrx.s1, jry.s1, jrz.s1, je2.s1,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.s2, jrx.s2, jry.s2, jrz.s2, je2.s2,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.s3, jrx.s3, jry.s3, jrz.s3, je2.s3,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.s4, jrx.s4, jry.s4, jrz.s4, je2.s4,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.s5, jrx.s5, jry.s5, jrz.s5, je2.s5,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.s6, jrx.s6, jry.s6, jrz.s6, je2.s6,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.s7, jrx.s7, jry.s7, jrz.s7, je2.s7,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.s8, jrx.s8, jry.s8, jrz.s8, je2.s8,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.s9, jrx.s9, jry.s9, jrz.s9, je2.s9,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.sA, jrx.sA, jry.sA, jrz.sA, je2.sA,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.sB, jrx.sB, jry.sB, jrz.sB, je2.sB,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.sC, jrx.sC, jry.sC, jrz.sC, je2.sC,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.sD, jrx.sD, jry.sD, jrz.sD, je2.sD,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.sE, jrx.sE, jry.sE, jrz.sE, je2.sE,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.sF, jrx.sF, jry.sF, jrz.sF, je2.sF,
                        &iax, &iay, &iaz);
    }
*/
/*
    // vload8: global -> private
    for (; (j + 8) < nj; j += 8) {
        REAL8 jm = vload8(0, _jm + j);
        REAL8 jrx = vload8(0, _jrx + j);
        REAL8 jry = vload8(0, _jry + j);
        REAL8 jrz = vload8(0, _jrz + j);
        REAL8 je2 = vload8(0, _je2 + j);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.s0, jrx.s0, jry.s0, jrz.s0, je2.s0,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.s1, jrx.s1, jry.s1, jrz.s1, je2.s1,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.s2, jrx.s2, jry.s2, jrz.s2, je2.s2,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.s3, jrx.s3, jry.s3, jrz.s3, je2.s3,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.s4, jrx.s4, jry.s4, jrz.s4, je2.s4,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.s5, jrx.s5, jry.s5, jrz.s5, je2.s5,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.s6, jrx.s6, jry.s6, jrz.s6, je2.s6,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.s7, jrx.s7, jry.s7, jrz.s7, je2.s7,
                        &iax, &iay, &iaz);
    }
*/
/*
    // vload4: global -> private
    for (; (j + 4) < nj; j += 4) {
        REAL4 jm = vload4(0, _jm + j);
        REAL4 jrx = vload4(0, _jrx + j);
        REAL4 jry = vload4(0, _jry + j);
        REAL4 jrz = vload4(0, _jrz + j);
        REAL4 je2 = vload4(0, _je2 + j);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.s0, jrx.s0, jry.s0, jrz.s0, je2.s0,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.s1, jrx.s1, jry.s1, jrz.s1, je2.s1,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.s2, jrx.s2, jry.s2, jrz.s2, je2.s2,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.s3, jrx.s3, jry.s3, jrz.s3, je2.s3,
                        &iax, &iay, &iaz);
    }
*/
/*
    // vload2: global -> private
    for (; (j + 2) < nj; j += 2) {
        REAL2 jm = vload2(0, _jm + j);
        REAL2 jrx = vload2(0, _jrx + j);
        REAL2 jry = vload2(0, _jry + j);
        REAL2 jrz = vload2(0, _jrz + j);
        REAL2 je2 = vload2(0, _je2 + j);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.s0, jrx.s0, jry.s0, jrz.s0, je2.s0,
                        &iax, &iay, &iaz);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm.s1, jrx.s1, jry.s1, jrz.s1, je2.s1,
                        &iax, &iay, &iaz);
    }
*/

    // scalar copy: global -> private
    for (; j < nj; ++j) {
        REALn jm = (REALn)(_jm[j]);
        REALn jrx = (REALn)(_jrx[j]);
        REALn jry = (REALn)(_jry[j]);
        REALn jrz = (REALn)(_jrz[j]);
        REALn je2 = (REALn)(_je2[j]);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm, jrx, jry, jrz, je2,
                        &iax, &iay, &iaz);
    }

    vstoren(iax, i, _iax);
    vstoren(iay, i, _iay);
    vstoren(iaz, i, _iaz);
}


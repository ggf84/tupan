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
    __global REAL * restrict _iaz)
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
/*
    UINT j = 0;
    __local REAL __jm[LSIZE];
    __local REAL __jrx[LSIZE];
    __local REAL __jry[LSIZE];
    __local REAL __jrz[LSIZE];
    __local REAL __je2[LSIZE];
    for (; (j + LSIZE) < nj; j += LSIZE) {
        event_t e[5];
        e[0] = async_work_group_copy(__jm, _jm + j, LSIZE, 0);
        e[1] = async_work_group_copy(__jrx, _jrx + j, LSIZE, 0);
        e[2] = async_work_group_copy(__jry, _jry + j, LSIZE, 0);
        e[3] = async_work_group_copy(__jrz, _jrz + j, LSIZE, 0);
        e[4] = async_work_group_copy(__je2, _je2 + j, LSIZE, 0);
        wait_group_events(5, e);
        for (UINT k = 0; k < LSIZE; ++k) {
            acc_kernel_core(im, irx, iry, irz, ie2,
                            __jm[k], __jrx[k], __jry[k], __jrz[k], __je2[k],
                            &iax, &iay, &iaz);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (; j < nj; ++j) {
        acc_kernel_core(im, irx, iry, irz, ie2,
                        _jm[j], _jrx[j], _jry[j], _jrz[j], _je2[j],
                        &iax, &iay, &iaz);
    }
*/

#define WIDTH 2

    UINT j = 0;
    __local concat(REAL, WIDTH) __jm[LSIZE / WIDTH];
    __local concat(REAL, WIDTH) __jrx[LSIZE / WIDTH];
    __local concat(REAL, WIDTH) __jry[LSIZE / WIDTH];
    __local concat(REAL, WIDTH) __jrz[LSIZE / WIDTH];
    __local concat(REAL, WIDTH) __je2[LSIZE / WIDTH];

    UINT lsize = min((UINT)(LSIZE / WIDTH), (UINT)(get_local_size(0)));
    UINT lid = get_local_id(0) % lsize;

    for (; (j + WIDTH * lsize) < nj; j += WIDTH * lsize) {

        concat(REAL, WIDTH) jm = concat(vload, WIDTH)(lid, _jm + j);
        concat(REAL, WIDTH) jrx = concat(vload, WIDTH)(lid, _jrx + j);
        concat(REAL, WIDTH) jry = concat(vload, WIDTH)(lid, _jry + j);
        concat(REAL, WIDTH) jrz = concat(vload, WIDTH)(lid, _jrz + j);
        concat(REAL, WIDTH) je2 = concat(vload, WIDTH)(lid, _je2 + j);
        barrier(CLK_LOCAL_MEM_FENCE);
        __jm[lid] = jm;
        __jrx[lid] = jrx;
        __jry[lid] = jry;
        __jrz[lid] = jrz;
        __je2[lid] = je2;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (UINT k = 0; k < lsize; ++k) {
            jm = __jm[k];
            jrx = __jrx[k];
            jry = __jry[k];
            jrz = __jrz[k];
            je2 = __je2[k];
#if WIDTH == 1
            acc_kernel_core(im, irx, iry, irz, ie2,
                            jm, jrx, jry, jrz, je2,
                            &iax, &iay, &iaz);
#endif
#if WIDTH > 1
            acc_kernel_core(im, irx, iry, irz, ie2,
                            jm.s0, jrx.s0, jry.s0, jrz.s0, je2.s0,
                            &iax, &iay, &iaz);
            acc_kernel_core(im, irx, iry, irz, ie2,
                            jm.s1, jrx.s1, jry.s1, jrz.s1, je2.s1,
                            &iax, &iay, &iaz);
#endif
#if WIDTH > 3
            acc_kernel_core(im, irx, iry, irz, ie2,
                            jm.s2, jrx.s2, jry.s2, jrz.s2, je2.s2,
                            &iax, &iay, &iaz);
            acc_kernel_core(im, irx, iry, irz, ie2,
                            jm.s3, jrx.s3, jry.s3, jrz.s3, je2.s3,
                            &iax, &iay, &iaz);
#endif
#if WIDTH > 7
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
#endif
#if WIDTH > 15
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
#endif
        }
    }
    for (; j < nj; ++j) {
        acc_kernel_core(im, irx, iry, irz, ie2,
                        _jm[j], _jrx[j], _jry[j], _jrz[j], _je2[j],
                        &iax, &iay, &iaz);
    }

    vstoren(iax, i, _iax);
    vstoren(iay, i, _iay);
    vstoren(iaz, i, _iaz);
}


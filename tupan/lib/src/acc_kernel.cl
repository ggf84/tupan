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
    UINT gid = get_global_id(0);
    gid = min(VECTOR_WIDTH * gid, (ni - VECTOR_WIDTH));

    REALn im = vloadn(0, _im + gid);
    REALn irx = vloadn(0, _irx + gid);
    REALn iry = vloadn(0, _iry + gid);
    REALn irz = vloadn(0, _irz + gid);
    REALn ie2 = vloadn(0, _ie2 + gid);
    REALn iax = (REALn)(0);
    REALn iay = (REALn)(0);
    REALn iaz = (REALn)(0);

    UINT j = 0;
#ifdef FAST_LOCAL_MEM
    __local REAL4 __jm[LSIZE];
    __local REAL4 __jrx[LSIZE];
    __local REAL4 __jry[LSIZE];
    __local REAL4 __jrz[LSIZE];
    __local REAL4 __je2[LSIZE];
    UINT stride = min((UINT)(get_local_size(0)), (UINT)(LSIZE));
    #pragma unroll 4
    for (; stride > 0; stride /= 2) {
        UINT lid = get_local_id(0) % stride;
        for (; (j + 4 * stride - 1) < nj; j += 4 * stride) {
            REAL4 jm = vload4(lid, _jm + j);
            REAL4 jrx = vload4(lid, _jrx + j);
            REAL4 jry = vload4(lid, _jry + j);
            REAL4 jrz = vload4(lid, _jrz + j);
            REAL4 je2 = vload4(lid, _je2 + j);
            barrier(CLK_LOCAL_MEM_FENCE);
            __jm[lid] = jm;
            __jrx[lid] = jrx;
            __jry[lid] = jry;
            __jrz[lid] = jrz;
            __je2[lid] = je2;
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll UNROLL
            for (UINT k = 0; k < stride; ++k) {
                jm = __jm[k];
                jrx = __jrx[k];
                jry = __jry[k];
                jrz = __jrz[k];
                je2 = __je2[k];
                acc_kernel_core(im, irx, iry, irz, ie2,
                                jm.s0, jrx.s0, jry.s0, jrz.s0, je2.s0,
                                &iax, &iay, &iaz);
                #pragma unroll
                for (UINT l = 1; l < 4; ++l) {
                    jm = jm.s1230;
                    jrx = jrx.s1230;
                    jry = jry.s1230;
                    jrz = jrz.s1230;
                    je2 = je2.s1230;
                    acc_kernel_core(im, irx, iry, irz, ie2,
                                    jm.s0, jrx.s0, jry.s0, jrz.s0, je2.s0,
                                    &iax, &iay, &iaz);
                }
            }
        }
    }
#endif
    for (; j < nj; ++j) {
        acc_kernel_core(im, irx, iry, irz, ie2,
                        _jm[j], _jrx[j], _jry[j], _jrz[j], _je2[j],
                        &iax, &iay, &iaz);
    }

    vstoren(iax, 0, _iax + gid);
    vstoren(iay, 0, _iay + gid);
    vstoren(iaz, 0, _iaz + gid);
}


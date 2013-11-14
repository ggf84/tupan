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
    UINT gid = get_global_id(0);
    gid = min(VECTOR_WIDTH * gid, (ni - VECTOR_WIDTH));

    CLIGHT clight = CLIGHT_Init(order, inv1, inv2, inv3, inv4, inv5, inv6, inv7);

    REALn im = vloadn(0, _im + gid);
    REALn irx = vloadn(0, _irx + gid);
    REALn iry = vloadn(0, _iry + gid);
    REALn irz = vloadn(0, _irz + gid);
    REALn ie2 = vloadn(0, _ie2 + gid);
    REALn ivx = vloadn(0, _ivx + gid);
    REALn ivy = vloadn(0, _ivy + gid);
    REALn ivz = vloadn(0, _ivz + gid);
    REALn ipnax = (REALn)(0);
    REALn ipnay = (REALn)(0);
    REALn ipnaz = (REALn)(0);

    UINT j = 0;
#ifdef FAST_LOCAL_MEM
    __local REAL4 __jm[LSIZE];
    __local REAL4 __jrx[LSIZE];
    __local REAL4 __jry[LSIZE];
    __local REAL4 __jrz[LSIZE];
    __local REAL4 __je2[LSIZE];
    __local REAL4 __jvx[LSIZE];
    __local REAL4 __jvy[LSIZE];
    __local REAL4 __jvz[LSIZE];
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
            REAL4 jvx = vload4(lid, _jvx + j);
            REAL4 jvy = vload4(lid, _jvy + j);
            REAL4 jvz = vload4(lid, _jvz + j);
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
            #pragma unroll UNROLL
            for (UINT k = 0; k < stride; ++k) {
                jm = __jm[k];
                jrx = __jrx[k];
                jry = __jry[k];
                jrz = __jrz[k];
                je2 = __je2[k];
                jvx = __jvx[k];
                jvy = __jvy[k];
                jvz = __jvz[k];
                pnacc_kernel_core(im, irx, iry, irz,
                                  ie2, ivx, ivy, ivz,
                                  jm.s0, jrx.s0, jry.s0, jrz.s0,
                                  je2.s0, jvx.s0, jvy.s0, jvz.s0,
                                  clight,
                                  &ipnax, &ipnay, &ipnaz);
                #pragma unroll
                for (UINT l = 1; l < 4; ++l) {
                    jm = jm.s1230;
                    jrx = jrx.s1230;
                    jry = jry.s1230;
                    jrz = jrz.s1230;
                    je2 = je2.s1230;
                    jvx = jvx.s1230;
                    jvy = jvy.s1230;
                    jvz = jvz.s1230;
                    pnacc_kernel_core(im, irx, iry, irz,
                                      ie2, ivx, ivy, ivz,
                                      jm.s0, jrx.s0, jry.s0, jrz.s0,
                                      je2.s0, jvx.s0, jvy.s0, jvz.s0,
                                      clight,
                                      &ipnax, &ipnay, &ipnaz);
                }
            }
        }
    }
#endif
    for (; j < nj; ++j) {
        pnacc_kernel_core(im, irx, iry, irz,
                          ie2, ivx, ivy, ivz,
                          _jm[j], _jrx[j], _jry[j], _jrz[j],
                          _je2[j], _jvx[j], _jvy[j], _jvz[j],
                          clight,
                          &ipnax, &ipnay, &ipnaz);
    }

    vstoren(ipnax, 0, _ipnax + gid);
    vstoren(ipnay, 0, _ipnay + gid);
    vstoren(ipnaz, 0, _ipnaz + gid);
}


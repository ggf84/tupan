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
    UINT gid = get_global_id(0);
    gid = min(VECTOR_WIDTH * gid, (ni - VECTOR_WIDTH));

    REALn im = vloadn(0, _im + gid);
    REALn irx = vloadn(0, _irx + gid);
    REALn iry = vloadn(0, _iry + gid);
    REALn irz = vloadn(0, _irz + gid);
    REALn ie2 = vloadn(0, _ie2 + gid);
    REALn ivx = vloadn(0, _ivx + gid);
    REALn ivy = vloadn(0, _ivy + gid);
    REALn ivz = vloadn(0, _ivz + gid);
    REALn iax = vloadn(0, _iax + gid);
    REALn iay = vloadn(0, _iay + gid);
    REALn iaz = vloadn(0, _iaz + gid);
    REALn ijx = vloadn(0, _ijx + gid);
    REALn ijy = vloadn(0, _ijy + gid);
    REALn ijz = vloadn(0, _ijz + gid);
    REALn isx = (REALn)(0);
    REALn isy = (REALn)(0);
    REALn isz = (REALn)(0);
    REALn icx = (REALn)(0);
    REALn icy = (REALn)(0);
    REALn icz = (REALn)(0);

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
    __local REAL4 __jax[LSIZE];
    __local REAL4 __jay[LSIZE];
    __local REAL4 __jaz[LSIZE];
    __local REAL4 __jjx[LSIZE];
    __local REAL4 __jjy[LSIZE];
    __local REAL4 __jjz[LSIZE];
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
            REAL4 jax = vload4(lid, _jax + j);
            REAL4 jay = vload4(lid, _jay + j);
            REAL4 jaz = vload4(lid, _jaz + j);
            REAL4 jjx = vload4(lid, _jjx + j);
            REAL4 jjy = vload4(lid, _jjy + j);
            REAL4 jjz = vload4(lid, _jjz + j);
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
                jax = __jax[k];
                jay = __jay[k];
                jaz = __jaz[k];
                jjx = __jjx[k];
                jjy = __jjy[k];
                jjz = __jjz[k];
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
                    jax = jax.s1230;
                    jay = jay.s1230;
                    jaz = jaz.s1230;
                    jjx = jjx.s1230;
                    jjy = jjy.s1230;
                    jjz = jjz.s1230;
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
                }
            }
        }
    }
#endif
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

    vstoren(isx, 0, _isx + gid);
    vstoren(isy, 0, _isy + gid);
    vstoren(isz, 0, _isz + gid);
    vstoren(icx, 0, _icx + gid);
    vstoren(icy, 0, _icy + gid);
    vstoren(icz, 0, _icz + gid);
}


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
    gid *= ((VW * gid) < ni);

    REALn im = vloadn(gid, _im);
    REALn irx = vloadn(gid, _irx);
    REALn iry = vloadn(gid, _iry);
    REALn irz = vloadn(gid, _irz);
    REALn ie2 = vloadn(gid, _ie2);
    REALn ivx = vloadn(gid, _ivx);
    REALn ivy = vloadn(gid, _ivy);
    REALn ivz = vloadn(gid, _ivz);
    REALn iax = vloadn(gid, _iax);
    REALn iay = vloadn(gid, _iay);
    REALn iaz = vloadn(gid, _iaz);
    REALn ijx = vloadn(gid, _ijx);
    REALn ijy = vloadn(gid, _ijy);
    REALn ijz = vloadn(gid, _ijz);

    REALn isx = (REALn)(0);
    REALn isy = (REALn)(0);
    REALn isz = (REALn)(0);
    REALn icx = (REALn)(0);
    REALn icy = (REALn)(0);
    REALn icz = (REALn)(0);

    UINT j = 0;

    #ifdef FAST_LOCAL_MEM
    UINT lsize = get_local_size(0);
    for (; (j + lsize - 1) < nj; j += lsize) {
        UINT lid = get_local_id(0);
        __local REAL __jm[LSIZE];
        __local REAL __jrx[LSIZE];
        __local REAL __jry[LSIZE];
        __local REAL __jrz[LSIZE];
        __local REAL __je2[LSIZE];
        __local REAL __jvx[LSIZE];
        __local REAL __jvy[LSIZE];
        __local REAL __jvz[LSIZE];
        __local REAL __jax[LSIZE];
        __local REAL __jay[LSIZE];
        __local REAL __jaz[LSIZE];
        __local REAL __jjx[LSIZE];
        __local REAL __jjy[LSIZE];
        __local REAL __jjz[LSIZE];
        __jm[lid] = _jm[j + lid];
        __jrx[lid] = _jrx[j + lid];
        __jry[lid] = _jry[j + lid];
        __jrz[lid] = _jrz[j + lid];
        __je2[lid] = _je2[j + lid];
        __jvx[lid] = _jvx[j + lid];
        __jvy[lid] = _jvy[j + lid];
        __jvz[lid] = _jvz[j + lid];
        __jax[lid] = _jax[j + lid];
        __jay[lid] = _jay[j + lid];
        __jaz[lid] = _jaz[j + lid];
        __jjx[lid] = _jjx[j + lid];
        __jjy[lid] = _jjy[j + lid];
        __jjz[lid] = _jjz[j + lid];
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll UNROLL
        for (UINT k = 0; k < lsize; ++k) {
            snap_crackle_kernel_core(
                im, irx, iry, irz,
                ie2, ivx, ivy, ivz,
                iax, iay, iaz,
                ijx, ijy, ijz,
                __jm[k], __jrx[k], __jry[k], __jrz[k],
                __je2[k], __jvx[k], __jvy[k], __jvz[k],
                __jax[k], __jay[k], __jaz[k],
                __jjx[k], __jjy[k], __jjz[k],
                &isx, &isy, &isz,
                &icx, &icy, &icz);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #endif

    #pragma unroll UNROLL
    for (; j < nj; ++j) {
        snap_crackle_kernel_core(
            im, irx, iry, irz,
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

    vstoren(isx, gid, _isx);
    vstoren(isy, gid, _isy);
    vstoren(isz, gid, _isz);
    vstoren(icx, gid, _icx);
    vstoren(icy, gid, _icy);
    vstoren(icz, gid, _icz);
}


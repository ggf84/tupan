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
    UINT lsize = get_local_size(0);
    UINT lid = get_local_id(0);
    UINT gid = VECTOR_WIDTH * get_global_id(0);
    gid = min(gid, (ni - VECTOR_WIDTH));

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

#ifdef FAST_LOCAL_MEM
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
    for (UINT j = 0; j < nj; j += lsize) {
        lid = min(lid, (nj - j) - 1);
        lsize = min(lsize, (nj - j));
        barrier(CLK_LOCAL_MEM_FENCE);
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
            snap_crackle_kernel_core(im, irx, iry, irz,
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
    }
#else
    for (UINT j = 0; j < nj; ++j) {
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
#endif

    vstoren(isx, 0, _isx + gid);
    vstoren(isy, 0, _isy + gid);
    vstoren(isz, 0, _isz + gid);
    vstoren(icx, 0, _icx + gid);
    vstoren(icy, 0, _icy + gid);
    vstoren(icz, 0, _icz + gid);
}


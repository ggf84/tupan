#include "nreg_kernels_common.h"


__kernel void nreg_Xkernel(
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
    const REAL dt,
    __global REAL * restrict _idrx,
    __global REAL * restrict _idry,
    __global REAL * restrict _idrz,
    __global REAL * restrict _iax,
    __global REAL * restrict _iay,
    __global REAL * restrict _iaz,
    __global REAL * restrict _iu)
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
    REALn idrx = (REALn)(0);
    REALn idry = (REALn)(0);
    REALn idrz = (REALn)(0);
    REALn iax = (REALn)(0);
    REALn iay = (REALn)(0);
    REALn iaz = (REALn)(0);
    REALn iu = (REALn)(0);

    UINT j = 0;
    __local REAL __jm[LSIZE];
    __local REAL __jrx[LSIZE];
    __local REAL __jry[LSIZE];
    __local REAL __jrz[LSIZE];
    __local REAL __je2[LSIZE];
    __local REAL __jvx[LSIZE];
    __local REAL __jvy[LSIZE];
    __local REAL __jvz[LSIZE];
    for (; (j + lsize) < nj; j += lsize) {
        barrier(CLK_LOCAL_MEM_FENCE);
        __jm[lid] = _jm[j + lid];
        __jrx[lid] = _jrx[j + lid];
        __jry[lid] = _jry[j + lid];
        __jrz[lid] = _jrz[j + lid];
        __je2[lid] = _je2[j + lid];
        __jvx[lid] = _jvx[j + lid];
        __jvy[lid] = _jvy[j + lid];
        __jvz[lid] = _jvz[j + lid];
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll UNROLL
        for (UINT k = 0; k < lsize; ++k) {
            nreg_Xkernel_core(dt,
                              im, irx, iry, irz,
                              ie2, ivx, ivy, ivz,
                              __jm[k], __jrx[k], __jry[k], __jrz[k],
                              __je2[k], __jvx[k], __jvy[k], __jvz[k],
                              &idrx, &idry, &idrz,
                              &iax, &iay, &iaz, &iu);
        }
    }
    for (; j < nj; ++j) {
        nreg_Xkernel_core(dt,
                          im, irx, iry, irz,
                          ie2, ivx, ivy, ivz,
                          _jm[j], _jrx[j], _jry[j], _jrz[j],
                          _je2[j], _jvx[j], _jvy[j], _jvz[j],
                          &idrx, &idry, &idrz,
                          &iax, &iay, &iaz, &iu);
    }

    vstoren(idrx, 0, _idrx + gid);
    vstoren(idry, 0, _idry + gid);
    vstoren(idrz, 0, _idrz + gid);
    vstoren(iax, 0, _iax + gid);
    vstoren(iay, 0, _iay + gid);
    vstoren(iaz, 0, _iaz + gid);
    vstoren(im * iu, 0, _iu + gid);
}


__kernel void nreg_Vkernel(
    const UINT ni,
    __global const REAL * restrict _im,
    __global const REAL * restrict _ivx,
    __global const REAL * restrict _ivy,
    __global const REAL * restrict _ivz,
    __global const REAL * restrict _iax,
    __global const REAL * restrict _iay,
    __global const REAL * restrict _iaz,
    const UINT nj,
    __global const REAL * restrict _jm,
    __global const REAL * restrict _jvx,
    __global const REAL * restrict _jvy,
    __global const REAL * restrict _jvz,
    __global const REAL * restrict _jax,
    __global const REAL * restrict _jay,
    __global const REAL * restrict _jaz,
    const REAL dt,
    __global REAL * restrict _idvx,
    __global REAL * restrict _idvy,
    __global REAL * restrict _idvz,
    __global REAL * restrict _ik)
{
    UINT lsize = get_local_size(0);
    UINT lid = get_local_id(0);
    UINT gid = VECTOR_WIDTH * get_global_id(0);
    gid = min(gid, (ni - VECTOR_WIDTH));

    REALn im = vloadn(0, _im + gid);
    REALn ivx = vloadn(0, _ivx + gid);
    REALn ivy = vloadn(0, _ivy + gid);
    REALn ivz = vloadn(0, _ivz + gid);
    REALn iax = vloadn(0, _iax + gid);
    REALn iay = vloadn(0, _iay + gid);
    REALn iaz = vloadn(0, _iaz + gid);
    REALn idvx = (REALn)(0);
    REALn idvy = (REALn)(0);
    REALn idvz = (REALn)(0);
    REALn ik = (REALn)(0);

    UINT j = 0;
    __local REAL __jm[LSIZE];
    __local REAL __jvx[LSIZE];
    __local REAL __jvy[LSIZE];
    __local REAL __jvz[LSIZE];
    __local REAL __jax[LSIZE];
    __local REAL __jay[LSIZE];
    __local REAL __jaz[LSIZE];
    for (; (j + lsize) < nj; j += lsize) {
        barrier(CLK_LOCAL_MEM_FENCE);
        __jm[lid] = _jm[j + lid];
        __jvx[lid] = _jvx[j + lid];
        __jvy[lid] = _jvy[j + lid];
        __jvz[lid] = _jvz[j + lid];
        __jax[lid] = _jax[j + lid];
        __jay[lid] = _jay[j + lid];
        __jaz[lid] = _jaz[j + lid];
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll UNROLL
        for (UINT k = 0; k < lsize; ++k) {
            nreg_Vkernel_core(dt,
                              im, ivx, ivy, ivz,
                              iax, iay, iaz,
                              __jm[k], __jvx[k], __jvy[k], __jvz[k],
                              __jax[k], __jay[k], __jaz[k],
                              &idvx, &idvy, &idvz, &ik);
        }
    }
    for (; j < nj; ++j) {
        nreg_Vkernel_core(dt,
                          im, ivx, ivy, ivz,
                          iax, iay, iaz,
                          _jm[j], _jvx[j], _jvy[j], _jvz[j],
                          _jax[j], _jay[j], _jaz[j],
                          &idvx, &idvy, &idvz, &ik);
    }

    vstoren(idvx, 0, _idvx + gid);
    vstoren(idvy, 0, _idvy + gid);
    vstoren(idvz, 0, _idvz + gid);
    vstoren(im * ik, 0, _ik + gid);
}


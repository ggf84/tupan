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
    UINT lid = get_local_id(0);
    UINT lsize = get_local_size(0);
    UINT i = VW * lsize * get_group_id(0);

    if ((i + VW * lid) >= ni) {
        i -= VW * lid;
    }

    REALn im = vloadn(lid, _im + i);
    REALn irx = vloadn(lid, _irx + i);
    REALn iry = vloadn(lid, _iry + i);
    REALn irz = vloadn(lid, _irz + i);
    REALn ie2 = vloadn(lid, _ie2 + i);
    REALn ivx = vloadn(lid, _ivx + i);
    REALn ivy = vloadn(lid, _ivy + i);
    REALn ivz = vloadn(lid, _ivz + i);

    REALn idrx = (REALn)(0);
    REALn idry = (REALn)(0);
    REALn idrz = (REALn)(0);
    REALn iax = (REALn)(0);
    REALn iay = (REALn)(0);
    REALn iaz = (REALn)(0);
    REALn iu = (REALn)(0);

    UINT j = 0;

    #ifdef FAST_LOCAL_MEM
        __local REAL __jm[LSIZE];
        __local REAL __jrx[LSIZE];
        __local REAL __jry[LSIZE];
        __local REAL __jrz[LSIZE];
        __local REAL __je2[LSIZE];
        __local REAL __jvx[LSIZE];
        __local REAL __jvy[LSIZE];
        __local REAL __jvz[LSIZE];
        for (; (j + lsize - 1) < nj; j += lsize) {
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
                nreg_Xkernel_core(
                    dt,
                    im, irx, iry, irz,
                    ie2, ivx, ivy, ivz,
                    __jm[k], __jrx[k], __jry[k], __jrz[k],
                    __je2[k], __jvx[k], __jvy[k], __jvz[k],
                    &idrx, &idry, &idrz,
                    &iax, &iay, &iaz, &iu);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    #endif

    #pragma unroll UNROLL
    for (; j < nj; ++j) {
        nreg_Xkernel_core(
            dt,
            im, irx, iry, irz,
            ie2, ivx, ivy, ivz,
            _jm[j], _jrx[j], _jry[j], _jrz[j],
            _je2[j], _jvx[j], _jvy[j], _jvz[j],
            &idrx, &idry, &idrz,
            &iax, &iay, &iaz, &iu);
    }

    vstoren(idrx, lid, _idrx + i);
    vstoren(idry, lid, _idry + i);
    vstoren(idrz, lid, _idrz + i);
    vstoren(iax, lid, _iax + i);
    vstoren(iay, lid, _iay + i);
    vstoren(iaz, lid, _iaz + i);
    vstoren(im * iu, lid, _iu + i);
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
    UINT lid = get_local_id(0);
    UINT lsize = get_local_size(0);
    UINT i = VW * lsize * get_group_id(0);

    if ((i + VW * lid) >= ni) {
        i -= VW * lid;
    }

    REALn im = vloadn(lid, _im + i);
    REALn ivx = vloadn(lid, _ivx + i);
    REALn ivy = vloadn(lid, _ivy + i);
    REALn ivz = vloadn(lid, _ivz + i);
    REALn iax = vloadn(lid, _iax + i);
    REALn iay = vloadn(lid, _iay + i);
    REALn iaz = vloadn(lid, _iaz + i);

    REALn idvx = (REALn)(0);
    REALn idvy = (REALn)(0);
    REALn idvz = (REALn)(0);
    REALn ik = (REALn)(0);

    UINT j = 0;

    #ifdef FAST_LOCAL_MEM
        __local REAL __jm[LSIZE];
        __local REAL __jvx[LSIZE];
        __local REAL __jvy[LSIZE];
        __local REAL __jvz[LSIZE];
        __local REAL __jax[LSIZE];
        __local REAL __jay[LSIZE];
        __local REAL __jaz[LSIZE];
        for (; (j + lsize - 1) < nj; j += lsize) {
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
                nreg_Vkernel_core(
                    dt,
                    im, ivx, ivy, ivz,
                    iax, iay, iaz,
                    __jm[k], __jvx[k], __jvy[k], __jvz[k],
                    __jax[k], __jay[k], __jaz[k],
                    &idvx, &idvy, &idvz, &ik);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    #endif

    #pragma unroll UNROLL
    for (; j < nj; ++j) {
        nreg_Vkernel_core(
            dt,
            im, ivx, ivy, ivz,
            iax, iay, iaz,
            _jm[j], _jvx[j], _jvy[j], _jvz[j],
            _jax[j], _jay[j], _jaz[j],
            &idvx, &idvy, &idvz, &ik);
    }

    vstoren(idvx, lid, _idvx + i);
    vstoren(idvy, lid, _idvy + i);
    vstoren(idvz, lid, _idvz + i);
    vstoren(im * ik, lid, _ik + i);
}


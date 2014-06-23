#include "nreg_kernels_common.h"


__kernel
__attribute__((reqd_work_group_size(LSIZE, 1, 1)))
void nreg_Xkernel(
    const UINT ni,
    __global const REALn * restrict _im,
    __global const REALn * restrict _irx,
    __global const REALn * restrict _iry,
    __global const REALn * restrict _irz,
    __global const REALn * restrict _ie2,
    __global const REALn * restrict _ivx,
    __global const REALn * restrict _ivy,
    __global const REALn * restrict _ivz,
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
    __global REALn * restrict _idrx,
    __global REALn * restrict _idry,
    __global REALn * restrict _idrz,
    __global REALn * restrict _iax,
    __global REALn * restrict _iay,
    __global REALn * restrict _iaz,
    __global REALn * restrict _iu)
{
    for (UINT i = LSIZE * get_group_id(0);
         VW * i < ni; i += LSIZE * get_num_groups(0)) {
        UINT lid = get_local_id(0);
        UINT gid = i + lid;
        gid = ((VW * gid) < ni) ? (gid):(0);

        REALn im = _im[gid];
        REALn irx = _irx[gid];
        REALn iry = _iry[gid];
        REALn irz = _irz[gid];
        REALn ie2 = _ie2[gid];
        REALn ivx = _ivx[gid];
        REALn ivy = _ivy[gid];
        REALn ivz = _ivz[gid];

        REALn idrx = (REALn)(0);
        REALn idry = (REALn)(0);
        REALn idrz = (REALn)(0);
        REALn iax = (REALn)(0);
        REALn iay = (REALn)(0);
        REALn iaz = (REALn)(0);
        REALn iu = (REALn)(0);

        UINT j = 0;

        #ifdef FAST_LOCAL_MEM
        for (; (j + LSIZE - 1) < nj; j += LSIZE) {
            barrier(CLK_LOCAL_MEM_FENCE);
            __local REAL __jm[LSIZE];
            __local REAL __jrx[LSIZE];
            __local REAL __jry[LSIZE];
            __local REAL __jrz[LSIZE];
            __local REAL __je2[LSIZE];
            __local REAL __jvx[LSIZE];
            __local REAL __jvy[LSIZE];
            __local REAL __jvz[LSIZE];
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
            for (UINT k = 0; k < LSIZE; ++k) {
                nreg_Xkernel_core(
                    dt,
                    im, irx, iry, irz,
                    ie2, ivx, ivy, ivz,
                    __jm[k], __jrx[k], __jry[k], __jrz[k],
                    __je2[k], __jvx[k], __jvy[k], __jvz[k],
                    &idrx, &idry, &idrz,
                    &iax, &iay, &iaz, &iu);
            }
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

        _idrx[gid] = idrx;
        _idry[gid] = idry;
        _idrz[gid] = idrz;
        _iax[gid] = iax;
        _iay[gid] = iay;
        _iaz[gid] = iaz;
        _iu[gid] = im * iu;
    }
}


__kernel
__attribute__((reqd_work_group_size(LSIZE, 1, 1)))
void nreg_Vkernel(
    const UINT ni,
    __global const REALn * restrict _im,
    __global const REALn * restrict _ivx,
    __global const REALn * restrict _ivy,
    __global const REALn * restrict _ivz,
    __global const REALn * restrict _iax,
    __global const REALn * restrict _iay,
    __global const REALn * restrict _iaz,
    const UINT nj,
    __global const REAL * restrict _jm,
    __global const REAL * restrict _jvx,
    __global const REAL * restrict _jvy,
    __global const REAL * restrict _jvz,
    __global const REAL * restrict _jax,
    __global const REAL * restrict _jay,
    __global const REAL * restrict _jaz,
    const REAL dt,
    __global REALn * restrict _idvx,
    __global REALn * restrict _idvy,
    __global REALn * restrict _idvz,
    __global REALn * restrict _ik)
{
    for (UINT i = LSIZE * get_group_id(0);
         VW * i < ni; i += LSIZE * get_num_groups(0)) {
        UINT lid = get_local_id(0);
        UINT gid = i + lid;
        gid = ((VW * gid) < ni) ? (gid):(0);

        REALn im = _im[gid];
        REALn ivx = _ivx[gid];
        REALn ivy = _ivy[gid];
        REALn ivz = _ivz[gid];
        REALn iax = _iax[gid];
        REALn iay = _iay[gid];
        REALn iaz = _iaz[gid];

        REALn idvx = (REALn)(0);
        REALn idvy = (REALn)(0);
        REALn idvz = (REALn)(0);
        REALn ik = (REALn)(0);

        UINT j = 0;

        #ifdef FAST_LOCAL_MEM
        for (; (j + LSIZE - 1) < nj; j += LSIZE) {
            barrier(CLK_LOCAL_MEM_FENCE);
            __local REAL __jm[LSIZE];
            __local REAL __jvx[LSIZE];
            __local REAL __jvy[LSIZE];
            __local REAL __jvz[LSIZE];
            __local REAL __jax[LSIZE];
            __local REAL __jay[LSIZE];
            __local REAL __jaz[LSIZE];
            __jm[lid] = _jm[j + lid];
            __jvx[lid] = _jvx[j + lid];
            __jvy[lid] = _jvy[j + lid];
            __jvz[lid] = _jvz[j + lid];
            __jax[lid] = _jax[j + lid];
            __jay[lid] = _jay[j + lid];
            __jaz[lid] = _jaz[j + lid];
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll UNROLL
            for (UINT k = 0; k < LSIZE; ++k) {
                nreg_Vkernel_core(
                    dt,
                    im, ivx, ivy, ivz,
                    iax, iay, iaz,
                    __jm[k], __jvx[k], __jvy[k], __jvz[k],
                    __jax[k], __jay[k], __jaz[k],
                    &idvx, &idvy, &idvz, &ik);
            }
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

        _idvx[gid] = idvx;
        _idvy[gid] = idvy;
        _idvz[gid] = idvz;
        _ik[gid] = im * ik;
    }
}


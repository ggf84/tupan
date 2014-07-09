#include "nreg_kernels_common.h"


kernel
__attribute__((reqd_work_group_size(LSIZE, 1, 1)))
void nreg_Xkernel(
    const UINT ni,
    global const REALn * restrict __im,
    global const REALn * restrict __irx,
    global const REALn * restrict __iry,
    global const REALn * restrict __irz,
    global const REALn * restrict __ie2,
    global const REALn * restrict __ivx,
    global const REALn * restrict __ivy,
    global const REALn * restrict __ivz,
    const UINT nj,
    global const REAL * restrict __jm,
    global const REAL * restrict __jrx,
    global const REAL * restrict __jry,
    global const REAL * restrict __jrz,
    global const REAL * restrict __je2,
    global const REAL * restrict __jvx,
    global const REAL * restrict __jvy,
    global const REAL * restrict __jvz,
    const REAL dt,
    global REALn * restrict __idrx,
    global REALn * restrict __idry,
    global REALn * restrict __idrz,
    global REALn * restrict __iax,
    global REALn * restrict __iay,
    global REALn * restrict __iaz,
    global REALn * restrict __iu)
{
    for (UINT i = LSIZE * get_group_id(0);
         VW * i < ni; i += LSIZE * get_num_groups(0)) {
        UINT lid = get_local_id(0);
        UINT gid = i + lid;
        gid = ((VW * gid) < ni) ? (gid):(0);

        REALn im = __im[gid];
        REALn irx = __irx[gid];
        REALn iry = __iry[gid];
        REALn irz = __irz[gid];
        REALn ie2 = __ie2[gid];
        REALn ivx = __ivx[gid];
        REALn ivy = __ivy[gid];
        REALn ivz = __ivz[gid];

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
            local REAL _jm[LSIZE];
            local REAL _jrx[LSIZE];
            local REAL _jry[LSIZE];
            local REAL _jrz[LSIZE];
            local REAL _je2[LSIZE];
            local REAL _jvx[LSIZE];
            local REAL _jvy[LSIZE];
            local REAL _jvz[LSIZE];
            _jm[lid] = __jm[j + lid];
            _jrx[lid] = __jrx[j + lid];
            _jry[lid] = __jry[j + lid];
            _jrz[lid] = __jrz[j + lid];
            _je2[lid] = __je2[j + lid];
            _jvx[lid] = __jvx[j + lid];
            _jvy[lid] = __jvy[j + lid];
            _jvz[lid] = __jvz[j + lid];
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll UNROLL
            for (UINT k = 0; k < LSIZE; ++k) {
                nreg_Xkernel_core(
                    dt,
                    im, irx, iry, irz,
                    ie2, ivx, ivy, ivz,
                    _jm[k], _jrx[k], _jry[k], _jrz[k],
                    _je2[k], _jvx[k], _jvy[k], _jvz[k],
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
                __jm[j], __jrx[j], __jry[j], __jrz[j],
                __je2[j], __jvx[j], __jvy[j], __jvz[j],
                &idrx, &idry, &idrz,
                &iax, &iay, &iaz, &iu);
        }

        __idrx[gid] = idrx;
        __idry[gid] = idry;
        __idrz[gid] = idrz;
        __iax[gid] = iax;
        __iay[gid] = iay;
        __iaz[gid] = iaz;
        __iu[gid] = im * iu;
    }
}


kernel
__attribute__((reqd_work_group_size(LSIZE, 1, 1)))
void nreg_Vkernel(
    const UINT ni,
    global const REALn * restrict __im,
    global const REALn * restrict __ivx,
    global const REALn * restrict __ivy,
    global const REALn * restrict __ivz,
    global const REALn * restrict __iax,
    global const REALn * restrict __iay,
    global const REALn * restrict __iaz,
    const UINT nj,
    global const REAL * restrict __jm,
    global const REAL * restrict __jvx,
    global const REAL * restrict __jvy,
    global const REAL * restrict __jvz,
    global const REAL * restrict __jax,
    global const REAL * restrict __jay,
    global const REAL * restrict __jaz,
    const REAL dt,
    global REALn * restrict __idvx,
    global REALn * restrict __idvy,
    global REALn * restrict __idvz,
    global REALn * restrict __ik)
{
    for (UINT i = LSIZE * get_group_id(0);
         VW * i < ni; i += LSIZE * get_num_groups(0)) {
        UINT lid = get_local_id(0);
        UINT gid = i + lid;
        gid = ((VW * gid) < ni) ? (gid):(0);

        REALn im = __im[gid];
        REALn ivx = __ivx[gid];
        REALn ivy = __ivy[gid];
        REALn ivz = __ivz[gid];
        REALn iax = __iax[gid];
        REALn iay = __iay[gid];
        REALn iaz = __iaz[gid];

        REALn idvx = (REALn)(0);
        REALn idvy = (REALn)(0);
        REALn idvz = (REALn)(0);
        REALn ik = (REALn)(0);

        UINT j = 0;

        #ifdef FAST_LOCAL_MEM
        for (; (j + LSIZE - 1) < nj; j += LSIZE) {
            barrier(CLK_LOCAL_MEM_FENCE);
            local REAL _jm[LSIZE];
            local REAL _jvx[LSIZE];
            local REAL _jvy[LSIZE];
            local REAL _jvz[LSIZE];
            local REAL _jax[LSIZE];
            local REAL _jay[LSIZE];
            local REAL _jaz[LSIZE];
            _jm[lid] = __jm[j + lid];
            _jvx[lid] = __jvx[j + lid];
            _jvy[lid] = __jvy[j + lid];
            _jvz[lid] = __jvz[j + lid];
            _jax[lid] = __jax[j + lid];
            _jay[lid] = __jay[j + lid];
            _jaz[lid] = __jaz[j + lid];
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll UNROLL
            for (UINT k = 0; k < LSIZE; ++k) {
                nreg_Vkernel_core(
                    dt,
                    im, ivx, ivy, ivz,
                    iax, iay, iaz,
                    _jm[k], _jvx[k], _jvy[k], _jvz[k],
                    _jax[k], _jay[k], _jaz[k],
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
                __jm[j], __jvx[j], __jvy[j], __jvz[j],
                __jax[j], __jay[j], __jaz[j],
                &idvx, &idvy, &idvz, &ik);
        }

        __idvx[gid] = idvx;
        __idvy[gid] = idvy;
        __idvz[gid] = idvz;
        __ik[gid] = im * ik;
    }
}


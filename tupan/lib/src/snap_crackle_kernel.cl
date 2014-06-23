#include "snap_crackle_kernel_common.h"


__kernel
__attribute__((reqd_work_group_size(LSIZE, 1, 1)))
void snap_crackle_kernel(
    const UINT ni,
    __global const REALn * restrict _im,
    __global const REALn * restrict _irx,
    __global const REALn * restrict _iry,
    __global const REALn * restrict _irz,
    __global const REALn * restrict _ie2,
    __global const REALn * restrict _ivx,
    __global const REALn * restrict _ivy,
    __global const REALn * restrict _ivz,
    __global const REALn * restrict _iax,
    __global const REALn * restrict _iay,
    __global const REALn * restrict _iaz,
    __global const REALn * restrict _ijx,
    __global const REALn * restrict _ijy,
    __global const REALn * restrict _ijz,
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
    __global REALn * restrict _isx,
    __global REALn * restrict _isy,
    __global REALn * restrict _isz,
    __global REALn * restrict _icx,
    __global REALn * restrict _icy,
    __global REALn * restrict _icz)
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
        REALn iax = _iax[gid];
        REALn iay = _iay[gid];
        REALn iaz = _iaz[gid];
        REALn ijx = _ijx[gid];
        REALn ijy = _ijy[gid];
        REALn ijz = _ijz[gid];

        REALn isx = (REALn)(0);
        REALn isy = (REALn)(0);
        REALn isz = (REALn)(0);
        REALn icx = (REALn)(0);
        REALn icy = (REALn)(0);
        REALn icz = (REALn)(0);

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
            for (UINT k = 0; k < LSIZE; ++k) {
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

        _isx[gid] = isx;
        _isy[gid] = isy;
        _isz[gid] = isz;
        _icx[gid] = icx;
        _icy[gid] = icy;
        _icz[gid] = icz;
    }
}


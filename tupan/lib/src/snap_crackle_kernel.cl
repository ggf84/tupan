#include "snap_crackle_kernel_common.h"


kernel
__attribute__((reqd_work_group_size(LSIZE, 1, 1)))
void snap_crackle_kernel(
    const UINT ni,
    global const REALn * restrict __im,
    global const REALn * restrict __irx,
    global const REALn * restrict __iry,
    global const REALn * restrict __irz,
    global const REALn * restrict __ie2,
    global const REALn * restrict __ivx,
    global const REALn * restrict __ivy,
    global const REALn * restrict __ivz,
    global const REALn * restrict __iax,
    global const REALn * restrict __iay,
    global const REALn * restrict __iaz,
    global const REALn * restrict __ijx,
    global const REALn * restrict __ijy,
    global const REALn * restrict __ijz,
    const UINT nj,
    global const REAL * restrict __jm,
    global const REAL * restrict __jrx,
    global const REAL * restrict __jry,
    global const REAL * restrict __jrz,
    global const REAL * restrict __je2,
    global const REAL * restrict __jvx,
    global const REAL * restrict __jvy,
    global const REAL * restrict __jvz,
    global const REAL * restrict __jax,
    global const REAL * restrict __jay,
    global const REAL * restrict __jaz,
    global const REAL * restrict __jjx,
    global const REAL * restrict __jjy,
    global const REAL * restrict __jjz,
    global REALn * restrict __isx,
    global REALn * restrict __isy,
    global REALn * restrict __isz,
    global REALn * restrict __icx,
    global REALn * restrict __icy,
    global REALn * restrict __icz)
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
        REALn iax = __iax[gid];
        REALn iay = __iay[gid];
        REALn iaz = __iaz[gid];
        REALn ijx = __ijx[gid];
        REALn ijy = __ijy[gid];
        REALn ijz = __ijz[gid];

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
            local REAL _jm[LSIZE];
            local REAL _jrx[LSIZE];
            local REAL _jry[LSIZE];
            local REAL _jrz[LSIZE];
            local REAL _je2[LSIZE];
            local REAL _jvx[LSIZE];
            local REAL _jvy[LSIZE];
            local REAL _jvz[LSIZE];
            local REAL _jax[LSIZE];
            local REAL _jay[LSIZE];
            local REAL _jaz[LSIZE];
            local REAL _jjx[LSIZE];
            local REAL _jjy[LSIZE];
            local REAL _jjz[LSIZE];
            _jm[lid] = __jm[j + lid];
            _jrx[lid] = __jrx[j + lid];
            _jry[lid] = __jry[j + lid];
            _jrz[lid] = __jrz[j + lid];
            _je2[lid] = __je2[j + lid];
            _jvx[lid] = __jvx[j + lid];
            _jvy[lid] = __jvy[j + lid];
            _jvz[lid] = __jvz[j + lid];
            _jax[lid] = __jax[j + lid];
            _jay[lid] = __jay[j + lid];
            _jaz[lid] = __jaz[j + lid];
            _jjx[lid] = __jjx[j + lid];
            _jjy[lid] = __jjy[j + lid];
            _jjz[lid] = __jjz[j + lid];
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll UNROLL
            for (UINT k = 0; k < LSIZE; ++k) {
                snap_crackle_kernel_core(
                    im, irx, iry, irz,
                    ie2, ivx, ivy, ivz,
                    iax, iay, iaz,
                    ijx, ijy, ijz,
                    _jm[k], _jrx[k], _jry[k], _jrz[k],
                    _je2[k], _jvx[k], _jvy[k], _jvz[k],
                    _jax[k], _jay[k], _jaz[k],
                    _jjx[k], _jjy[k], _jjz[k],
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
                __jm[j], __jrx[j], __jry[j], __jrz[j],
                __je2[j], __jvx[j], __jvy[j], __jvz[j],
                __jax[j], __jay[j], __jaz[j],
                __jjx[j], __jjy[j], __jjz[j],
                &isx, &isy, &isz,
                &icx, &icy, &icz);
        }

        __isx[gid] = isx;
        __isy[gid] = isy;
        __isz[gid] = isz;
        __icx[gid] = icx;
        __icy[gid] = icy;
        __icz[gid] = icz;
    }
}


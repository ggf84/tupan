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
    __global REAL * restrict _icz,
    __local REAL *__jm,
    __local REAL *__jrx,
    __local REAL *__jry,
    __local REAL *__jrz,
    __local REAL *__je2,
    __local REAL *__jvx,
    __local REAL *__jvy,
    __local REAL *__jvz,
    __local REAL *__jax,
    __local REAL *__jay,
    __local REAL *__jaz,
    __local REAL *__jjx,
    __local REAL *__jjy,
    __local REAL *__jjz)
{
    UINT i = get_global_id(0);

    REALn im = vloadn(i, _im);
    REALn irx = vloadn(i, _irx);
    REALn iry = vloadn(i, _iry);
    REALn irz = vloadn(i, _irz);
    REALn ie2 = vloadn(i, _ie2);
    REALn ivx = vloadn(i, _ivx);
    REALn ivy = vloadn(i, _ivy);
    REALn ivz = vloadn(i, _ivz);
    REALn iax = vloadn(i, _iax);
    REALn iay = vloadn(i, _iay);
    REALn iaz = vloadn(i, _iaz);
    REALn ijx = vloadn(i, _ijx);
    REALn ijy = vloadn(i, _ijy);
    REALn ijz = vloadn(i, _ijz);

    REALn isx = (REALn)(0);
    REALn isy = (REALn)(0);
    REALn isz = (REALn)(0);
    REALn icx = (REALn)(0);
    REALn icy = (REALn)(0);
    REALn icz = (REALn)(0);

    UINT j = 0;
    for (; (j + LSIZE) < nj; j += LSIZE) {
        event_t e[14];
        e[0] = async_work_group_copy(__jm, _jm + j, LSIZE, 0);
        e[1] = async_work_group_copy(__jrx, _jrx + j, LSIZE, 0);
        e[2] = async_work_group_copy(__jry, _jry + j, LSIZE, 0);
        e[3] = async_work_group_copy(__jrz, _jrz + j, LSIZE, 0);
        e[4] = async_work_group_copy(__je2, _je2 + j, LSIZE, 0);
        e[5] = async_work_group_copy(__jvx, _jvx + j, LSIZE, 0);
        e[6] = async_work_group_copy(__jvy, _jvy + j, LSIZE, 0);
        e[7] = async_work_group_copy(__jvz, _jvz + j, LSIZE, 0);
        e[8] = async_work_group_copy(__jax, _jax + j, LSIZE, 0);
        e[9] = async_work_group_copy(__jay, _jay + j, LSIZE, 0);
        e[10] = async_work_group_copy(__jaz, _jaz + j, LSIZE, 0);
        e[11] = async_work_group_copy(__jjx, _jjx + j, LSIZE, 0);
        e[12] = async_work_group_copy(__jjy, _jjy + j, LSIZE, 0);
        e[13] = async_work_group_copy(__jjz, _jjz + j, LSIZE, 0);
        wait_group_events(14, e);
        for (UINT k = 0; k < LSIZE; ++k) {
            snap_crackle_kernel_core(im, irx, iry, irz,
                                     ie2, ivx, ivy, ivz,
                                     iax, iay, iaz, ijx, ijy, ijz,
                                     __jm[k], __jrx[k], __jry[k], __jrz[k],
                                     __je2[k], __jvx[k], __jvy[k], __jvz[k],
                                     __jax[k], __jay[k], __jaz[k],
                                     __jjx[k], __jjy[k], __jjz[k],
                                     &isx, &isy, &isz,
                                     &icx, &icy, &icz);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (; j < nj; ++j) {
        snap_crackle_kernel_core(im, irx, iry, irz,
                                 ie2, ivx, ivy, ivz,
                                 iax, iay, iaz, ijx, ijy, ijz,
                                 _jm[j], _jrx[j], _jry[j], _jrz[j],
                                 _je2[j], _jvx[j], _jvy[j], _jvz[j],
                                 _jax[j], _jay[j], _jaz[j],
                                 _jjx[j], _jjy[j], _jjz[j],
                                 &isx, &isy, &isz,
                                 &icx, &icy, &icz);
    }

    vstoren(isx, i, _isx);
    vstoren(isy, i, _isy);
    vstoren(isz, i, _isz);
    vstoren(icx, i, _icx);
    vstoren(icy, i, _icy);
    vstoren(icz, i, _icz);
}


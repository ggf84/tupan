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

    for (UINT j = 0; j < nj; ++j) {
        REALn jm = (REALn)(_jm[j]);
        REALn jrx = (REALn)(_jrx[j]);
        REALn jry = (REALn)(_jry[j]);
        REALn jrz = (REALn)(_jrz[j]);
        REALn je2 = (REALn)(_je2[j]);
        REALn jvx = (REALn)(_jvx[j]);
        REALn jvy = (REALn)(_jvy[j]);
        REALn jvz = (REALn)(_jvz[j]);
        REALn jax = (REALn)(_jax[j]);
        REALn jay = (REALn)(_jay[j]);
        REALn jaz = (REALn)(_jaz[j]);
        REALn jjx = (REALn)(_jjx[j]);
        REALn jjy = (REALn)(_jjy[j]);
        REALn jjz = (REALn)(_jjz[j]);
        snap_crackle_kernel_core(im, irx, iry, irz,
                                 ie2, ivx, ivy, ivz,
                                 iax, iay, iaz, ijx, ijy, ijz,
                                 jm, jrx, jry, jrz,
                                 je2, jvx, jvy, jvz,
                                 jax, jay, jaz, jjx, jjy, jjz,
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


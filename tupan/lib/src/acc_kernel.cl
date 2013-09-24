#include "acc_kernel_common.h"


#define LSIZE 64
#define WITH_TILES


__kernel void acc_kernel(
    const UINT ni,
    __global const REAL * restrict _im,
    __global const REAL * restrict _irx,
    __global const REAL * restrict _iry,
    __global const REAL * restrict _irz,
    __global const REAL * restrict _ie2,
    const UINT nj,
    __global const REAL * restrict _jm,
    __global const REAL * restrict _jrx,
    __global const REAL * restrict _jry,
    __global const REAL * restrict _jrz,
    __global const REAL * restrict _je2,
    __global REAL * restrict _iax,
    __global REAL * restrict _iay,
    __global REAL * restrict _iaz,
    __local REAL * __jm,
    __local REAL * __jrx,
    __local REAL * __jry,
    __local REAL * __jrz,
    __local REAL * __je2)
{
    UINT i = get_global_id(0);

    REALn im = vloadn(i, _im);
    REALn irx = vloadn(i, _irx);
    REALn iry = vloadn(i, _iry);
    REALn irz = vloadn(i, _irz);
    REALn ie2 = vloadn(i, _ie2);

    REALn iax = (REALn)(0);
    REALn iay = (REALn)(0);
    REALn iaz = (REALn)(0);

    UINT tile = 0;

#ifdef WITH_TILES

    UINT ntiles = nj / LSIZE;
    for (; tile < ntiles; tile += LSIZE) {

//        REALn ___jm[LSIZE],
//              ___jrx[LSIZE],
//              ___jry[LSIZE],
//              ___jrz[LSIZE],
//              ___je2[LSIZE];
//        for (UINT l = 0; l < LSIZE; ++l) ___jm[l] = _jm[tile + l];
//        for (UINT l = 0; l < LSIZE; ++l) ___jrx[l] = _jrx[tile + l];
//        for (UINT l = 0; l < LSIZE; ++l) ___jry[l] = _jry[tile + l];
//        for (UINT l = 0; l < LSIZE; ++l) ___jrz[l] = _jrz[tile + l];
//        for (UINT l = 0; l < LSIZE; ++l) ___je2[l] = _je2[tile + l];
//        for (UINT j = 0; j < LSIZE; ++j) {
//            REALn jm = (REALn)(___jm[j]);
//            REALn jrx = (REALn)(___jrx[j]);
//            REALn jry = (REALn)(___jry[j]);
//            REALn jrz = (REALn)(___jrz[j]);
//            REALn je2 = (REALn)(___je2[j]);
//            acc_kernel_core(im, irx, iry, irz, ie2,
//                            jm, jrx, jry, jrz, je2,
//                            &iax, &iay, &iaz);
//        }


        event_t e[5];
        e[0] = async_work_group_copy(__jm,  _jm  + tile, (UINT)LSIZE, 0);
        e[1] = async_work_group_copy(__jrx,  _jrx  + tile, (UINT)LSIZE, 0);
        e[2] = async_work_group_copy(__jry,  _jry  + tile, (UINT)LSIZE, 0);
        e[3] = async_work_group_copy(__jrz,  _jrz  + tile, (UINT)LSIZE, 0);
        e[4] = async_work_group_copy(__je2,  _je2  + tile, (UINT)LSIZE, 0);
        wait_group_events(5, e);
        for (UINT j = 0; j < LSIZE; ++j) {
            REALn jm = (REALn)(__jm[j]);
            REALn jrx = (REALn)(__jrx[j]);
            REALn jry = (REALn)(__jry[j]);
            REALn jrz = (REALn)(__jrz[j]);
            REALn je2 = (REALn)(__je2[j]);
            acc_kernel_core(im, irx, iry, irz, ie2,
                            jm, jrx, jry, jrz, je2,
                            &iax, &iay, &iaz);
        }
//        barrier(CLK_LOCAL_MEM_FENCE);

    }

#endif

    for (UINT j = tile; j < nj; ++j) {
        REALn jm = (REALn)(_jm[j]);
        REALn jrx = (REALn)(_jrx[j]);
        REALn jry = (REALn)(_jry[j]);
        REALn jrz = (REALn)(_jrz[j]);
        REALn je2 = (REALn)(_je2[j]);
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm, jrx, jry, jrz, je2,
                        &iax, &iay, &iaz);
    }

    vstoren(iax, i, _iax);
    vstoren(iay, i, _iay);
    vstoren(iaz, i, _iaz);
}


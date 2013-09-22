#include "acc_kernel_common.h"


#if WIDTH == 1
#define loadn(dst, src, offset, n) \
    {   \
        UINT nn = n - 1;  \
        UINT i0 = min(offset, nn);   \
        dst = (REALn)(src[i0]);  \
    }
#elif WIDTH == 2
#define loadn(dst, src, offset, n) \
    {   \
        UINT nn = n - 1;  \
        UINT i0 = min(offset, nn);   \
        UINT i1 = min(offset+1, nn);   \
        dst = (REALn)(src[i0], src[i1]);  \
    }
#elif WIDTH == 4
#define loadn(dst, src, offset, n) \
    {   \
        UINT nn = n - 1;  \
        UINT i0 = min(offset, nn);   \
        UINT i1 = min(offset+1, nn);   \
        UINT i2 = min(offset+2, nn);   \
        UINT i3 = min(offset+3, nn);   \
        dst = (REALn)(src[i0], src[i1], src[i2], src[i3]);  \
    }
#elif WIDTH == 8
#define loadn(dst, src, offset, n) \
    {   \
        UINT nn = n - 1;  \
        UINT i0 = min(offset, nn);   \
        UINT i1 = min(offset+1, nn);   \
        UINT i2 = min(offset+2, nn);   \
        UINT i3 = min(offset+3, nn);   \
        UINT i4 = min(offset+4, nn);   \
        UINT i5 = min(offset+5, nn);   \
        UINT i6 = min(offset+6, nn);   \
        UINT i7 = min(offset+7, nn);   \
        dst = (REALn)(src[i0], src[i1], src[i2], src[i3],   \
                      src[i4], src[i5], src[i6], src[i7]);  \
    }
#endif



inline void acc_kernel_main_loop(
    const REALn im,
    const REALn irx,
    const REALn iry,
    const REALn irz,
    const REALn ie2,
    const UINT nj,
    __global const REAL *_jm,
    __global const REAL *_jrx,
    __global const REAL *_jry,
    __global const REAL *_jrz,
    __global const REAL *_je2,
    REALn *iax,
    REALn *iay,
    REALn *iaz)
{
    for (UINT j = 0; j < nj; j += WIDTH) {
        REALn jm, jrx, jry, jrz, je2;

        loadn(jm, _jm, j, nj);
        loadn(jrx, _jrx, j, nj);
        loadn(jry, _jry, j, nj);
        loadn(jrz, _jrz, j, nj);
        loadn(je2, _je2, j, nj);

#if WIDTH > 1
        UINT kmax = min((UINT)(WIDTH - 1), ((nj - 1) - j));
//        printf("%d %d %d %d %d\n", nj, nb, j, ((nj - 1) - j), kmax);
        for (UINT k = 0; k < kmax; ++k) {
            acc_kernel_core(im, irx, iry, irz, ie2,
                            jm, jrx, jry, jrz, je2,
                            &(*iax), &(*iay), &(*iaz));
#if WIDTH == 2
            UINTn mask = (UINTn)(1, 0);
#elif WIDTH == 4
            UINTn mask = (UINTn)(3, 0, 1, 2);
#endif
            jm = shuffle(jm, mask);
            jrx = shuffle(jrx, mask);
            jry = shuffle(jry, mask);
            jrz = shuffle(jrz, mask);
            je2 = shuffle(je2, mask);
        }
#endif
        acc_kernel_core(im, irx, iry, irz, ie2,
                        jm, jrx, jry, jrz, je2,
                        &(*iax), &(*iay), &(*iaz));
    }
}


__kernel void acc_kernel(
    const UINT ni,
    __global const REAL *_im,
    __global const REAL *_irx,
    __global const REAL *_iry,
    __global const REAL *_irz,
    __global const REAL *_ie2,
    const UINT nj,
    __global const REAL *_jm,
    __global const REAL *_jrx,
    __global const REAL *_jry,
    __global const REAL *_jrz,
    __global const REAL *_je2,
    __global REAL *_iax,
    __global REAL *_iay,
    __global REAL *_iaz,
    __local REAL *__jm,
    __local REAL *__jrx,
    __local REAL *__jry,
    __local REAL *__jrz,
    __local REAL *__je2)
{
    UINT i = get_global_id(0);

    REALn im, irx, iry, irz, ie2;
    im = vloadn(i, _im);
    irx = vloadn(i, _irx);
    iry = vloadn(i, _iry);
    irz = vloadn(i, _irz);
    ie2 = vloadn(i, _ie2);

    REALn iax = (REALn)(0);
    REALn iay = (REALn)(0);
    REALn iaz = (REALn)(0);

    acc_kernel_main_loop(
        im, irx, iry, irz, ie2,
        nj,
        _jm, _jrx, _jry, _jrz, _je2,
        &iax, &iay, &iaz);

    vstoren(iax, i, _iax);
    vstoren(iay, i, _iay);
    vstoren(iaz, i, _iaz);
}


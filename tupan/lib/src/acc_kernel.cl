#include "acc_kernel_common.h"


__kernel void acc_kernel__(
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
    __global REAL * restrict _iaz)
{
    UINT gid = get_global_id(0) * WPT * VW;

    UINT imask[WPT];

    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        imask[i] = (VW * i + gid) < ni;

    REALn im[WPT], irx[WPT], iry[WPT], irz[WPT], ie2[WPT];

    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            im[i] = vloadn(i, _im + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            irx[i] = vloadn(i, _irx + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            iry[i] = vloadn(i, _iry + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            irz[i] = vloadn(i, _irz + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            ie2[i] = vloadn(i, _ie2 + gid);

    REALn iax[WPT], iay[WPT], iaz[WPT];

    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            iax[i] = (REALn)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            iay[i] = (REALn)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            iaz[i] = (REALn)(0);

#ifdef FAST_LOCAL_MEM
    __local REAL __jm[LSIZE];
    __local REAL __jrx[LSIZE];
    __local REAL __jry[LSIZE];
    __local REAL __jrz[LSIZE];
    __local REAL __je2[LSIZE];
    UINT j = 0;
    UINT lid = get_local_id(0);
    for (UINT stride = get_local_size(0); stride > 0; stride /= 2) {
        INT mask = lid < stride;
        for (; (j + stride - 1) < nj; j += stride) {
            if (mask) {
                __jm[lid] = _jm[j + lid];
                __jrx[lid] = _jrx[j + lid];
                __jry[lid] = _jry[j + lid];
                __jrz[lid] = _jrz[j + lid];
                __je2[lid] = _je2[j + lid];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll UNROLL
            for (UINT k = 0; k < stride; ++k) {
                #pragma unroll
                for (UINT i = 0; i < WPT; ++i) {
                    acc_kernel_core(im[i], irx[i], iry[i], irz[i], ie2[i],
                                    __jm[k], __jrx[k], __jry[k], __jrz[k], __je2[k],
                                    &iax[i], &iay[i], &iaz[i]);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
#else
    #pragma unroll UNROLL
    for (UINT j = 0; j < nj; ++j) {
        #pragma unroll
        for (UINT i = 0; i < WPT; ++i) {
            acc_kernel_core(im[i], irx[i], iry[i], irz[i], ie2[i],
                            _jm[j], _jrx[j], _jry[j], _jrz[j], _je2[j],
                            &iax[i], &iay[i], &iaz[i]);
        }
    }
#endif

    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            vstoren(iax[i], i, _iax + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            vstoren(iay[i], i, _iay + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            vstoren(iaz[i], i, _iaz + gid);
}
















__kernel void acc_kernel___(
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
    __global REAL * restrict _iaz)
{
    UINT gid = get_global_id(0);
    gid = min(gid, ni - 1);

    REAL im = _im[gid];
    REAL irx = _irx[gid];
    REAL iry = _iry[gid];
    REAL irz = _irz[gid];
    REAL ie2 = _ie2[gid];

    REALn __iax = (REALn)(0);
    REALn __iay = (REALn)(0);
    REALn __iaz = (REALn)(0);

    UINT j = 0;
#ifdef FAST_LOCAL_MEM
    __local REALn __jm[LSIZE];
    __local REALn __jrx[LSIZE];
    __local REALn __jry[LSIZE];
    __local REALn __jrz[LSIZE];
    __local REALn __je2[LSIZE];
    UINT lid = get_local_id(0);
    UINT lsize = get_local_size(0);
    for (; (j + VW * lsize - 1) < nj; j += VW * lsize) {

        barrier(CLK_LOCAL_MEM_FENCE);

        __jm[lid] = vloadn(lid, _jm + j);
        __jrx[lid] = vloadn(lid, _jrx + j);
        __jry[lid] = vloadn(lid, _jry + j);
        __jrz[lid] = vloadn(lid, _jrz + j);
        __je2[lid] = vloadn(lid, _je2 + j);

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll UNROLL
        for (UINT k = 0; k < lsize; ++k) {
            call(acc_kernel_core, VW)(
                im, irx, iry, irz, ie2,
                __jm[k], __jrx[k], __jry[k], __jrz[k], __je2[k],
                &__iax, &__iay, &__iaz);
        }
    }
#endif
    #pragma unroll UNROLL
    for (; (j + VW - 1) < nj; j += VW) {
        REALn jm = vloadn(0, _jm + j);
        REALn jrx = vloadn(0, _jrx + j);
        REALn jry = vloadn(0, _jry + j);
        REALn jrz = vloadn(0, _jrz + j);
        REALn je2 = vloadn(0, _je2 + j);

        call(acc_kernel_core, VW)(
            im, irx, iry, irz, ie2,
            jm, jrx, jry, jrz, je2,
            &__iax, &__iay, &__iaz);

    }

    REAL iax = (REAL)(0);
    REAL iay = (REAL)(0);
    REAL iaz = (REAL)(0);

#if VW == 1
    iax = __iax;
    iay = __iay;
    iaz = __iaz;
#else
    iax += __iax.s0;
    iay += __iay.s0;
    iaz += __iaz.s0;
    #pragma unroll VW
    for (UINT i = 1; i < VW; ++i) {
        #if VW == 2
        __iax = __iax.s10;
        __iay = __iay.s10;
        __iaz = __iaz.s10;
        #elif VW == 4
        __iax = __iax.s1230;
        __iay = __iay.s1230;
        __iaz = __iaz.s1230;
        #elif VW == 8
        __iax = __iax.s12345670;
        __iay = __iay.s12345670;
        __iaz = __iaz.s12345670;
        #elif VW == 16
        __iax = __iax.s123456789abcdef0;
        __iay = __iay.s123456789abcdef0;
        __iaz = __iaz.s123456789abcdef0;
        #endif
        iax += __iax.s0;
        iay += __iay.s0;
        iaz += __iaz.s0;
    }
#endif

    #pragma unroll UNROLL
    for (; j < nj; ++j) {
        call(acc_kernel_core, 1)(
            im, irx, iry, irz, ie2,
            _jm[j], _jrx[j], _jry[j], _jrz[j], _je2[j],
            &iax, &iay, &iaz);
    }

    _iax[gid] = iax;
    _iay[gid] = iay;
    _iaz[gid] = iaz;
}
















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
    __global REAL * restrict _iaz)
{
    for (UINT gid = get_global_id(0); gid * VW < ni; gid += get_global_size(0)) {

        UINT i[VW];
        i[0] = gid * VW;
        i[0] = min(i[0], ni - 1);
        #pragma unroll VW
        for (UINT ii = 1; ii < VW; ++ii) {
            i[ii] = i[ii-1] + 1;
            i[ii] = min(i[ii], ni - 1);
        }

        concat(REAL, VW) im, irx, iry, irz, ie2;

        #if VW == 1
        im = _im[i[0]];
        #else
        #pragma unroll VW
        for (UINT ii = 0; ii < VW; ++ii) {
            im.s0 = _im[i[ii]];
            #if VW == 2
            im = im.s10;
            #elif VW == 4
            im = im.s1230;
            #elif VW == 8
            im = im.s12345670;
            #elif VW == 16
            im = im.s123456789abcdef0;
            #endif
        }
        #endif

        #if VW == 1
        irx = _irx[i[0]];
        #else
        #pragma unroll VW
        for (UINT ii = 0; ii < VW; ++ii) {
            irx.s0 = _irx[i[ii]];
            #if VW == 2
            irx = irx.s10;
            #elif VW == 4
            irx = irx.s1230;
            #elif VW == 8
            irx = irx.s12345670;
            #elif VW == 16
            irx = irx.s123456789abcdef0;
            #endif
        }
        #endif

        #if VW == 1
        iry = _iry[i[0]];
        #else
        #pragma unroll VW
        for (UINT ii = 0; ii < VW; ++ii) {
            iry.s0 = _iry[i[ii]];
            #if VW == 2
            iry = iry.s10;
            #elif VW == 4
            iry = iry.s1230;
            #elif VW == 8
            iry = iry.s12345670;
            #elif VW == 16
            iry = iry.s123456789abcdef0;
            #endif
        }
        #endif

        #if VW == 1
        irz = _irz[i[0]];
        #else
        #pragma unroll VW
        for (UINT ii = 0; ii < VW; ++ii) {
            irz.s0 = _irz[i[ii]];
            #if VW == 2
            irz = irz.s10;
            #elif VW == 4
            irz = irz.s1230;
            #elif VW == 8
            irz = irz.s12345670;
            #elif VW == 16
            irz = irz.s123456789abcdef0;
            #endif
        }
        #endif

        #if VW == 1
        ie2 = _ie2[i[0]];
        #else
        #pragma unroll VW
        for (UINT ii = 0; ii < VW; ++ii) {
            ie2.s0 = _ie2[i[ii]];
            #if VW == 2
            ie2 = ie2.s10;
            #elif VW == 4
            ie2 = ie2.s1230;
            #elif VW == 8
            ie2 = ie2.s12345670;
            #elif VW == 16
            ie2 = ie2.s123456789abcdef0;
            #endif
        }
        #endif

        concat(REAL, VW) iax = (concat(REAL, VW))(0);
        concat(REAL, VW) iay = (concat(REAL, VW))(0);
        concat(REAL, VW) iaz = (concat(REAL, VW))(0);

        UINT j = 0;

    #define WW 4

        #if WW > 1
        for (; (j + WW - 1) < nj; j += WW) {
            concat(REAL, WW) jm = concat(vload, WW)(0, _jm + j);
            concat(REAL, WW) jrx = concat(vload, WW)(0, _jrx + j);
            concat(REAL, WW) jry = concat(vload, WW)(0, _jry + j);
            concat(REAL, WW) jrz = concat(vload, WW)(0, _jrz + j);
            concat(REAL, WW) je2 = concat(vload, WW)(0, _je2 + j);

            call(acc_kernel_core, VW)(
                im, irx, iry, irz, ie2,
                jm.s0, jrx.s0, jry.s0, jrz.s0, je2.s0,
                &iax, &iay, &iaz);

            #pragma unroll WW
            for (UINT i = 1; i < WW; ++i) {
                #if WW == 2
                jm = jm.s10;
                #elif WW == 4
                jm = jm.s1230;
                #elif WW == 8
                jm = jm.s12345670;
                #elif WW == 16
                jm = jm.s123456789abcdef0;
                #endif

                #if WW == 2
                jrx = jrx.s10;
                #elif WW == 4
                jrx = jrx.s1230;
                #elif WW == 8
                jrx = jrx.s12345670;
                #elif WW == 16
                jrx = jrx.s123456789abcdef0;
                #endif

                #if WW == 2
                jry = jry.s10;
                #elif WW == 4
                jry = jry.s1230;
                #elif WW == 8
                jry = jry.s12345670;
                #elif WW == 16
                jry = jry.s123456789abcdef0;
                #endif

                #if WW == 2
                jrz = jrz.s10;
                #elif WW == 4
                jrz = jrz.s1230;
                #elif WW == 8
                jrz = jrz.s12345670;
                #elif WW == 16
                jrz = jrz.s123456789abcdef0;
                #endif

                #if WW == 2
                je2 = je2.s10;
                #elif WW == 4
                je2 = je2.s1230;
                #elif WW == 8
                je2 = je2.s12345670;
                #elif WW == 16
                je2 = je2.s123456789abcdef0;
                #endif

                call(acc_kernel_core, VW)(
                    im, irx, iry, irz, ie2,
                    jm.s0, jrx.s0, jry.s0, jrz.s0, je2.s0,
                    &iax, &iay, &iaz);
            }
        }
        #endif


        #pragma unroll UNROLL
        for (; j < nj; ++j) {
            call(acc_kernel_core, VW)(
                im, irx, iry, irz, ie2,
                _jm[j], _jrx[j], _jry[j], _jrz[j], _je2[j],
                &iax, &iay, &iaz);
        }


        #if VW == 1
        _iax[i[0]] = iax;
        #else
        #pragma unroll VW
        for (UINT ii = 0; ii < VW; ++ii) {
            _iax[i[ii]] = iax.s0;
            #if VW == 2
            iax = iax.s10;
            #elif VW == 4
            iax = iax.s1230;
            #elif VW == 8
            iax = iax.s12345670;
            #elif VW == 16
            iax = iax.s123456789abcdef0;
            #endif
        }
        #endif

        #if VW == 1
        _iay[i[0]] = iay;
        #else
        #pragma unroll VW
        for (UINT ii = 0; ii < VW; ++ii) {
            _iay[i[ii]] = iay.s0;
            #if VW == 2
            iay = iay.s10;
            #elif VW == 4
            iay = iay.s1230;
            #elif VW == 8
            iay = iay.s12345670;
            #elif VW == 16
            iay = iay.s123456789abcdef0;
            #endif
        }
        #endif

        #if VW == 1
        _iaz[i[0]] = iaz;
        #else
        #pragma unroll VW
        for (UINT ii = 0; ii < VW; ++ii) {
            _iaz[i[ii]] = iaz.s0;
            #if VW == 2
            iaz = iaz.s10;
            #elif VW == 4
            iaz = iaz.s1230;
            #elif VW == 8
            iaz = iaz.s12345670;
            #elif VW == 16
            iaz = iaz.s123456789abcdef0;
            #endif
        }
        #endif
    }
}


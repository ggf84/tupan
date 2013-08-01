#ifndef __PN_TERMS_H__
#define __PN_TERMS_H__

#include "common.h"

// defines some fractions used in PN expressions
#define _1_2 (((REAL)1)/2)
#define _1_4 (((REAL)1)/4)
#define _3_2 (((REAL)3)/2)
#define _3_4 (((REAL)3)/4)
#define _5_2 (((REAL)5)/2)
#define _5_4 (((REAL)5)/4)
#define _8_5 (((REAL)8)/5)
#define _9_2 (((REAL)9)/2)
#define _12_5 (((REAL)12)/5)
#define _13_12 (((REAL)13)/12)
#define _15_2 (((REAL)15)/2)
#define _15_4 (((REAL)15)/4)
#define _15_8 (((REAL)15)/8)
#define _17_2 (((REAL)17)/2)
#define _24_5 (((REAL)24)/5)
#define _24_7 (((REAL)24)/7)
#define _27_4 (((REAL)27)/4)
#define _32_5 (((REAL)32)/5)
#define _35_16 (((REAL)35)/16)
#define _39_2 (((REAL)39)/2)
#define _41_16 (((REAL)41)/16)
#define _43_2 (((REAL)43)/2)
#define _44_15 (((REAL)44)/15)
#define _48_35 (((REAL)48)/35)
#define _52_15 (((REAL)52)/15)
#define _55_4 (((REAL)55)/4)
#define _56_15 (((REAL)56)/15)
#define _57_4 (((REAL)57)/4)
#define _63_4 (((REAL)63)/4)
#define _69_2 (((REAL)69)/2)
#define _81_8 (((REAL)81)/8)
#define _83_8 (((REAL)83)/8)
#define _90_7 (((REAL)90)/7)
#define _91_2 (((REAL)91)/2)
#define _91_8 (((REAL)91)/8)
#define _95_12 (((REAL)95)/12)
#define _123_32 (((REAL)123)/32)
#define _123_64 (((REAL)123)/64)
#define _132_35 (((REAL)132)/35)
#define _137_8 (((REAL)137)/8)
#define _152_21 (((REAL)152)/21)
#define _152_35 (((REAL)152)/35)
#define _171_2 (((REAL)171)/2)
#define _177_4 (((REAL)177)/4)
#define _180_7 (((REAL)180)/7)
#define _184_21 (((REAL)184)/21)
#define _191_4 (((REAL)191)/4)
#define _204_35 (((REAL)204)/35)
#define _205_2 (((REAL)205)/2)
#define _207_8 (((REAL)207)/8)
#define _208_15 (((REAL)208)/15)
#define _225_2 (((REAL)225)/2)
#define _229_4 (((REAL)229)/4)
#define _243_4 (((REAL)243)/4)
#define _246_35 (((REAL)246)/35)
#define _259_4 (((REAL)259)/4)
#define _269_4 (((REAL)269)/4)
#define _283_2 (((REAL)283)/2)
#define _288_5 (((REAL)288)/5)
#define _292_35 (((REAL)292)/35)
#define _307_8 (((REAL)307)/8)
#define _311_4 (((REAL)311)/4)
#define _334_35 (((REAL)334)/35)
#define _348_5 (((REAL)348)/5)
#define _357_4 (((REAL)357)/4)
#define _357_8 (((REAL)357)/8)
#define _372_5 (((REAL)372)/5)
#define _375_4 (((REAL)375)/4)
#define _383_2 (((REAL)383)/2)
#define _415_8 (((REAL)415)/8)
#define _454_15 (((REAL)454)/15)
#define _455_8 (((REAL)455)/8)
#define _471_8 (((REAL)471)/8)
#define _479_8 (((REAL)479)/8)
#define _534_35 (((REAL)534)/35)
#define _545_3 (((REAL)545)/3)
#define _547_3 (((REAL)547)/3)
#define _565_4 (((REAL)565)/4)
#define _582_5 (((REAL)582)/5)
#define _615_64 (((REAL)615)/64)
#define _654_35 (((REAL)654)/35)
#define _684_5 (((REAL)684)/5)
#define _696_5 (((REAL)696)/5)
#define _723_4 (((REAL)723)/4)
#define _732_35 (((REAL)732)/35)
#define _744_5 (((REAL)744)/5)
#define _854_15 (((REAL)854)/15)
#define _939_4 (((REAL)939)/4)
#define _984_35 (((REAL)984)/35)
#define _1028_21 (((REAL)1028)/21)
#define _1068_35 (((REAL)1068)/35)
#define _1113_8 (((REAL)1113)/8)
#define _1252_35 (((REAL)1252)/35)
#define _1308_35 (((REAL)1308)/35)
#define _1336_35 (((REAL)1336)/35)
#define _1432_35 (((REAL)1432)/35)
#define _1746_5 (((REAL)1746)/5)
#define _1768_105 (((REAL)1768)/105)
#define _1954_5 (((REAL)1954)/5)
#define _2069_8 (((REAL)2069)/8)
#define _2056_21 (((REAL)2056)/21)
#define _2224_21 (((REAL)2224)/21)
#define _2864_35 (((REAL)2864)/35)
#define _2864_105 (((REAL)2864)/105)
#define _2872_21 (((REAL)2872)/21)
#define _3172_21 (((REAL)3172)/21)
#define _3568_105 (((REAL)3568)/105)
#define _3992_105 (((REAL)3992)/105)
#define _4328_105 (((REAL)4328)/105)
#define _4888_105 (((REAL)4888)/105)
#define _5056_105 (((REAL)5056)/105)
#define _5752_105 (((REAL)5752)/105)
#define _5812_105 (((REAL)5812)/105)
#define _6224_105 (((REAL)6224)/105)
#define _6388_105 (((REAL)6388)/105)
#define _10048_105 (((REAL)10048)/105)
#define _13576_105 (((REAL)13576)/105)


inline void p2p_pn2(
    REAL im,
    REAL jm,
    REAL inv_r,
    REAL iv2,
    REAL jv2,
    REAL ivjv,
    REAL niv,
    REAL njv,
    REAL njv2,
    CLIGHT clight,
    PN *pn2)
{
    // Include ~1/c^2 terms (6+2 == 8 terms)
    pn2->a = - iv2
             - 2 * jv2
             + 4 * ivjv
             + _3_2 * njv2
             + inv_r * ( + 5 * im
                         + 4 * jm );

    pn2->b = + 4 * niv
             - 3 * njv;

    pn2->a *= clight.inv2;
    pn2->b *= clight.inv2;
}   // 11+3+2 == 16 FLOPs


inline void p2p_pn4(
    REAL im,
    REAL jm,
    REAL im2,
    REAL jm2,
    REAL imjm,
    REAL inv_r,
    REAL inv_r2,
    REAL iv2,
    REAL jv2,
    REAL jv4,
    REAL ivjv,
    REAL ivjvivjv,
    REAL nv,
    REAL niv,
    REAL njv,
    REAL niv2,
    REAL njv2,
    REAL nivnjv,
    CLIGHT clight,
    PN *pn4)
{
    // Include ~1/c^4 terms (21+10 == 31 terms)
    pn4->a = - 2 * ( + jv4
                     + ivjvivjv )
             + 4 * jv2 * ivjv
             + njv2 * ( + _3_2 * iv2
                        + _9_2 * jv2
                        - 6 * ivjv
                        - _15_8 * njv2 )
             - inv_r2 * ( + _57_4 * im2
                          + 9 * jm2
                          + _69_2 * imjm )
             + inv_r * ( + im * ( - _15_4 * iv2
                                  + _5_4 * jv2
                                  - _5_2 * ivjv
                                  + _39_2 * niv2
                                  - 39 * nivnjv
                                  + _17_2 * njv2 )
                         + jm * ( + 4 * jv2
                                  - 8 * ivjv
                                  + 2 * niv2
                                  - 4 * nivnjv
                                  - 6 * njv2 ) );

    pn4->b = + inv_r * ( + im * ( - _63_4 * niv
                                  + _55_4 * njv )
                         - jm * 2 * ( + niv
                                      + njv ) )
             + iv2 * njv
             - ivjv * nv * 4
             + jv2 * ( + 4 * niv
                       - 5 * njv )
             + njv2 * ( - 6 * niv
                        + _9_2 * njv );

    pn4->a *= clight.inv4;
    pn4->b *= clight.inv4;
}   // 46+24+2 == 72 FLOPs


inline void p2p_pn5(
    REAL im,
    REAL jm,
    REAL inv_r,
    REAL v2,
    REAL nv,
    CLIGHT clight,
    PN *pn5)
{
    // Include ~1/c^5 terms (3+3 == 6 terms)
    pn5->a = + nv * ( + inv_r * ( - _24_5 * im
                                  + _208_15 * jm )
                      + _12_5 * v2 );

    pn5->b = - v2
             + inv_r * ( + _8_5 * im
                         - _32_5 * jm );

    REAL m_c5r = (im * inv_r) * clight.inv5;
    pn5->a *= m_c5r;
    pn5->b *= m_c5r;
}   // 7+5+4 == 16 FLOPs


inline void p2p_pn6(
    REAL im,
    REAL jm,
    REAL im2,
    REAL jm2,
    REAL imjm,
    REAL inv_r,
    REAL inv_r2,
    REAL v2,
    REAL iv2,
    REAL jv2,
    REAL jv4,
    REAL ivjv,
    REAL ivjvivjv,
    REAL nv,
    REAL nvnv,
    REAL niv,
    REAL njv,
    REAL niv2,
    REAL njv2,
    REAL nivnjv,
    CLIGHT clight,
    PN *pn6)
{
    // Include ~1/c^6 terms (66+37 == 103 terms)
    pn6->a = + njv2 * ( + 3 * ivjvivjv
                        + _3_2 * iv2 * jv2
                        - 12 * ivjv * jv2
                        + _15_2 * jv4
                        + njv2 * ( + _15_2 * ( + ivjv
                                               - jv2
                                               - _1_4 * iv2 )
                                   + _35_16 * njv2 ) )
             + 2 * jv2 * ( - ivjvivjv
                           + jv2 * ( + 2 * ivjv
                                     - jv2 ) )
             + im * inv_r * ( + niv * ( + njv * ( + 244 * ivjv
                                                  - _205_2 * iv2
                                                  - _283_2 * jv2
                                                  + _383_2 * njv2 )
                                        + niv * ( + _229_4 * ( + iv2
                                                               + jv2
                                                               - 2 * ivjv )
                                                  - _723_4 * njv2
                                                  + niv * ( + _171_2 * ( + njv
                                                                         - _1_4 * niv ) ) ) )
                              + njv2 * ( + _191_4 * iv2
                                         + _259_4 * jv2
                                         - _225_2 * ivjv
                                         - _455_8 * njv2 )
                              + ivjv * ( + _91_2 * iv2
                                         + 43 * jv2
                                         - _177_4 * ivjv )
                              - _91_8 * iv2 * ( + iv2
                                                + 2 * jv2 )
                              - _81_8 * jv4 )
             + jm * inv_r * 4 * ( + jv4
                                  + njv * ( + niv * ( + ivjv
                                                      - jv2 )
                                            + njv * ( + 3 * ( + ivjv
                                                              - jv2 )
                                                      - _3_2 * niv2
                                                      + njv * ( + 3 * niv
                                                                + _3_2 * njv ) ) )
                                  + ivjv * ( + ivjv
                                             - 2 * jv2 ) )
             + jm2 * inv_r2 * ( - niv2
                                + 2 * nivnjv
                                + _43_2 * njv2
                                + 18 * ivjv
                                - 9 * jv2 )
             + imjm * inv_r2 * ( + _415_8 * niv2
                                 - _375_4 * nivnjv
                                 + _1113_8 * njv2
                                 + 18 * iv2
                                 + PI2 * ( + _123_64 * v2
                                           - _615_64 * nvnv )
                                 + 33 * ( + ivjv
                                          - _1_2 * jv2 ) )
             + im2 * inv_r2 * ( - _2069_8 * niv2
                                + 543 * nivnjv
                                - _939_4 * njv2
                                + _471_8 * iv2
                                + _357_8 * ( + jv2
                                             - 2 * ivjv ) )
             + inv_r * inv_r2 * ( + 16 * jm * jm2
                                  + im2 * jm * ( + _547_3
                                                 - _41_16 * PI2 )
                                  - _13_12 * im * im2
                                  + im * jm2 * ( + _545_3
                                                 - _41_16 * PI2 ) );

    pn6->b = + njv * ( + jv2 * ( + iv2
                                 + 8 * ivjv
                                 - 7 * jv2 )
                       - 2 * ivjvivjv
                       + njv * ( + 6 * niv * ( + ivjv
                                               - 2 * jv2 )
                                 + njv * ( + 6 * ( + 2 * jv2
                                                   - ivjv
                                                   - _1_4 * iv2 )
                                           + njv * ( + _15_2 * ( + niv
                                                                 - _3_4 * njv ) ) ) ) )
             + 4 * niv * ( + jv4
                           - ivjv * jv2 )
             + jm * inv_r * ( + njv * ( + 4 * ( + ivjv
                                                - jv2
                                                - _1_2 * niv2 )
                                        + njv * ( + 2 * ( + 4 * niv
                                                          + njv ) ) )
                              + 2 * niv * ( + ivjv
                                            - jv2 ) )
             + im * inv_r * ( + niv * ( + _207_8 * iv2
                                        + _81_8 * jv2
                                        - 36 * ivjv
                                        - _269_4 * njv2
                                        + niv * ( + _565_4 * njv
                                                  - _243_4 * niv ) )
                              + njv * ( + _83_8 * jv2
                                        + _27_4 * ivjv
                                        - _137_8 * iv2
                                        - _95_12 * njv2 ) )
             + inv_r2 * ( + jm2 * ( + 4 * niv
                                    + 5 * njv )
                          + im2 * ( + _311_4 * niv
                                    - _357_4 * njv )
                          + imjm * ( + _479_8 * njv
                                     - _307_8 * niv
                                     + _123_32 * PI2 * nv ) );

    pn6->a *= clight.inv6;
    pn6->b *= clight.inv6;
}   // ??+??+2 == ??? FLOPs


inline void p2p_pn7(
    REAL im,
    REAL jm,
    REAL im2,
    REAL jm2,
    REAL imjm,
    REAL inv_r,
    REAL inv_r2,
    REAL v2,
    REAL iv2,
    REAL jv2,
    REAL iv4,
    REAL jv4,
    REAL ivjv,
    REAL nv,
    REAL nvnv,
    REAL niv,
    REAL njv,
    REAL niv2,
    REAL njv2,
    REAL nivnjv,
    CLIGHT clight,
    PN *pn7)
{
    // Include ~1/c^7 terms (40+25 == 65 terms)
    pn7->a = + im2 * inv_r2 * ( + _3992_105 * niv
                                - _4328_105 * njv )
             + imjm * inv_r * inv_r2 * ( - _13576_105 * niv
                                         + _2872_21 * njv )
             + jm2 * inv_r * inv_r2 * ( - _3172_21 * nv )
             + im * inv_r * ( + niv * ( + 48 * niv2
                                        - _4888_105 * iv2
                                        + _2056_21 * ivjv
                                        - _1028_21 * jv2 )
                              + nivnjv * ( - _696_5 * niv
                                           + _744_5 * njv )
                              + njv * ( - _288_5 * njv2
                                        + _5056_105 * iv2
                                        - _2224_21 * ivjv
                                        + _5812_105 * jv2 ) )
             + jm * inv_r * ( + niv * ( - _582_5 * niv2
                                        - _2864_35 * ivjv
                                        + _1432_35 * jv2 )
                              + nivnjv * ( + _1746_5 * niv
                                           - _1954_5 * njv )
                              + _3568_105 * nv * iv2
                              + njv * ( + 158 * njv2
                                        - _5752_105 * jv2
                                        + _10048_105 * ivjv ) )
             + ( + nv * ( - 56 * nvnv * nvnv
                          - _246_35 * iv4 )
                 + niv * ( + v2 * ( + 60 * niv2
                                    - 180 * nivnjv
                                    + 174 * njv2 )
                           + ivjv * ( + _1068_35 * ( + iv2
                                                     - ivjv )
                                      + _984_35 * jv2 )
                           - _534_35 * iv2 * jv2
                           - _204_35 * jv4 )
                 + njv * ( - 54 * njv2 * v2
                           + ivjv * ( - _984_35 * iv2
                                      + _180_7 * ivjv
                                      - _732_35 * jv2 )
                           + _90_7 * iv2 * jv2
                           + _24_7 * jv4 ) );

    pn7->b = - im2 * inv_r2 * _184_21
             + imjm * inv_r2 * _6224_105
             + jm2 * inv_r2 * _6388_105
             + im * inv_r * ( + _52_15 * niv2
                              - _56_15 * nivnjv
                              - _44_15 * njv2
                              - _132_35 * iv2
                              + _152_35 * ivjv
                              - _48_35 * jv2 )
             + jm * inv_r * ( + _454_15 * niv2
                              - _372_5 * nivnjv
                              + _854_15 * njv2
                              - _152_21 * iv2
                              + _2864_105 * ivjv
                              - _1768_105 * jv2 )
             + ( + 60 * nvnv * nvnv
                 + v2 * ( - _348_5 * niv2
                          + _684_5 * nivnjv
                          - 66 * njv2 )
                 + _334_35 * iv4
                 + ivjv * ( - _1336_35 * iv2
                            + _1308_35 * ivjv
                            - _1252_35 * jv2 )
                 + _654_35 * iv2 * jv2
                 + _292_35 * jv4 );

    REAL m_c7r = (im * inv_r) * clight.inv7;
    pn7->a *= m_c7r;
    pn7->b *= m_c7r;
}   // ??+??+4 == ??? FLOPs


inline void p2p_pnterms(
    REAL im,
    REAL jm,
    REAL rx,
    REAL ry,
    REAL rz,
    REAL vx,
    REAL vy,
    REAL vz,
    REAL v2,
    REAL ivx,
    REAL ivy,
    REAL ivz,
    REAL jvx,
    REAL jvy,
    REAL jvz,
    REAL inv_r,
    REAL inv_r2,
    REAL inv_r3,
    CLIGHT clight,
    PN *pn)
{
    PN pn1 = {0, 0};
    PN pn2 = {0, 0};
    PN pn3 = {0, 0};
    PN pn4 = {0, 0};
    PN pn5 = {0, 0};
    PN pn6 = {0, 0};
    PN pn7 = {0, 0};

    REAL nx, ny, nz;
    nx = rx * inv_r;                                                 // 1 FLOPs
    ny = ry * inv_r;                                                 // 1 FLOPs
    nz = rz * inv_r;                                                 // 1 FLOPs
    REAL iv2 = ivx * ivx + ivy * ivy + ivz * ivz;                    // 5 FLOPs
    REAL jv2 = jvx * jvx + jvy * jvy + jvz * jvz;                    // 5 FLOPs

    if (clight.order > 0) {
        // XXX: not implemented.
        if (clight.order > 1) {
            REAL niv = nx * ivx + ny * ivy + nz * ivz;               // 5 FLOPs
            REAL njv = nx * jvx + ny * jvy + nz * jvz;               // 5 FLOPs
            REAL njv2 = njv * njv;                                   // 1 FLOPs
            REAL ivjv = ivx * jvx + ivy * jvy + ivz * jvz;           // 5 FLOPs
            p2p_pn2(im, jm, inv_r,
                    iv2, jv2, ivjv,
                    niv, njv, njv2,
                    clight, &pn2);                                   // 16 FLOPs
            if (clight.order > 2) {
                // XXX: not implemented.
                if (clight.order > 3) {
                    REAL im2 = im * im;                              // 1 FLOPs
                    REAL jm2 = jm * jm;                              // 1 FLOPs
                    REAL imjm = im * jm;                             // 1 FLOPs
                    REAL jv4 = jv2 * jv2;                            // 1 FLOPs
                    REAL ivjvivjv = ivjv * ivjv;                     // 1 FLOPs
                    REAL nv = nx * vx + ny * vy + nz * vz;           // 5 FLOPs
                    REAL niv = nx * ivx + ny * ivy + nz * ivz;       // 5 FLOPs
                    REAL njv = nx * jvx + ny * jvy + nz * jvz;       // 5 FLOPs
                    REAL niv2 = niv * niv;                           // 1 FLOPs
                    REAL njv2 = njv * njv;                           // 1 FLOPs
                    REAL nivnjv = niv * njv;                         // 1 FLOPs
                    p2p_pn4(im, jm, im2, jm2, imjm, inv_r, inv_r2,
                            iv2, jv2, jv4, ivjv, ivjvivjv,
                            nv, niv, njv, niv2, njv2, nivnjv,
                            clight, &pn4);                           // 72 FLOPs
                    if (clight.order > 4) {
                        p2p_pn5(im, jm, inv_r, v2, nv, clight, &pn5);// 16 FLOPs
                        if (clight.order > 5) {
                            REAL nvnv = nv * nv;                     // 1 FLOPs
                            p2p_pn6(im, jm, im2, jm2, imjm, inv_r, inv_r2,
                                    v2, iv2, jv2, jv4, ivjv, ivjvivjv,
                                    nv, nvnv, niv, njv, niv2, njv2, nivnjv,
                                    clight, &pn6);                   // ??? FLOPs
                            if (clight.order > 6) {
                                REAL iv4 = iv2 * iv2;                // 1 FLOPs
                                p2p_pn7(im, jm, im2, jm2, imjm, inv_r, inv_r2,
                                        v2, iv2, jv2, iv4, jv4, ivjv,
                                        nv, nvnv, niv, njv, niv2, njv2, nivnjv,
                                        clight, &pn7);               // ??? FLOPs
                            }
                        }
                    }
                }
            }
        }
    }

    // Form the 213 terms post-Newtonian
    // ----> (((((((65   ) + 103  ) + 6    ) + 31   ) + 0    ) + 8    ) + 0    )
    pn->a += (((((((pn7.a) + pn6.a) + pn5.a) + pn4.a) + pn3.a) + pn2.a) + pn1.a);// 7 FLOPs
    pn->b += (((((((pn7.b) + pn6.b) + pn5.b) + pn4.b) + pn3.b) + pn2.b) + pn1.b);// 7 FLOPs

    REAL gm_r3 = jm * inv_r3;                                        // 1 FLOPs
    REAL gm_r2 = jm * inv_r2;                                        // 1 FLOPs
    pn->a *= gm_r3;                                                  // 1 FLOPs
    pn->b *= gm_r2;                                                  // 1 FLOPs
}   // ??+??+?? == ??? FLOPs

#endif  // __PN_TERMS_H__

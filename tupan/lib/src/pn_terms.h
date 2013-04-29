#ifndef __PN_TERMS_H__
#define __PN_TERMS_H__

#include "common.h"

inline PN
p2p_pn2(
    REAL mi,
    REAL mj,
    REAL inv_r,
    REAL iv2,
    REAL jv2,
    REAL ivjv,
    REAL niv,
    REAL njv,
    REAL njv2,
    CLIGHT clight)
{
    // Include ~1/c^2 terms (6+2 == 8 terms)
    PN pn2;

    pn2.a = - iv2
            - 2.0 * jv2
            + 4.0 * ivjv
            + 1.5 * njv2
            + inv_r * ( + 5.0 * mi
                        + 4.0 * mj );

    pn2.b = + 4.0 * niv
            - 3.0 * njv;

    pn2.a *=  clight.inv2;
    pn2.b *=  clight.inv2;

    return pn2;
}   // 11+3+2 == 16 FLOPs


inline PN
p2p_pn4(
    REAL mi,
    REAL mj,
    REAL mi2,
    REAL mj2,
    REAL mimj,
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
    CLIGHT clight)
{
    // Include ~1/c^4 terms (21+10 == 31 terms)
    PN pn4;

    pn4.a = - 2.0 * ( + jv4
                      + ivjvivjv )
            + 4.0 * jv2 * ivjv
            + njv2 * ( + 1.5 * iv2
                       + 4.5 * jv2
                       - 6.0 * ivjv
                       - 1.875 * njv2 )
            - inv_r2 * ( + 14.25 * mi2
                         + 9.0 * mj2
                         + 34.5 * mimj )
            + inv_r * ( + mi * ( - 3.75 * iv2
                                 + 1.25 * jv2
                                 - 2.5 * ivjv
                                 + 19.5 * niv2
                                 - 39.0 * nivnjv
                                 + 8.5 * njv2 )
                        + mj * ( + 4.0 * jv2
                                 - 8.0 * ivjv
                                 + 2.0 * niv2
                                 - 4.0 * nivnjv
                                 - 6.0 * njv2 ) );

    pn4.b = + inv_r * ( + mi * ( - 15.75 * niv
                                 + 13.75 * njv )
                        - mj * 2.0 * ( + niv
                                       + njv ) )
            + iv2 * njv
            - ivjv * nv * 4.0
            + jv2 * ( + 4.0 * niv
                      - 5.0 * njv )
            + njv2 * ( - 6.0 * niv
                       + 4.5 * njv );

    pn4.a *=  clight.inv4;
    pn4.b *=  clight.inv4;

    return pn4;
}   // 46+24+2 == 72 FLOPs


inline PN
p2p_pn5(
    REAL mi,
    REAL mj,
    REAL inv_r,
    REAL v2,
    REAL nv,
    CLIGHT clight)
{
    // Include ~1/c^5 terms (3+3 == 6 terms)
    PN pn5;

    pn5.a = + nv * ( + inv_r * ( - 4.8 * mi
                                 + 1.3866666666666667e+1 * mj )          // +208/15
                     + 2.4 * v2 );

    pn5.b = - v2
            + inv_r * ( + 1.6 * mi
                        - 6.4 * mj );

    REAL m_c5r = (mi * inv_r) * clight.inv5;
    pn5.a *= m_c5r;
    pn5.b *= m_c5r;

    return pn5;
}   // 7+5+4 == 16 FLOPs


inline PN
p2p_pn6(
    REAL mi,
    REAL mj,
    REAL mi2,
    REAL mj2,
    REAL mimj,
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
    CLIGHT clight)
{
    // Include ~1/c^6 terms (66+37 == 103 terms)
    PN pn6;

    pn6.a = + njv2 * ( + 3.0 * ivjvivjv
                       + 1.5 * iv2 * jv2
                       - 12.0 * ivjv * jv2
                       + 7.5 * jv4
                       + njv2 * ( + 7.5 * ( + ivjv
                                            - jv2
                                            - 0.25 * iv2 )
                                  + 2.1875 * njv2 ) )
           + 2.0 * jv2 * ( - ivjvivjv
                           + jv2 * ( + 2.0 * ivjv
                                     - jv2 ) )
           + mi * inv_r * ( + niv * ( + njv * ( + 244.0 * ivjv
                                                - 102.5 * iv2
                                                - 141.5 * jv2
                                                + 191.5 * njv2 )
                                      + niv * ( + 57.25 * ( + iv2
                                                            + jv2
                                                            - 2.0 * ivjv )
                                                - 180.75 * njv2
                                                + niv * ( + 85.5 * ( + njv
                                                                     - 0.25 * niv ) ) ) )
                            + njv2 * ( + 47.75 * iv2
                                       + 64.75 * jv2
                                       - 112.5 * ivjv
                                       - 56.875 * njv2 )
                            + ivjv * ( + 45.5 * iv2
                                       + 43.0 * jv2
                                       - 44.25 * ivjv )
                            - 11.375 * iv2 * ( + iv2
                                               + 2.0 * jv2 )
                            - 10.125 * jv4 )
           + mj * inv_r * 4.0 * ( + jv4
                                  + njv * ( + niv * ( + ivjv
                                                      - jv2 )
                                            + njv * ( + 3.0 * ( + ivjv
                                                                - jv2 )
                                                      - 1.5 * niv2
                                                      + njv * ( + 3.0 * niv
                                                                + 1.5 * njv ) ) )
                                  + ivjv * ( + ivjv
                                             - 2.0 * jv2 ) )
           + mj2 * inv_r2 * ( - niv2
                              + 2.0 * nivnjv
                              + 21.5 * njv2
                              + 18.0 * ivjv
                              - 9.0 * jv2 )
           + mimj * inv_r2 * ( + 51.875 * niv2
                               - 93.75 * nivnjv
                               + 139.125 * njv2
                               + 18.0 * iv2
                               + PI2 * ( + 1.921875 * v2
                                         - 9.609375 * nvnv )
                               + 33.0 * ( + ivjv
                                          - 0.5 * jv2 ) )
           + mi2 * inv_r2 * ( - 258.625 * niv2
                              + 543.0 * nivnjv
                              - 234.75 * njv2
                              + 58.875 * iv2
                              + 44.625 * ( + jv2
                                           - 2.0 * ivjv ) )
           + inv_r * inv_r2 * ( + 16.0 * mj * mj2
                                + mi2 * mj * ( + 1.8233333333333333e+2   // +547/3
                                               - 2.5625 * PI2 )
                                - 1.0833333333333333 * mi * mi2          // -13/12
                                + mi * mj2 * ( + 1.8166666666666667e+2   // +545/3
                                               - 2.5625 * PI2 ) );

	pn6.b = + njv * ( + jv2 * ( + iv2
                                + 8.0 * ivjv
                                - 7.0 * jv2 )
                      - 2.0 * ivjvivjv
                      + njv * ( + 6.0 * niv * ( + ivjv
                                                - 2.0 * jv2 )
                                + njv * ( + 6.0 * ( + 2.0 * jv2
                                                    - ivjv
                                                    - 0.25 * iv2 )
                                          + njv * ( + 7.5 * ( + niv
                                                              - 0.75 * njv ) ) ) ) )
           + 4.0 * niv * ( + jv4
                           - ivjv * jv2 )
           + mj * inv_r * ( + njv * ( + 4.0 * ( + ivjv
                                                - jv2
                                                - 0.5 * niv2 )
                                      + njv * ( + 2.0 * ( + 4.0 * niv
                                                          + njv ) ) )
                            + 2.0 * niv * ( + ivjv
                                            - jv2 ) )
           + mi * inv_r * ( + niv * ( + 25.875 * iv2
                                      + 10.125 * jv2
                                      - 36.0 * ivjv
                                      - 67.25 * njv2
                                      + niv * ( + 141.25 * njv
                                                - 60.75 * niv ) )
                            + njv * ( + 10.375 * jv2
                                      + 6.75 * ivjv
                                      - 17.125 * iv2
                                      - 7.916666666666667 * njv2 ) )     // -95/12
           + inv_r2 * ( + mj2 * ( + 4.0 * niv
                                  + 5.0 * njv )
                        + mi2 * ( + 77.75 * niv
                                  - 89.25 * njv )
                        + mimj * ( + 59.875 * njv
                                   - 38.375 * niv
                                   + 3.84375 * PI2 * nv ) );

    pn6.a *= clight.inv6;
    pn6.b *= clight.inv6;

    return pn6;
}   // ??+??+?? == ??? FLOPs


inline PN
p2p_pn7(
    REAL mi,
    REAL mj,
    REAL mi2,
    REAL mj2,
    REAL mimj,
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
    CLIGHT clight)
{
    // Include ~1/c^7 terms (40+25 == 65 terms)
    PN pn7;

    pn7.a = + mi2 * inv_r2 * ( + 3.801904761904762e+1 * niv              // +3992/105
                               - 4.121904761904762e+1 * njv )            // -4328/105
            + mimj * inv_r * inv_r2 * ( - 1.2929523809523809e+2 * niv    // -13576/105
                                        + 1.3676190476190475e+2 * njv )  // +2872/21
            + mj2 * inv_r * inv_r2 * ( - 1.5104761904761905e+2 * nv )    // -3172/21
            + mi * inv_r * ( + niv * ( + 4.8e+1 * niv2                   // +48
                                       - 4.655238095238095e+1 * iv2      // -4888/105
                                       + 9.79047619047619e+1 * ivjv      // +2056/21
                                       - 4.895238095238095e+1 * jv2 )    // -1028/21
                             + nivnjv * ( - 1.392e+2 * niv               // -696/5
                                          + 1.488e+2 * njv )             // +744/5
                             + njv * ( - 5.76e+1 * njv2                  // -288/5
                                       + 4.8152380952380955e+1 * iv2     // +5056/105
                                       - 1.059047619047619e+2 * ivjv     // -2224/21
                                       + 5.535238095238095e+1 * jv2 ) )  // +5812/105
            + mj * inv_r * ( + niv * ( - 1.164e+2 * niv2                 // -582/5
                                       - 8.182857142857141e+1 * ivjv     // -2864/35
                                       + 4.091428571428571e+1 * jv2 )    // +1432/35
                             + nivnjv * ( + 3.492e+2 * niv               // +1746/5
                                          - 3.908e+2 * njv )             // -1954/5
                             + 3.3980952380952383e+1 * nv * iv2          // +3568/105
                             + njv * ( + 1.58e+2 * njv2                  // +158
                                       - 5.478095238095238e+1 * jv2      // -5752/105
                                       + 9.569523809523808e+1 * ivjv ) ) // +10048/105
            + ( + nv * ( - 5.6e+1 * nvnv * nvnv                          // -56
                         - 7.0285714285714285 * iv4 )                    // -246/35
                + niv * ( + v2 * ( + 6.0e+1 * niv2                       // +60
                                   - 1.80e+2 * nivnjv                    // -180
                                   + 1.74e+2 * njv2 )                    // +174
                          + ivjv * ( + 3.051428571428571e+1 * ( + iv2    // +1068/35
                                                                - ivjv )
                                     + 2.8114285714285714e+1 * jv2 )     // +984/35
                          - 1.5257142857142856e+1 * iv2 * jv2            // -534/35
                          - 5.828571428571428 * jv4 )                    // -204/35
                + njv * ( - 54.0 * njv2 * v2                             // -54
                          + ivjv * ( - 2.8114285714285714e+1 * iv2       // -984/35
                                     + 2.5714285714285716e+1 * ivjv      // +180/7
                                     - 2.0914285714285716e+1 * jv2 )     // -732/35
                          + 1.2857142857142858e+1 * iv2 * jv2            // +90/7
                          + 3.4285714285714284 * jv4 ) );                // +24/7

    pn7.b = - mi2 * inv_r2 * 8.761904761904763                           // -184/21
            + mimj * inv_r2 * 5.927619047619047e+1                       // +6224/105
            + mj2 * inv_r2 * 6.083809523809523e+1                        // +6388/105
            + mi * inv_r * ( + 3.466666666666667 * niv2                  // +52/15
                             - 3.7333333333333334 * nivnjv               // -56/15
                             - 2.933333333333333 * njv2                  // -44/15
                             - 3.7714285714285714 * iv2                  // -132/35
                             + 4.3428571428571425 * ivjv                 // +152/35
                             - 1.3714285714285714 * jv2 )                // -48/35
            + mj * inv_r * ( + 3.0266666666666664e+1 * niv2              // +454/15
                             - 7.44e+1 * nivnjv                          // -372/5
                             + 5.693333333333333e+1 * njv2               // +854/15
                             - 7.238095238095238 * iv2                   // -152/21
                             + 2.7276190476190476e+1 * ivjv              // +2864/105
                             - 1.6838095238095239e+1 * jv2 )             // -1768/105
            + ( + 6.0e+1 * nvnv * nvnv                                   // +60
                + v2 * ( - 6.96e+1 * niv2                                // -348/5
                         + 1.368e+2 * nivnjv                             // +684/5
                         - 6.6e+1 * njv2 )                               // -66
                + 9.542857142857143 * iv4                                // +334/35
                + ivjv * ( - 3.817142857142857e+1 * iv2                  // -1336/35
                           + 3.7371428571428575e+1 * ivjv                // +1308/35
                           - 3.5771428571428574e+1 * jv2 )               // -1252/35
                + 1.8685714285714288e+1 * iv2 * jv2                      // +654/35
                + 8.342857142857143 * jv4 );                             // +292/35

    REAL m_c7r = (mi * inv_r) * clight.inv7;
    pn7.a *= m_c7r;
    pn7.b *= m_c7r;

    return pn7;
}   // ??+??+?? == ??? FLOPs


inline PN
p2p_pnterms(
    REAL mi,
    REAL mj,
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
    CLIGHT clight)
{
    PN pn1 = {0.0, 0.0};
    PN pn2 = {0.0, 0.0};
    PN pn3 = {0.0, 0.0};
    PN pn4 = {0.0, 0.0};
    PN pn5 = {0.0, 0.0};
    PN pn6 = {0.0, 0.0};
    PN pn7 = {0.0, 0.0};

    REAL nx, ny, nz;
    nx = rx * inv_r;                                                // 1 FLOPs
    ny = ry * inv_r;                                                // 1 FLOPs
    nz = rz * inv_r;                                                // 1 FLOPs
    REAL iv2 = ivx * ivx + ivy * ivy + ivz * ivz;
    REAL jv2 = jvx * jvx + jvy * jvy + jvz * jvz;

    if (clight.order > 0) {
        // XXX: not implemented.
        if (clight.order > 1) {
            REAL niv = nx * ivx + ny * ivy + nz * ivz;
            REAL njv = nx * jvx + ny * jvy + nz * jvz;
            REAL njv2 = njv * njv;
            REAL ivjv = ivx * jvx + ivy * jvy + ivz * jvz;
            pn2 = p2p_pn2(mi, mj, inv_r,
                          iv2, jv2, ivjv,
                          niv, njv, njv2,
                          clight);
            if (clight.order > 2) {
                // XXX: not implemented.
                if (clight.order > 3) {
                    REAL mi2 = mi * mi;
                    REAL mj2 = mj * mj;
                    REAL mimj = mi * mj;
                    REAL jv4 = jv2 * jv2;
                    REAL ivjvivjv = ivjv * ivjv;
                    REAL nv = nx * vx + ny * vy + nz * vz;
                    REAL niv = nx * ivx + ny * ivy + nz * ivz;
                    REAL njv = nx * jvx + ny * jvy + nz * jvz;
                    REAL niv2 = niv * niv;
                    REAL njv2 = njv * njv;
                    REAL nivnjv = niv * njv;
                    pn4 = p2p_pn4(mi, mj, mi2, mj2, mimj, inv_r, inv_r2,
                                  iv2, jv2, jv4, ivjv, ivjvivjv,
                                  nv, niv, njv, niv2, njv2, nivnjv,
                                  clight);
                    if (clight.order > 4) {
                        pn5 = p2p_pn5(mi, mj, inv_r, v2, nv, clight);
                        if (clight.order > 5) {
                            REAL nvnv = nv * nv;
                            pn6 = p2p_pn6(mi, mj, mi2, mj2, mimj, inv_r, inv_r2,
                                          v2, iv2, jv2, jv4, ivjv, ivjvivjv,
                                          nv, nvnv, niv, njv, niv2, njv2, nivnjv,
                                          clight);
                            if (clight.order > 6) {
                                REAL iv4 = iv2 * iv2;
                                pn7 = p2p_pn7(mi, mj, mi2, mj2, mimj, inv_r, inv_r2,
                                              v2, iv2, jv2, iv4, jv4, ivjv,
                                              nv, nvnv, niv, njv, niv2, njv2, nivnjv,
                                              clight);
                            }
                        }
                    }
                }
            }
        }
    }

    // Form the 213 terms post-Newtonian
    PN pn = {0.0, 0.0};

    // ---> (((((((65   ) + 103  ) + 6    ) + 31   ) + 0    ) + 8    ) + 0    )
    pn.a += (((((((pn7.a) + pn6.a) + pn5.a) + pn4.a) + pn3.a) + pn2.a) + pn1.a);
    pn.b += (((((((pn7.b) + pn6.b) + pn5.b) + pn4.b) + pn3.b) + pn2.b) + pn1.b);

    REAL gm_r3 = mj * inv_r3;
    REAL gm_r2 = mj * inv_r2;
    pn.a *= gm_r3;
    pn.b *= gm_r2;

    return pn;
}

#endif  // __PN_TERMS_H__

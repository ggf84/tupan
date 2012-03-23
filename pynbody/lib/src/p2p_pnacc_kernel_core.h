#ifndef P2P_PNACC_KERNEL_CORE_H
#define P2P_PNACC_KERNEL_CORE_H

#include"common.h"


inline REAL2
p2p_pn2(REAL mi, REAL mj, REAL inv_r,
        REAL vi2, REAL vj2, REAL vivj,
        REAL nvi, REAL nvj, REAL nvj2,
        CLIGHT clight)
{
    // Include ~1/c^2 terms (6+2 == 8 terms)
    REAL2 pn2;

    pn2.x = - vi2
            - 2.0 * vj2
            + 4.0 * vivj
            + 1.5 * nvj2
            + inv_r * ( + 5.0 * mi
                        + 4.0 * mj );

    pn2.y = + 4.0 * nvi
            - 3.0 * nvj;

    pn2.x *=  clight.inv2;
    pn2.y *=  clight.inv2;

    return pn2;
}   // 11+3+2 == 16 FLOPs


inline REAL2
p2p_pn4(REAL mi, REAL mj, REAL mi2, REAL mj2, REAL mimj, REAL inv_r, REAL inv_r2,
        REAL vi2, REAL vj2, REAL vj4, REAL vivj, REAL vivjvivj,
        REAL nv, REAL nvi, REAL nvj, REAL nvi2, REAL nvj2, REAL nvinvj,
        CLIGHT clight)
{
    // Include ~1/c^4 terms (21+10 == 31 terms)
    REAL2 pn4;

    pn4.x = - 2.0 * ( + vj4
                      + vivjvivj )
            + 4.0 * vj2 * vivj
            + nvj2 * ( + 1.5 * vi2
                       + 4.5 * vj2
                       - 6.0 * vivj
                       - 1.875 * nvj2 )
            - inv_r2 * ( + 14.25 * mi2
                         + 9.0 * mj2
                         + 34.5 * mimj )
            + inv_r * ( + mi * ( - 3.75 * vi2
                                 + 1.25 * vj2
                                 - 2.5 * vivj
                                 + 19.5 * nvi2
                                 - 39.0 * nvinvj
                                 + 8.5 * nvj2 )
                        + mj * ( + 4.0 * vj2
                                 - 8.0 * vivj
                                 + 2.0 * nvi2
                                 - 4.0 * nvinvj
                                 - 6.0 * nvj2 ) );

    pn4.y = + inv_r * ( + mi * ( - 15.75 * nvi
                                 + 13.75 * nvj )
                        - mj * 2.0 * ( + nvi
                                       + nvj ) )
            + vi2 * nvj
            - vivj * nv * 4.0
            + vj2 * ( + 4.0 * nvi
                      - 5.0 * nvj )
            + nvj2 * ( - 6.0 * nvi
                       + 4.5 * nvj );

    pn4.x *=  clight.inv4;
    pn4.y *=  clight.inv4;

    return pn4;
}   // 46+24+2 == 72 FLOPs


inline REAL2
p2p_pn5(REAL mi, REAL mj, REAL inv_r, REAL v2, REAL nv, CLIGHT clight)
{
    // Include ~1/c^5 terms (3+3 == 6 terms)
    REAL2 pn5;

    pn5.x = + nv * ( + inv_r * ( - 4.8 * mi
                                 + 1.3866666666666667e+1 * mj )          // +208/15
                     + 2.4 * v2 );

	pn5.y = - v2
            + inv_r * ( + 1.6 * mi
                        - 6.4 * mj );

    REAL m_c5r = (mi * inv_r) * clight.inv5;
    pn5.x *= m_c5r;
    pn5.y *= m_c5r;

    return pn5;
}   // 7+5+4 == 16 FLOPs


inline REAL2
p2p_pn6(REAL mi, REAL mj, REAL mi2, REAL mj2, REAL mimj, REAL inv_r, REAL inv_r2,
        REAL v2, REAL vi2, REAL vj2, REAL vj4, REAL vivj, REAL vivjvivj,
        REAL nv, REAL nvnv, REAL nvi, REAL nvj, REAL nvi2, REAL nvj2, REAL nvinvj,
        CLIGHT clight)
{
    // Include ~1/c^6 terms (66+37 == 103 terms)
    REAL2 pn6;

    pn6.x = + nvj2 * ( + 3.0 * vivjvivj
                       + 1.5 * vi2 * vj2
                       - 12.0 * vivj * vj2
                       + 7.5 * vj4
                       + nvj2 * ( + 7.5 * ( + vivj
                                            - vj2
                                            - 0.25 * vi2 )
                                  + 2.1875 * nvj2 ) )
           + 2.0 * vj2 * ( - vivjvivj
                           + vj2 * ( + 2.0 * vivj
                                     - vj2 ) )
           + mi * inv_r * ( + nvi * ( + nvj * ( + 244.0 * vivj
                                                - 102.5 * vi2
                                                - 141.5 * vj2
                                                + 191.5 * nvj2 )
                                      + nvi * ( + 57.25 * ( + vi2
                                                            + vj2
                                                            - 2.0 * vivj )
                                                - 180.75 * nvj2
                                                + nvi * ( + 85.5 * ( + nvj
                                                                     - 0.25 * nvi ) ) ) )
                            + nvj2 * ( + 47.75 * vi2
                                       + 64.75 * vj2
                                       - 112.5 * vivj
                                       - 56.875 * nvj2 )
                            + vivj * ( + 45.5 * vi2
                                       + 43.0 * vj2
                                       - 44.25 * vivj )
                            - 11.375 * vi2 * ( + vi2
                                               + 2.0 * vj2 )
                            - 10.125 * vj4 )
           + mj * inv_r * 4.0 * ( + vj4
                                  + nvj * ( + nvi * ( + vivj
                                                      - vj2 )
                                            + nvj * ( + 3.0 * ( + vivj
                                                                - vj2 )
                                                      - 1.5 * nvi2
                                                      + nvj * ( + 3.0 * nvi
                                                                + 1.5 * nvj ) ) )
                                  + vivj * ( + vivj
                                             - 2.0 * vj2 ) )
           + mj2 * inv_r2 * ( - nvi2
                              + 2.0 * nvinvj
                              + 21.5 * nvj2
                              + 18.0 * vivj
                              - 9.0 * vj2 )
           + mimj * inv_r2 * ( + 51.875 * nvi2
                               - 93.75 * nvinvj
                               + 139.125 * nvj2
                               + 18.0 * vi2
                               + PI2 * ( + 1.921875 * v2
                                         - 9.609375 * nvnv )
                               + 33.0 * ( + vivj
                                          - 0.5 * vj2 ) )
           + mi2 * inv_r2 * ( - 258.625 * nvi2
                              + 543.0 * nvinvj
                              - 234.75 * nvj2
                              + 58.875 * vi2
                              + 44.625 * ( + vj2
                                           - 2.0 * vivj ) )
           + inv_r * inv_r2 * ( + 16.0 * mj * mj2
                                + mi2 * mj * ( + 1.8233333333333333e+2   // +547/3
                                               - 2.5625 * PI2 )
                                - 1.0833333333333333 * mi * mi2          // -13/12
                                + mi * mj2 * ( + 1.8166666666666667e+2   // +545/3
                                               - 2.5625 * PI2 ) );

	pn6.y = + nvj * ( + vj2 * ( + vi2
                                + 8.0 * vivj
                                - 7.0 * vj2 )
                      - 2.0 * vivjvivj
                      + nvj * ( + 6.0 * nvi * ( + vivj
                                                - 2.0 * vj2 )
                                + nvj * ( + 6.0 * ( + 2.0 * vj2
                                                    - vivj
                                                    - 0.25 * vi2 )
                                          + nvj * ( + 7.5 * ( + nvi
                                                              - 0.75 * nvj ) ) ) ) )
           + 4.0 * nvi * ( + vj4
                           - vivj * vj2 )
           + mj * inv_r * ( + nvj * ( + 4.0 * ( + vivj
                                                - vj2
                                                - 0.5 * nvi2 )
                                      + nvj * ( + 2.0 * ( + 4.0 * nvi
                                                          + nvj ) ) )
                            + 2.0 * nvi * ( + vivj
                                            - vj2 ) )
           + mi * inv_r * ( + nvi * ( + 25.875 * vi2
                                      + 10.125 * vj2
                                      - 36.0 * vivj
                                      - 67.25 * nvj2
                                      + nvi * ( + 141.25 * nvj
                                                - 60.75 * nvi ) )
                            + nvj * ( + 10.375 * vj2
                                      + 6.75 * vivj
                                      - 17.125 * vi2
                                      - 7.916666666666667 * nvj2 ) )     // -95/12
           + inv_r2 * ( + mj2 * ( + 4.0 * nvi
                                  + 5.0 * nvj )
                        + mi2 * ( + 77.75 * nvi
                                  - 89.25 * nvj )
                        + mimj * ( + 59.875 * nvj
                                   - 38.375 * nvi
                                   + 3.84375 * PI2 * nv ) );

    pn6.x *= clight.inv6;
    pn6.y *= clight.inv6;

    return pn6;
}   // ??+??+?? == ??? FLOPs


inline REAL2
p2p_pn7(REAL mi, REAL mj, REAL mi2, REAL mj2, REAL mimj, REAL inv_r, REAL inv_r2,
        REAL v2, REAL vi2, REAL vj2, REAL vi4, REAL vj4, REAL vivj,
        REAL nv, REAL nvnv, REAL nvi, REAL nvj, REAL nvi2, REAL nvj2, REAL nvinvj,
        CLIGHT clight)
{
    // Include ~1/c^7 terms (40+25 == 65 terms)
    REAL2 pn7;

    pn7.x = + mi2 * inv_r2 * ( + 3.801904761904762e+1 * nvi              // +3992/105
                               - 4.121904761904762e+1 * nvj )            // -4328/105
            + mimj * inv_r * inv_r2 * ( - 1.2929523809523809e+2 * nvi    // -13576/105
                                        + 1.3676190476190475e+2 * nvj )  // +2872/21
            + mj2 * inv_r * inv_r2 * ( - 1.5104761904761905e+2 * nv )    // -3172/21
            + mi * inv_r * ( + nvi * ( + 4.8e+1 * nvi2                   // +48
                                       - 4.655238095238095e+1 * vi2      // -4888/105
                                       + 9.79047619047619e+1 * vivj      // +2056/21
                                       - 4.895238095238095e+1 * vj2 )    // -1028/21
                             + nvinvj * ( - 1.392e+2 * nvi               // -696/5
                                          + 1.488e+2 * nvj )             // +744/5
                             + nvj * ( - 5.76e+1 * nvj2                  // -288/5
                                       + 4.8152380952380955e+1 * vi2     // +5056/105
                                       - 1.059047619047619e+2 * vivj     // -2224/21
                                       + 5.535238095238095e+1 * vj2 ) )  // +5812/105
            + mj * inv_r * ( + nvi * ( - 1.164e+2 * nvi2                 // -582/5
                                       - 8.182857142857141e+1 * vivj     // -2864/35
                                       + 4.091428571428571e+1 * vj2 )    // +1432/35
                             + nvinvj * ( + 3.492e+2 * nvi               // +1746/5
                                          - 3.908e+2 * nvj )             // -1954/5
                             + 3.3980952380952383e+1 * nv * vi2          // +3568/105
                             + nvj * ( + 1.58e+2 * nvj2                  // +158
                                       - 5.478095238095238e+1 * vj2      // -5752/105
                                       + 9.569523809523808e+1 * vivj ) ) // +10048/105
            + ( + nv * ( - 5.6e+1 * nvnv * nvnv                          // -56
                         - 7.0285714285714285 * vi4 )                    // -246/35
                + nvi * ( + v2 * ( + 6.0e+1 * nvi2                       // +60
                                   - 1.80e+2 * nvinvj                    // -180
                                   + 1.74e+2 * nvj2 )                    // +174
                          + vivj * ( + 3.051428571428571e+1 * ( + vi2    // +1068/35
                                                                - vivj )
                                     + 2.8114285714285714e+1 * vj2 )     // +984/35
                          - 1.5257142857142856e+1 * vi2 * vj2            // -534/35
                          - 5.828571428571428 * vj4 )                    // -204/35
                + nvj * ( - 54.0 * nvj2 * v2                             // -54
                          + vivj * ( - 2.8114285714285714e+1 * vi2       // -984/35
                                     + 2.5714285714285716e+1 * vivj      // +180/7
                                     - 2.0914285714285716e+1 * vj2 )     // -732/35
                          + 1.2857142857142858e+1 * vi2 * vj2            // +90/7
                          + 3.4285714285714284 * vj4 ) );                // +24/7

    pn7.y = - mi2 * inv_r2 * 8.761904761904763                           // -184/21
            + mimj * inv_r2 * 5.927619047619047e+1                       // +6224/105
            + mj2 * inv_r2 * 6.083809523809523e+1                        // +6388/105
            + mi * inv_r * ( + 3.466666666666667 * nvi2                  // +52/15
                             - 3.7333333333333334 * nvinvj               // -56/15
                             - 2.933333333333333 * nvj2                  // -44/15
                             - 3.7714285714285714 * vi2                  // -132/35
                             + 4.3428571428571425 * vivj                 // +152/35
                             - 1.3714285714285714 * vj2 )                // -48/35
            + mj * inv_r * ( + 3.0266666666666664e+1 * nvi2              // +454/15
                             - 7.44e+1 * nvinvj                          // -372/5
                             + 5.693333333333333e+1 * nvj2               // +854/15
                             - 7.238095238095238 * vi2                   // -152/21
                             + 2.7276190476190476e+1 * vivj              // +2864/105
                             - 1.6838095238095239e+1 * vj2 )             // -1768/105
            + ( + 6.0e+1 * nvnv * nvnv                                   // +60
                + v2 * ( - 6.96e+1 * nvi2                                // -348/5
                         + 1.368e+2 * nvinvj                             // +684/5
                         - 6.6e+1 * nvj2 )                               // -66
                + 9.542857142857143 * vi4                                // +334/35
                + vivj * ( - 3.817142857142857e+1 * vi2                  // -1336/35
                           + 3.7371428571428575e+1 * vivj                // +1308/35
                           - 3.5771428571428574e+1 * vj2 )               // -1252/35
                + 1.8685714285714288e+1 * vi2 * vj2                      // +654/35
                + 8.342857142857143 * vj4 );                             // +292/35

    REAL m_c7r = (mi * inv_r) * clight.inv7;
    pn7.x *= m_c7r;
    pn7.y *= m_c7r;

    return pn7;
}   // ??+??+?? == ??? FLOPs


inline REAL2
p2p_pnterms(REAL mi, REAL mj,
            REAL inv_r, REAL inv_r2,
            REAL3 n, REAL3 v, REAL v2,
            REAL vi2, REAL3 vi,
            REAL vj2, REAL3 vj,
            CLIGHT clight)
{
    REAL2 pn1 = {0.0, 0.0};
    REAL2 pn2 = {0.0, 0.0};
    REAL2 pn3 = {0.0, 0.0};
    REAL2 pn4 = {0.0, 0.0};
    REAL2 pn5 = {0.0, 0.0};
    REAL2 pn6 = {0.0, 0.0};
    REAL2 pn7 = {0.0, 0.0};
    if (clight.order > 0) {
        // XXX: not implemented.
        if (clight.order > 1) {
            REAL nvi = n.x * vi.x + n.y * vi.y + n.z * vi.z;
            REAL nvj = n.x * vj.x + n.y * vj.y + n.z * vj.z;
            REAL nvj2 = nvj * nvj;
            REAL vivj = vi.x * vj.x + vi.y * vj.y + vi.z * vj.z;
            pn2 = p2p_pn2(mi, mj, inv_r,
                          vi2, vj2, vivj,
                          nvi, nvj, nvj2,
                          clight);
            if (clight.order > 2) {
                // XXX: not implemented.
                if (clight.order > 3) {
                    REAL mi2 = mi * mi;
                    REAL mj2 = mj * mj;
                    REAL mimj = mi * mj;
                    REAL vj4 = vj2 * vj2;
                    REAL vivjvivj = vivj * vivj;
                    REAL nv = n.x * v.x + n.y * v.y + n.z * v.z;
                    REAL nvi = n.x * vi.x + n.y * vi.y + n.z * vi.z;
                    REAL nvj = n.x * vj.x + n.y * vj.y + n.z * vj.z;
                    REAL nvi2 = nvi * nvi;
                    REAL nvj2 = nvj * nvj;
                    REAL nvinvj = nvi * nvj;
                    pn4 = p2p_pn4(mi, mj, mi2, mj2, mimj, inv_r, inv_r2,
                                  vi2, vj2, vj4, vivj, vivjvivj,
                                  nv, nvi, nvj, nvi2, nvj2, nvinvj,
                                  clight);
                    if (clight.order > 4) {
                        pn5 = p2p_pn5(mi, mj, inv_r, v2, nv, clight);
                        if (clight.order > 5) {
                            REAL nvnv = nv * nv;
                            pn6 = p2p_pn6(mi, mj, mi2, mj2, mimj, inv_r, inv_r2,
                                          v2, vi2, vj2, vj4, vivj, vivjvivj,
                                          nv, nvnv, nvi, nvj, nvi2, nvj2, nvinvj,
                                          clight);
                            if (clight.order > 6) {
                                REAL vi4 = vi2 * vi2;
                                pn7 = p2p_pn7(mi, mj, mi2, mj2, mimj, inv_r, inv_r2,
                                              v2, vi2, vj2, vi4, vj4, vivj,
                                              nv, nvnv, nvi, nvj, nvi2, nvj2, nvinvj,
                                              clight);
                            }
                        }
                    }
                }
            }
        }
    }

    // Form the 213 terms post-Newtonian
    REAL2 pn = {0.0, 0.0};

    // ---> (((((((65   ) + 103  ) + 6    ) + 31   ) + 0    ) + 8    ) + 0    )
    pn.x += (((((((pn7.x) + pn6.x) + pn5.x) + pn4.x) + pn3.x) + pn2.x) + pn1.x);
    pn.y += (((((((pn7.y) + pn6.y) + pn5.y) + pn4.y) + pn3.y) + pn2.y) + pn1.y);

    REAL gm_r2 = mj * inv_r2;
    pn.x *= gm_r2;
    pn.y *= gm_r2;

    return pn;
}


inline REAL3
p2p_pnacc_kernel_core(REAL3 pnacc,
                      const REAL4 ri, const REAL4 vi,
                      const REAL4 rj, const REAL4 vj,
                      const CLIGHT clight)
{
    REAL3 r;
    r.x = ri.x - rj.x;                                               // 1 FLOPs
    r.y = ri.y - rj.y;                                               // 1 FLOPs
    r.z = ri.z - rj.z;                                               // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs

    REAL mi = ri.w;
    REAL mj = rj.w;

    REAL3 v;
    v.x = vi.x - vj.x;                                               // 1 FLOPs
    v.y = vi.y - vj.y;                                               // 1 FLOPs
    v.z = vi.z - vj.z;                                               // 1 FLOPs
    REAL v2 = v.x * v.x + v.y * v.y + v.z * v.z;                     // 5 FLOPs

    REAL vi2 = vi.w;
    REAL vj2 = vj.w;

    REAL3 vixyz = {vi.x, vi.y, vi.z};
    REAL3 vjxyz = {vj.x, vj.y, vj.z};

    REAL inv_r2 = 1 / r2;                                            // 1 FLOPs
    inv_r2 = (r2 > 0) ? (inv_r2):(0);
    REAL inv_r = sqrt(inv_r2);                                       // 1 FLOPs

    REAL3 n;
    n.x = r.x * inv_r;                                               // 1 FLOPs
    n.y = r.y * inv_r;                                               // 1 FLOPs
    n.z = r.z * inv_r;                                               // 1 FLOPs

    REAL2 pn = p2p_pnterms(mi, mj,
                           inv_r, inv_r2,
                           n, v, v2,
                           vi2, vixyz,
                           vj2, vjxyz,
                           clight);                                  // ? FLOPs

    pnacc.x += pn.x * n.x + pn.y * v.x;                              // 4 FLOPs
    pnacc.y += pn.x * n.y + pn.y * v.y;                              // 4 FLOPs
    pnacc.z += pn.x * n.z + pn.y * v.z;                              // 4 FLOPs

    return pnacc;
}
// Total flop count: 36

#endif  // P2P_PNACC_KERNEL_CORE_H


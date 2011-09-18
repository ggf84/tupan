inline REAL2
p2p_pn2(REAL mi, REAL mj, REAL rinv,
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
            + rinv * ( + 5.0 * mi
                       + 4.0 * mj );

    pn2.y = + 4.0 * nvi
            - 3.0 * nvj;

    pn2.x *=  clight.inv2;
    pn2.y *=  clight.inv2;

    return pn2;
}   // 11+3+2 == 16 FLOPs


inline REAL2
p2p_pn4(REAL mi, REAL mj, REAL mi2, REAL mj2, REAL mimj, REAL rinv, REAL r2inv,
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
            - r2inv * ( + 14.25 * mi2
                        + 9.0 * mj2
                        + 34.5 * mimj )
            + rinv * ( + mi * ( - 3.75 * vi2
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

    pn4.y = + rinv * ( + mi * ( - 15.75 * nvi
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
p2p_pn5(REAL mi, REAL mj, REAL rinv, REAL v2, REAL nv, CLIGHT clight)
{
    // Include ~1/c^5 terms (3+3 == 6 terms)
    REAL2 pn5;

    pn5.x = + nv * ( + rinv * ( - 4.8 * mi
                                + 1.3866666666666667e+1 * mj )          // +208/15
                     + 2.4 * v2 );

	pn5.y = - v2
            + rinv * ( + 1.6 * mi
                       - 6.4 * mj );

    REAL m_c5r = (mi * rinv) * clight.inv5;
    pn5.x *= m_c5r;
    pn5.y *= m_c5r;

    return pn5;
}   // 7+5+4 == 16 FLOPs


#define PI2	9.869604401089358
inline REAL2
p2p_pn6(REAL mi, REAL mj, REAL mi2, REAL mj2, REAL mimj, REAL rinv, REAL r2inv,
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
           + mi * rinv * ( + nvi * ( + nvj * ( + 244.0 * vivj
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
           + mj * rinv * 4.0 * ( + vj4
                                 + nvj * ( + nvi * ( + vivj
                                                     - vj2 )
                                           + nvj * ( + 3.0 * ( + vivj
                                                               - vj2 )
                                                     - 1.5 * nvi2
                                                     + nvj * ( + 3.0 * nvi
                                                               + 1.5 * nvj ) ) )
                                 + vivj * ( + vivj
                                            - 2.0 * vj2 ) )
           + mj2 * r2inv * ( - nvi2
                             + 2.0 * nvinvj
                             + 21.5 * nvj2
                             + 18.0 * vivj
                             - 9.0 * vj2 )
           + mimj * r2inv * ( + 51.875 * nvi2
                              - 93.75 * nvinvj
                              + 139.125 * nvj2
                              + 18.0 * vi2
                              + PI2 * ( + 1.921875 * v2
                                        - 9.609375 * nvnv )
                              + 33.0 * ( + vivj
                                         - 0.5 * vj2 ) )
           + mi2 * r2inv * ( - 258.625 * nvi2
                             + 543.0 * nvinvj
                             - 234.75 * nvj2
                             + 58.875 * vi2
                             + 44.625 * ( + vj2
                                          - 2.0 * vivj ) )
           + rinv * r2inv * ( + 16.0 * mj * mj2
                              + mi2 * mj * ( + 1.8233333333333333e+2    // +547/3
                                             - 2.5625 * PI2 )
                              - 1.0833333333333333 * mi * mi2           // -13/12
                              + mi * mj2 * ( + 1.8166666666666667e+2    // +545/3
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
           + mj * rinv * ( + nvj * ( + 4.0 * ( + vivj
                                               - vj2
                                               - 0.5 * nvi2 )
                                     + nvj * ( + 2.0 * ( + 4.0 * nvi
                                                         + nvj ) ) )
                           + 2.0 * nvi * ( + vivj
                                           - vj2 ) )
           + mi * rinv * ( + nvi * ( + 25.875 * vi2
                                     + 10.125 * vj2
                                     - 36.0 * vivj
                                     - 67.25 * nvj2
                                     + nvi * ( + 141.25 * nvj
                                               - 60.75 * nvi ) )
                           + nvj * ( + 10.375 * vj2
                                     + 6.75 * vivj
                                     - 17.125 * vi2
                                     - 7.916666666666667 * nvj2 ) )     // -95/12
           + r2inv * ( + mj2 * ( + 4.0 * nvi
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
p2p_pn7(REAL mi, REAL mj, REAL mi2, REAL mj2, REAL mimj, REAL rinv, REAL r2inv,
        REAL v2, REAL vi2, REAL vj2, REAL vi4, REAL vj4, REAL vivj,
        REAL nv, REAL nvnv, REAL nvi, REAL nvj, REAL nvi2, REAL nvj2, REAL nvinvj,
        CLIGHT clight)
{
    // Include ~1/c^7 terms (40+25 == 65 terms)
    REAL2 pn7;

    pn7.x = + mi2 * r2inv * ( + 3.801904761904762e+1 * nvi              // +3992/105
                              - 4.121904761904762e+1 * nvj )            // -4328/105
            + mimj * rinv * r2inv * ( - 1.2929523809523809e+2 * nvi     // -13576/105
                                      + 1.3676190476190475e+2 * nvj )   // +2872/21
            + mj2 * rinv * r2inv * ( - 1.5104761904761905e+2 * nv )     // -3172/21
            + mi * rinv * ( + nvi * ( + 4.8e+1 * nvi2                   // +48
                                      - 4.655238095238095e+1 * vi2      // -4888/105
                                      + 9.79047619047619e+1 * vivj      // +2056/21
                                      - 4.895238095238095e+1 * vj2 )    // -1028/21
                            + nvinvj * ( - 1.392e+2 * nvi               // -696/5
                                         + 1.488e+2 * nvj )             // +744/5
                            + nvj * ( - 5.76e+1 * nvj2                  // -288/5
                                      + 4.8152380952380955e+1 * vi2     // +5056/105
                                      - 1.059047619047619e+2 * vivj     // -2224/21
                                      + 5.535238095238095e+1 * vj2 ) )  // +5812/105
            + mj * rinv * ( + nvi * ( - 1.164e+2 * nvi2                 // -582/5
                                      - 8.182857142857141e+1 * vivj     // -2864/35
                                      + 4.091428571428571e+1 * vj2 )    // +1432/35
                            + nvinvj * ( + 3.492e+2 * nvi               // +1746/5
                                         - 3.908e+2 * nvj )             // -1954/5
                            + 3.3980952380952383e+1 * nv * vi2          // +3568/105
                            + nvj * ( + 1.58e+2 * nvj2                  // +158
                                      - 5.478095238095238e+1 * vj2      // -5752/105
                                      + 9.569523809523808e+1 * vivj ) ) // +10048/105
            + ( + nv * ( - 5.6e+1 * nvnv * nvnv                         // -56
                         - 7.0285714285714285 * vi4 )                   // -246/35
                + nvi * ( + v2 * ( + 6.0e+1 * nvi2                      // +60
                                   - 1.80e+2 * nvinvj                   // -180
                                   + 1.74e+2 * nvj2 )                   // +174
                          + vivj * ( + 3.051428571428571e+1 * ( + vi2   // +1068/35
                                                                - vivj )
                                     + 2.8114285714285714e+1 * vj2 )    // +984/35
                          - 1.5257142857142856e+1 * vi2 * vj2           // -534/35
                          - 5.828571428571428 * vj4 )                   // -204/35
                + nvj * ( - 54.0 * nvj2 * v2                            // -54
                          + vivj * ( - 2.8114285714285714e+1 * vi2      // -984/35
                                     + 2.5714285714285716e+1 * vivj     // +180/7
                                     - 2.0914285714285716e+1 * vj2 )    // -732/35
                          + 1.2857142857142858e+1 * vi2 * vj2           // +90/7
                          + 3.4285714285714284 * vj4 ) );               // +24/7

    pn7.y = - mi2 * r2inv * 8.761904761904763                           // -184/21
            + mimj * r2inv * 5.927619047619047e+1                       // +6224/105
            + mj2 * r2inv * 6.083809523809523e+1                        // +6388/105
            + mi * rinv * ( + 3.466666666666667 * nvi2                  // +52/15
                            - 3.7333333333333334 * nvinvj               // -56/15
                            - 2.933333333333333 * nvj2                  // -44/15
                            - 3.7714285714285714 * vi2                  // -132/35
                            + 4.3428571428571425 * vivj                 // +152/35
                            - 1.3714285714285714 * vj2 )                // -48/35
            + mj * rinv * ( + 3.0266666666666664e+1 * nvi2              // +454/15
                            - 7.44e+1 * nvinvj                          // -372/5
                            + 5.693333333333333e+1 * nvj2               // +854/15
                            - 7.238095238095238 * vi2                   // -152/21
                            + 2.7276190476190476e+1 * vivj              // +2864/105
                            - 1.6838095238095239e+1 * vj2 )             // -1768/105
            + ( + 6.0e+1 * nvnv * nvnv                                  // +60
                + v2 * ( - 6.96e+1 * nvi2                               // -348/5
                         + 1.368e+2 * nvinvj                            // +684/5
                         - 6.6e+1 * nvj2 )                              // -66
                + 9.542857142857143 * vi4                               // +334/35
                + vivj * ( - 3.817142857142857e+1 * vi2                 // -1336/35
                           + 3.7371428571428575e+1 * vivj               // +1308/35
                           - 3.5771428571428574e+1 * vj2 )              // -1252/35
                + 1.8685714285714288e+1 * vi2 * vj2                     // +654/35
                + 8.342857142857143 * vj4 );                            // +292/35

    REAL m_c7r = (mi * rinv) * clight.inv7;
    pn7.x *= m_c7r;
    pn7.y *= m_c7r;

    return pn7;
}   // ??+??+?? == ??? FLOPs


inline REAL2
p2p_pnterms(REAL mi, REAL mj,
            REAL rinv, REAL r2inv,
            REAL3 n, REAL3 v, REAL v2,
            REAL vi2, REAL3 vi,
            REAL vj2, REAL3 vj,
            int pn_order,
            CLIGHT clight)
{
    REAL2 pn1 = {0.0, 0.0};
    REAL2 pn2 = {0.0, 0.0};
    REAL2 pn3 = {0.0, 0.0};
    REAL2 pn4 = {0.0, 0.0};
    REAL2 pn5 = {0.0, 0.0};
    REAL2 pn6 = {0.0, 0.0};
    REAL2 pn7 = {0.0, 0.0};
    if (pn_order > 0) {
        // XXX: not implemented.
        if (pn_order > 1) {
            REAL nvi = n.x * vi.x + n.y * vi.y + n.z * vi.z;
            REAL nvj = n.x * vj.x + n.y * vj.y + n.z * vj.z;
            REAL nvj2 = nvj * nvj;
            REAL vivj = vi.x * vj.x + vi.y * vj.y + vi.z * vj.z;
            pn2 = p2p_pn2(mi, mj, rinv,
                          vi2, vj2, vivj,
                          nvi, nvj, nvj2,
                          clight);
            if (pn_order > 2) {
                // XXX: not implemented.
                if (pn_order > 3) {
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
                    pn4 = p2p_pn4(mi, mj, mi2, mj2, mimj, rinv, r2inv,
                                  vi2, vj2, vj4, vivj, vivjvivj,
                                  nv, nvi, nvj, nvi2, nvj2, nvinvj,
                                  clight);
                    if (pn_order > 4) {
                        pn5 = p2p_pn5(mi, mj, rinv, v2, nv, clight);
                        if (pn_order > 5) {
                            REAL nvnv = nv * nv;
                            pn6 = p2p_pn6(mi, mj, mi2, mj2, mimj, rinv, r2inv,
                                          v2, vi2, vj2, vj4, vivj, vivjvivj,
                                          nv, nvnv, nvi, nvj, nvi2, nvj2, nvinvj,
                                          clight);
                            if (pn_order > 6) {
                                REAL vi4 = vi2 * vi2;
                                pn7 = p2p_pn7(mi, mj, mi2, mj2, mimj, rinv, r2inv,
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

    REAL gm_r2 = mj * r2inv;
    pn.x *= gm_r2;
    pn.y *= gm_r2;

    return pn;
}


inline REAL4
p2p_pnacc_kernel_core(REAL4 pnacc, REAL4 bip, REAL4 biv, REAL4 bjp, REAL4 bjv,
                      int pn_order, CLIGHT clight)
{
    REAL3 dr;
    dr.x = bip.x - bjp.x;                                            // 1 FLOPs
    dr.y = bip.y - bjp.y;                                            // 1 FLOPs
    dr.z = bip.z - bjp.z;                                            // 1 FLOPs
    REAL dr2 = dr.z * dr.z + (dr.y * dr.y + dr.x * dr.x);            // 5 FLOPs

    REAL mi = bip.w;
    REAL mj = bjp.w;

    REAL3 dv;
    dv.x = biv.x - bjv.x;                                            // 1 FLOPs
    dv.y = biv.y - bjv.y;                                            // 1 FLOPs
    dv.z = biv.z - bjv.z;                                            // 1 FLOPs
    REAL dv2 = dv.z * dv.z + (dv.y * dv.y + dv.x * dv.x);            // 5 FLOPs

    REAL vi2 = biv.w;
    REAL vj2 = bjv.w;

    REAL3 vi = {biv.x, biv.y, biv.z};
    REAL3 vj = {bjv.x, bjv.y, bjv.z};

    REAL rinv = rsqrt(dr2);                                          // 2 FLOPs
    rinv = ((dr2 > 0) ? rinv:0);
    REAL r2inv = rinv * rinv;                                        // 1 FLOPs

    REAL3 n;
    n.x = dr.x * rinv;                                               // 1 FLOPs
    n.y = dr.y * rinv;                                               // 1 FLOPs
    n.z = dr.z * rinv;                                               // 1 FLOPs

    REAL2 pn = p2p_pnterms(mi, mj,
                           rinv, r2inv,
                           n, dv, dv2,
                           vi2, vi,
                           vj2, vj,
                           pn_order,
                           clight);                                  // ? FLOPs

    pnacc.x += pn.x * n.x + pn.y * dv.x;                             // 4 FLOPs
    pnacc.y += pn.x * n.y + pn.y * dv.y;                             // 4 FLOPs
    pnacc.z += pn.x * n.z + pn.y * dv.z;                             // 4 FLOPs

    pnacc.w += (mi + mj) * rinv * r2inv;                             // 4 FLOPs

    return pnacc;
}   // Total flop count: 38


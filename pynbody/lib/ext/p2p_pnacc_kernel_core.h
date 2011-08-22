inline REAL2
p2p_pn2(REAL mi, REAL mj, REAL rinv,
        REAL vi2, REAL vj2, REAL vivj,
        REAL nvi, REAL nvj, REAL nvj2,
        CLIGHT clight)
{
    /* Include ~1/c^2 terms (6+2 == 8 terms) */
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
}


inline REAL2
p2p_pn4(REAL mi, REAL mj, REAL mi2, REAL mj2, REAL mimj, REAL rinv, REAL r2inv,
        REAL vi2, REAL vj2, REAL vj4, REAL vivj, REAL vivjvivj,
        REAL nv, REAL nvi, REAL nvj, REAL nvi2, REAL nvj2, REAL nvinvj,
        CLIGHT clight)
{
    /* Include ~1/c^4 terms (21+10 == 31 terms) */
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
}




inline REAL2
p2p_pn5(REAL mi, REAL mj, REAL rinv, REAL v2, REAL nv, CLIGHT clight)
{
    /* Include ~1/c^5 terms (3+3 == 6 terms) */
    REAL2 pn5;

    pn5.x = nv * ( + rinv * ( - 4.8 * mi
                              + 1.3866666666666667e+1 * mj )        /* 208/15 */
                   + 2.4 * v2 );

	pn5.y = - v2
            + rinv * ( + 1.6 * mi
                       - 6.4 * mj );

    REAL m_c5r = (mi * rinv) * clight.inv5;
    pn5.x *= m_c5r;
    pn5.y *= m_c5r;

    return pn5;
}



inline REAL2
p2p_pnterms(REAL mi, REAL mj,
            REAL rinv, REAL r2inv,
            REAL3 n, REAL3 v, REAL v2,
            REAL vi2, REAL3 vi,
            REAL vj2, REAL3 vj,
            CLIGHT clight,
            int pn_order)
{
    REAL2 pn1 = {0.0, 0.0};
    REAL2 pn2 = {0.0, 0.0};
    REAL2 pn3 = {0.0, 0.0};
    REAL2 pn4 = {0.0, 0.0};
    REAL2 pn5 = {0.0, 0.0};
    REAL2 pn6 = {0.0, 0.0};
    REAL2 pn7 = {0.0, 0.0};
    if (pn_order > 0) {
        /* XXX: not implemented. */
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
                /* XXX: not implemented. */
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
                            /* XXX: not implemented. */
                            if (pn_order > 6) {
                                /* XXX: not implemented. */
                            }
                        }
                    }
                }
            }
        }
    }

    /* Form the 1 + 213 terms post-Newtonian */
    REAL2 pn = {0.0, 0.0};

    /* 1 += (((((((65   ) + 103  ) + 6    ) + 31   ) + 0    ) + 8    ) + 0    ) */
    pn.x += (((((((pn7.x) + pn6.x) + pn5.x) + pn4.x) + pn3.x) + pn2.x) + pn1.x);
    pn.y += (((((((pn7.y) + pn6.y) + pn5.y) + pn4.y) + pn3.y) + pn2.y) + pn1.y);

    REAL gm_r2 = mj * r2inv;
    pn.x *= gm_r2;
    pn.y *= gm_r2;

    return pn;
}


inline REAL4
p2p_pnacc_kernel_core(REAL4 pnacc, REAL4 bip, REAL4 biv, REAL4 bjp, REAL4 bjv,
                      CLIGHT clight, int pn_order)
{
    REAL3 dr;               /* XXX: REAL4 -> REAL3 */
    dr.x = bip.x - bjp.x;                                            // 1 FLOPs
    dr.y = bip.y - bjp.y;                                            // 1 FLOPs
    dr.z = bip.z - bjp.z;                                            // 1 FLOPs
//    dr.w = bip.w + bjp.w;   /* XXX: this line will be discarded */   // 1 FLOPs
    REAL dr2 = dr.z * dr.z + (dr.y * dr.y + dr.x * dr.x);            // 5 FLOPs

    REAL mi = bip.w;        /* XXX: biv -> bip */
    REAL mj = bjp.w;        /* XXX: bjv -> bjp */

    REAL3 dv;
    dv.x = biv.x - bjv.x;                                            // 1 FLOPs
    dv.y = biv.y - bjv.y;                                            // 1 FLOPs
    dv.z = biv.z - bjv.z;                                            // 1 FLOPs
    REAL dv2 = dv.z * dv.z + (dv.y * dv.y + dv.x * dv.x);            // 5 FLOPs

    REAL vi2 = biv.w;
    REAL vj2 = bjv.w;

    REAL3 vi = {biv.x, biv.y, biv.z};
    REAL3 vj = {bjv.x, bjv.y, bjv.z};

//    REAL rinv = rsqrt(dr2 + dr.w);                                   // 3 FLOPs
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
                           clight,
                           pn_order);                                // 3 FLOPs

    pnacc.x += pn.x * n.x + pn.y * dv.x;                             // 4 FLOPs
    pnacc.y += pn.x * n.y + pn.y * dv.y;                             // 4 FLOPs
    pnacc.z += pn.x * n.z + pn.y * dv.z;                             // 4 FLOPs

    pnacc.w += (mi + mj) * rinv * r2inv;                             // 4 FLOPs

    return pnacc;
}   // Total flop count: 43


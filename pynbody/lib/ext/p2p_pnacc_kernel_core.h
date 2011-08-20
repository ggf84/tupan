inline REAL2
p2p_pnterms(REAL mi, REAL mj,
            REAL rinv, REAL r2inv,
            REAL3 nij, REAL3 vij, REAL vij2,
            REAL vi2, REAL3 vi,
            REAL vj2, REAL3 vj)
{
    REAL gm_r2 = mj * r2inv;
    REAL2 pn = {-1.0, 0.0};
    pn.x *= gm_r2;
    pn.y *= gm_r2;
    return pn;
}


inline REAL4
p2p_pnacc_kernel_core(REAL4 pnacc, REAL4 bip, REAL4 biv, REAL4 bjp, REAL4 bjv)
{
    REAL4 dr;               /* XXX: REAL4 -> REAL3 */
    dr.x = bip.x - bjp.x;                                            // 1 FLOPs
    dr.y = bip.y - bjp.y;                                            // 1 FLOPs
    dr.z = bip.z - bjp.z;                                            // 1 FLOPs
    dr.w = bip.w + bjp.w;   /* XXX: this line will be discarded */   // 1 FLOPs
    REAL dr2 = dr.z * dr.z + (dr.y * dr.y + dr.x * dr.x);            // 5 FLOPs

    REAL mi = biv.w;        /* XXX: biv -> bip */
    REAL mj = bjv.w;        /* XXX: bjv -> bjp */

    REAL3 dv;
    dv.x = biv.x - bjv.x;                                            // 1 FLOPs
    dv.y = biv.y - bjv.y;                                            // 1 FLOPs
    dv.z = biv.z - bjv.z;                                            // 1 FLOPs
    REAL dv2 = dv.z * dv.z + (dv.y * dv.y + dv.x * dv.x);            // 5 FLOPs

    REAL vi2 = biv.w;
    REAL vj2 = bjv.w;

    REAL3 vi = {biv.x, biv.y, biv.z};
    REAL3 vj = {bjv.x, bjv.y, bjv.z};

    REAL rinv = rsqrt(dr2 + dr.w);                                   // 3 FLOPs
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
                           vj2, vj);                                 // 3 FLOPs

    pnacc.x += pn.x * n.x + pn.y * dv.x;                             // 4 FLOPs
    pnacc.y += pn.x * n.y + pn.y * dv.y;                             // 4 FLOPs
    pnacc.z += pn.x * n.z + pn.y * dv.z;                             // 4 FLOPs

    pnacc.w += (mi + mj) * rinv * r2inv;                             // 4 FLOPs

    return pnacc;
}   // Total flop count: 43


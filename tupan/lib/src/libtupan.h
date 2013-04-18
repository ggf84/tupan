
inline void
phi_kernel(
    const unsigned int ni,
    const REAL *irx, const REAL *iry, const REAL *irz, const REAL *imass,
    const REAL *ivx, const REAL *ivy, const REAL *ivz, const REAL *ieps2,
    const unsigned int nj,
    const REAL *jrx, const REAL *jry, const REAL *jrz, const REAL *jmass,
    const REAL *jvx, const REAL *jvy, const REAL *jvz, const REAL *jeps2,
    REAL *iphi
    );

inline void
acc_kernel(
    const unsigned int ni,
    const REAL *irx, const REAL *iry, const REAL *irz, const REAL *imass,
    const REAL *ivx, const REAL *ivy, const REAL *ivz, const REAL *ieps2,
    const unsigned int nj,
    const REAL *jrx, const REAL *jry, const REAL *jrz, const REAL *jmass,
    const REAL *jvx, const REAL *jvy, const REAL *jvz, const REAL *jeps2,
    REAL *iax, REAL *iay, REAL *iaz
    );

inline void
acc_jerk_kernel(
    const unsigned int ni,
    const REAL *irx, const REAL *iry, const REAL *irz, const REAL *imass,
    const REAL *ivx, const REAL *ivy, const REAL *ivz, const REAL *ieps2,
    const unsigned int nj,
    const REAL *jrx, const REAL *jry, const REAL *jrz, const REAL *jmass,
    const REAL *jvx, const REAL *jvy, const REAL *jvz, const REAL *jeps2,
    REAL *iax, REAL *iay, REAL *iaz,
    REAL *ijx, REAL *ijy, REAL *ijz
    );

inline void
tstep_kernel(
    const unsigned int ni,
    const REAL *irx, const REAL *iry, const REAL *irz, const REAL *imass,
    const REAL *ivx, const REAL *ivy, const REAL *ivz, const REAL *ieps2,
    const unsigned int nj,
    const REAL *jrx, const REAL *jry, const REAL *jrz, const REAL *jmass,
    const REAL *jvx, const REAL *jvy, const REAL *jvz, const REAL *jeps2,
    const REAL eta,
    REAL *idt,
    REAL *ijdtmin
    );

inline void
pnacc_kernel(
    const unsigned int ni,
    const REAL *irx, const REAL *iry, const REAL *irz, const REAL *imass,
    const REAL *ivx, const REAL *ivy, const REAL *ivz, const REAL *ieps2,
    const unsigned int nj,
    const REAL *jrx, const REAL *jry, const REAL *jrz, const REAL *jmass,
    const REAL *jvx, const REAL *jvy, const REAL *jvz, const REAL *jeps2,
    unsigned int order, const REAL inv1,
    const REAL inv2, const REAL inv3,
    const REAL inv4, const REAL inv5,
    const REAL inv6, const REAL inv7,
    REAL *ipnax, REAL *ipnay, REAL *ipnaz
    );

inline void
nreg_Xkernel(
    const unsigned int ni,
    const REAL *irx, const REAL *iry, const REAL *irz, const REAL *imass,
    const REAL *ivx, const REAL *ivy, const REAL *ivz, const REAL *ieps2,
    const unsigned int nj,
    const REAL *jrx, const REAL *jry, const REAL *jrz, const REAL *jmass,
    const REAL *jvx, const REAL *jvy, const REAL *jvz, const REAL *jeps2,
    const REAL dt,
    REAL *new_irx, REAL *new_iry, REAL *new_irz,
    REAL *iax, REAL *iay, REAL *iaz,
    REAL *iu
    );

inline void
nreg_Vkernel(
    const unsigned int ni,
    const REAL *ivx, const REAL *ivy, const REAL *ivz, const REAL *imass,
    const REAL *iax, const REAL *iay, const REAL *iaz,
    const unsigned int nj,
    const REAL *jvx, const REAL *jvy, const REAL *jvz, const REAL *jmass,
    const REAL *jax, const REAL *jay, const REAL *jaz,
    const REAL dt,
    REAL *new_ivx, REAL *new_ivy, REAL *new_ivz,
    REAL *ik
    );

inline void
sakura_kernel(
    const unsigned int ni,
    const REAL *irx, const REAL *iry, const REAL *irz, const REAL *imass,
    const REAL *ivx, const REAL *ivy, const REAL *ivz, const REAL *ieps2,
    const unsigned int nj,
    const REAL *jrx, const REAL *jry, const REAL *jrz, const REAL *jmass,
    const REAL *jvx, const REAL *jvy, const REAL *jvz, const REAL *jeps2,
    const REAL dt,
    REAL *idrx, REAL *idry, REAL *idrz,
    REAL *idvx, REAL *idvy, REAL *idvz
    );


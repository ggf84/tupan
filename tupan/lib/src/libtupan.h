
void phi_kernel(
    const unsigned int ni,
    const REAL *_im,
    const REAL *_irx,
    const REAL *_iry,
    const REAL *_irz,
    const REAL *_ie2,
    const REAL *_ivx,
    const REAL *_ivy,
    const REAL *_ivz,
    const unsigned int nj,
    const REAL *_jm,
    const REAL *_jrx,
    const REAL *_jry,
    const REAL *_jrz,
    const REAL *_je2,
    const REAL *_jvx,
    const REAL *_jvy,
    const REAL *_jvz,
    REAL *_iphi);

void acc_kernel(
    const unsigned int ni,
    const REAL *_im,
    const REAL *_irx,
    const REAL *_iry,
    const REAL *_irz,
    const REAL *_ie2,
    const REAL *_ivx,
    const REAL *_ivy,
    const REAL *_ivz,
    const unsigned int nj,
    const REAL *_jm,
    const REAL *_jrx,
    const REAL *_jry,
    const REAL *_jrz,
    const REAL *_je2,
    const REAL *_jvx,
    const REAL *_jvy,
    const REAL *_jvz,
    REAL *_iax,
    REAL *_iay,
    REAL *_iaz);

void acc_jerk_kernel(
    const unsigned int ni,
    const REAL *_im,
    const REAL *_irx,
    const REAL *_iry,
    const REAL *_irz,
    const REAL *_ie2,
    const REAL *_ivx,
    const REAL *_ivy,
    const REAL *_ivz,
    const unsigned int nj,
    const REAL *_jm,
    const REAL *_jrx,
    const REAL *_jry,
    const REAL *_jrz,
    const REAL *_je2,
    const REAL *_jvx,
    const REAL *_jvy,
    const REAL *_jvz,
    REAL *_iax,
    REAL *_iay,
    REAL *_iaz,
    REAL *_ijx,
    REAL *_ijy,
    REAL *_ijz);

void snap_crackle_kernel(
    const unsigned int ni,
    const REAL *_im,
    const REAL *_irx,
    const REAL *_iry,
    const REAL *_irz,
    const REAL *_ie2,
    const REAL *_ivx,
    const REAL *_ivy,
    const REAL *_ivz,
    const REAL *_iax,
    const REAL *_iay,
    const REAL *_iaz,
    const REAL *_ijx,
    const REAL *_ijy,
    const REAL *_ijz,
    const unsigned int nj,
    const REAL *_jm,
    const REAL *_jrx,
    const REAL *_jry,
    const REAL *_jrz,
    const REAL *_je2,
    const REAL *_jvx,
    const REAL *_jvy,
    const REAL *_jvz,
    const REAL *_jax,
    const REAL *_jay,
    const REAL *_jaz,
    const REAL *_jjx,
    const REAL *_jjy,
    const REAL *_jjz,
    REAL *_isx,
    REAL *_isy,
    REAL *_isz,
    REAL *_icx,
    REAL *_icy,
    REAL *_icz);

void tstep_kernel(
    const unsigned int ni,
    const REAL *_im,
    const REAL *_irx,
    const REAL *_iry,
    const REAL *_irz,
    const REAL *_ie2,
    const REAL *_ivx,
    const REAL *_ivy,
    const REAL *_ivz,
    const unsigned int nj,
    const REAL *_jm,
    const REAL *_jrx,
    const REAL *_jry,
    const REAL *_jrz,
    const REAL *_je2,
    const REAL *_jvx,
    const REAL *_jvy,
    const REAL *_jvz,
    const REAL eta,
    REAL *_idt_a,
    REAL *_idt_b);

void pnacc_kernel(
    const unsigned int ni,
    const REAL *_im,
    const REAL *_irx,
    const REAL *_iry,
    const REAL *_irz,
    const REAL *_ie2,
    const REAL *_ivx,
    const REAL *_ivy,
    const REAL *_ivz,
    const unsigned int nj,
    const REAL *_jm,
    const REAL *_jrx,
    const REAL *_jry,
    const REAL *_jrz,
    const REAL *_je2,
    const REAL *_jvx,
    const REAL *_jvy,
    const REAL *_jvz,
    unsigned int order,
    const REAL inv1,
    const REAL inv2,
    const REAL inv3,
    const REAL inv4,
    const REAL inv5,
    const REAL inv6,
    const REAL inv7,
    REAL *_ipnax,
    REAL *_ipnay,
    REAL *_ipnaz);

void nreg_Xkernel(
    const unsigned int ni,
    const REAL *_im,
    const REAL *_irx,
    const REAL *_iry,
    const REAL *_irz,
    const REAL *_ie2,
    const REAL *_ivx,
    const REAL *_ivy,
    const REAL *_ivz,
    const unsigned int nj,
    const REAL *_jm,
    const REAL *_jrx,
    const REAL *_jry,
    const REAL *_jrz,
    const REAL *_je2,
    const REAL *_jvx,
    const REAL *_jvy,
    const REAL *_jvz,
    const REAL dt,
    REAL *_idrx,
    REAL *_idry,
    REAL *_idrz,
    REAL *_iax,
    REAL *_iay,
    REAL *_iaz,
    REAL *_iu);

void nreg_Vkernel(
    const unsigned int ni,
    const REAL *_im,
    const REAL *_ivx,
    const REAL *_ivy,
    const REAL *_ivz,
    const REAL *_iax,
    const REAL *_iay,
    const REAL *_iaz,
    const unsigned int nj,
    const REAL *_jm,
    const REAL *_jvx,
    const REAL *_jvy,
    const REAL *_jvz,
    const REAL *_jax,
    const REAL *_jay,
    const REAL *_jaz,
    const REAL dt,
    REAL *_idvx,
    REAL *_idvy,
    REAL *_idvz,
    REAL *_ik);

void sakura_kernel(
    const unsigned int ni,
    const REAL *_im,
    const REAL *_irx,
    const REAL *_iry,
    const REAL *_irz,
    const REAL *_ie2,
    const REAL *_ivx,
    const REAL *_ivy,
    const REAL *_ivz,
    const unsigned int nj,
    const REAL *_jm,
    const REAL *_jrx,
    const REAL *_jry,
    const REAL *_jrz,
    const REAL *_je2,
    const REAL *_jvx,
    const REAL *_jvy,
    const REAL *_jvz,
    const REAL dt,
    REAL *_idrx,
    REAL *_idry,
    REAL *_idrz,
    REAL *_idvx,
    REAL *_idvy,
    REAL *_idvz);


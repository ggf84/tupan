void
phi_kernel(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	real_t __iphi[]);

void
phi_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	real_t __iphi[],
	real_t __jphi[]);

void
phi_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	real_t __iphi[]);

void
acc_kernel(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	real_t __iadot[]);

void
acc_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	real_t __iadot[],
	real_t __jadot[]);

void
acc_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	real_t __iadot[]);

void
acc_jrk_kernel(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	real_t __iadot[]);

void
acc_jrk_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	real_t __iadot[],
	real_t __jadot[]);

void
acc_jrk_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	real_t __iadot[]);

void
snp_crk_kernel(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	real_t __iadot[]);

void
snp_crk_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	real_t __iadot[],
	real_t __jadot[]);

void
snp_crk_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	real_t __iadot[]);

void
tstep_kernel(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	const real_t eta,
	real_t __idt_a[],
	real_t __idt_b[]);

void
tstep_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	const real_t eta,
	real_t __idt_a[],
	real_t __idt_b[],
	real_t __jdt_a[],
	real_t __jdt_b[]);

void
tstep_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const real_t eta,
	real_t __idt_a[],
	real_t __idt_b[]);

void
pnacc_kernel(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	const CLIGHT clight,
	real_t __ipnacc[]);

void
pnacc_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	const CLIGHT clight,
	real_t __ipnacc[],
	real_t __jpnacc[]);

void
pnacc_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const CLIGHT clight,
	real_t __ipnacc[]);

void
nreg_Xkernel(
	const uint_t ni,
	const real_t __im[],
	const real_t __irx[],
	const real_t __iry[],
	const real_t __irz[],
	const real_t __ie2[],
	const real_t __ivx[],
	const real_t __ivy[],
	const real_t __ivz[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __jrx[],
	const real_t __jry[],
	const real_t __jrz[],
	const real_t __je2[],
	const real_t __jvx[],
	const real_t __jvy[],
	const real_t __jvz[],
	const real_t dt,
	real_t __idrx[],
	real_t __idry[],
	real_t __idrz[],
	real_t __iax[],
	real_t __iay[],
	real_t __iaz[],
	real_t __iu[]);

void
nreg_Xkernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __irx[],
	const real_t __iry[],
	const real_t __irz[],
	const real_t __ie2[],
	const real_t __ivx[],
	const real_t __ivy[],
	const real_t __ivz[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __jrx[],
	const real_t __jry[],
	const real_t __jrz[],
	const real_t __je2[],
	const real_t __jvx[],
	const real_t __jvy[],
	const real_t __jvz[],
	const real_t dt,
	real_t __idrx[],
	real_t __idry[],
	real_t __idrz[],
	real_t __iax[],
	real_t __iay[],
	real_t __iaz[],
	real_t __iu[],
	real_t __jdrx[],
	real_t __jdry[],
	real_t __jdrz[],
	real_t __jax[],
	real_t __jay[],
	real_t __jaz[],
	real_t __ju[]);

void
nreg_Xkernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __irx[],
	const real_t __iry[],
	const real_t __irz[],
	const real_t __ie2[],
	const real_t __ivx[],
	const real_t __ivy[],
	const real_t __ivz[],
	const real_t dt,
	real_t __idrx[],
	real_t __idry[],
	real_t __idrz[],
	real_t __iax[],
	real_t __iay[],
	real_t __iaz[],
	real_t __iu[]);

void
nreg_Vkernel(
	const uint_t ni,
	const real_t __im[],
	const real_t __ivx[],
	const real_t __ivy[],
	const real_t __ivz[],
	const real_t __iax[],
	const real_t __iay[],
	const real_t __iaz[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __jvx[],
	const real_t __jvy[],
	const real_t __jvz[],
	const real_t __jax[],
	const real_t __jay[],
	const real_t __jaz[],
	const real_t dt,
	real_t __idvx[],
	real_t __idvy[],
	real_t __idvz[],
	real_t __ik[]);

void
nreg_Vkernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ivx[],
	const real_t __ivy[],
	const real_t __ivz[],
	const real_t __iax[],
	const real_t __iay[],
	const real_t __iaz[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __jvx[],
	const real_t __jvy[],
	const real_t __jvz[],
	const real_t __jax[],
	const real_t __jay[],
	const real_t __jaz[],
	const real_t dt,
	real_t __idvx[],
	real_t __idvy[],
	real_t __idvz[],
	real_t __ik[],
	real_t __jdvx[],
	real_t __jdvy[],
	real_t __jdvz[],
	real_t __jk[]);

void
nreg_Vkernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ivx[],
	const real_t __ivy[],
	const real_t __ivz[],
	const real_t __iax[],
	const real_t __iay[],
	const real_t __iaz[],
	const real_t dt,
	real_t __idvx[],
	real_t __idvy[],
	real_t __idvz[],
	real_t __ik[]);

void
sakura_kernel(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	const real_t dt,
	const int_t flag,
	real_t __idrdot[]);

void
sakura_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	const real_t dt,
	const int_t flag,
	real_t __idrdot[],
	real_t __jdrdot[]);

void
sakura_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const real_t dt,
	const int_t flag,
	real_t __idrdot[]);

void
kepler_solver_kernel(
	const uint_t ni,
	const real_t __im[],
	const real_t __irx[],
	const real_t __iry[],
	const real_t __irz[],
	const real_t __ie2[],
	const real_t __ivx[],
	const real_t __ivy[],
	const real_t __ivz[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __jrx[],
	const real_t __jry[],
	const real_t __jrz[],
	const real_t __je2[],
	const real_t __jvx[],
	const real_t __jvy[],
	const real_t __jvz[],
	const real_t dt,
	real_t __ir1x[],
	real_t __ir1y[],
	real_t __ir1z[],
	real_t __iv1x[],
	real_t __iv1y[],
	real_t __iv1z[],
	real_t __jr1x[],
	real_t __jr1y[],
	real_t __jr1z[],
	real_t __jv1x[],
	real_t __jv1y[],
	real_t __jv1z[]);


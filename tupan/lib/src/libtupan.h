void
phi_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __ipos[],
	real_t __iphi[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jpos[],
	real_t __jphi[]);

void
phi_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __ipos[],
	real_t __iphi[]);

void
acc_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __ipos[],
	real_t __iacc[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jpos[],
	real_t __jacc[]);

void
acc_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __ipos[],
	real_t __iacc[]);

void
acc_jrk_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __ipos[],
	const real_t __ivel[],
	real_t __iacc[],
	real_t __ijrk[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jpos[],
	const real_t __jvel[],
	real_t __jacc[],
	real_t __jjrk[]);

void
acc_jrk_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __ipos[],
	const real_t __ivel[],
	real_t __iacc[],
	real_t __ijrk[]);

void
snp_crk_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __ipos[],
	const real_t __ivel[],
	const real_t __iacc[],
	const real_t __ijrk[],
	real_t __if0[],
	real_t __if1[],
	real_t __if2[],
	real_t __if3[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jpos[],
	const real_t __jvel[],
	const real_t __jacc[],
	const real_t __jjrk[],
	real_t __jf0[],
	real_t __jf1[],
	real_t __jf2[],
	real_t __jf3[]);

void
snp_crk_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __ipos[],
	const real_t __ivel[],
	const real_t __iacc[],
	const real_t __ijrk[],
	real_t __if0[],
	real_t __if1[],
	real_t __if2[],
	real_t __if3[]);

void
tstep_kernel_rectangle(
	const real_t eta,
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __ipos[],
	const real_t __ivel[],
	real_t __iw2_a[],
	real_t __iw2_b[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jpos[],
	const real_t __jvel[],
	real_t __jw2_a[],
	real_t __jw2_b[]);

void
tstep_kernel_triangle(
	const real_t eta,
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __ipos[],
	const real_t __ivel[],
	real_t __iw2_a[],
	real_t __iw2_b[]);

void
pnacc_kernel_rectangle(
	const uint_t order,
	const real_t clight,
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __ipos[],
	const real_t __ivel[],
	real_t __ipnacc[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jpos[],
	const real_t __jvel[],
	real_t __jpnacc[]);

void
pnacc_kernel_triangle(
	const uint_t order,
	const real_t clight,
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __ipos[],
	const real_t __ivel[],
	real_t __ipnacc[]);

void
sakura_kernel_rectangle(
	const real_t dt,
	const int_t flag,
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __ipos[],
	const real_t __ivel[],
	real_t __idpos[],
	real_t __idvel[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jpos[],
	const real_t __jvel[],
	real_t __jdpos[],
	real_t __jdvel[]);

void
sakura_kernel_triangle(
	const real_t dt,
	const int_t flag,
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __ipos[],
	const real_t __ivel[],
	real_t __idpos[],
	real_t __idvel[]);

void
kepler_solver_kernel(
	const real_t dt,
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	real_t __irx[],
	real_t __iry[],
	real_t __irz[],
	real_t __ivx[],
	real_t __ivy[],
	real_t __ivz[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	real_t __jrx[],
	real_t __jry[],
	real_t __jrz[],
	real_t __jvx[],
	real_t __jvy[],
	real_t __jvz[]);


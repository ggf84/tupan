void
phi_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	real_t __iphi[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	real_t __jphi[]);

void
phi_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	real_t __iphi[]);

void
acc_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	real_t __iadot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	real_t __jadot[]);

void
acc_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	real_t __iadot[]);

void
acc_jrk_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	real_t __iadot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	real_t __jadot[]);

void
acc_jrk_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	real_t __iadot[]);

void
snp_crk_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	real_t __iadot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	real_t __jadot[]);

void
snp_crk_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	real_t __iadot[]);

void
tstep_kernel_rectangle(
	const real_t eta,
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	real_t __iw2_a[],
	real_t __iw2_b[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	real_t __jw2_a[],
	real_t __jw2_b[]);

void
tstep_kernel_triangle(
	const real_t eta,
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	real_t __iw2_a[],
	real_t __iw2_b[]);

void
pnacc_kernel_rectangle(
	const uint_t order,
	const real_t clight,
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	real_t __ipnacc[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	real_t __jpnacc[]);

void
pnacc_kernel_triangle(
	const uint_t order,
	const real_t clight,
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	real_t __ipnacc[]);

void
sakura_kernel_rectangle(
	const real_t dt,
	const int_t flag,
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	real_t __idrdot[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __je2[],
	const real_t __jrdot[],
	real_t __jdrdot[]);

void
sakura_kernel_triangle(
	const real_t dt,
	const int_t flag,
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	real_t __idrdot[]);

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


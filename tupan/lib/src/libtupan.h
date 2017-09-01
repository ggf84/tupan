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
	real_t __iw2_a[],
	real_t __iw2_b[],
	real_t __jw2_a[],
	real_t __jw2_b[]);

void
tstep_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const real_t eta,
	real_t __iw2_a[],
	real_t __iw2_b[]);

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
	const uint_t order,
	const real_t clight,
	real_t __ipnacc[],
	real_t __jpnacc[]);

void
pnacc_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __ie2[],
	const real_t __irdot[],
	const uint_t order,
	const real_t clight,
	real_t __ipnacc[]);

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


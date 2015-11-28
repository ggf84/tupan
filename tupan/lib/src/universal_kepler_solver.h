#ifndef __UNIVERSAL_KEPLER_SOLVER_H__
#define __UNIVERSAL_KEPLER_SOLVER_H__

#include "common.h"


static constant real_t f3[] =	/* f3_i = (2*i+3)! / (2*i+1)! */
{1./6, 1./20, 1./42, 1./72, 1./110, 1./156, 1./210, 1./272, 1./342, 1./420};

static constant real_t f2[] =	/* f2_i = (2*i+2)! / (2*i+0)! */
{1./2, 1./12, 1./30, 1./56, 1./90 , 1./132, 1./182, 1./240, 1./306, 1./380};


static inline void
stumpff_cs(
	real_t z,
	real_t cs[4])
{
	uint_t n = 0;
	while (fabs(z) > 1) {
		z *= (real_t)(0.25);
		++n;
	}
	cs[3] = (((((((((z * f3[9] + 1) *
					 z * f3[8] + 1) *
					 z * f3[7] + 1) *
					 z * f3[6] + 1) *
					 z * f3[5] + 1) *
					 z * f3[4] + 1) *
					 z * f3[3] + 1) *
					 z * f3[2] + 1) *
					 z * f3[1] + 1) * f3[0];
	cs[2] = (((((((((z * f2[9] + 1) *
					 z * f2[8] + 1) *
					 z * f2[7] + 1) *
					 z * f2[6] + 1) *
					 z * f2[5] + 1) *
					 z * f2[4] + 1) *
					 z * f2[3] + 1) *
					 z * f2[2] + 1) *
					 z * f2[1] + 1) * f2[0];
	cs[1] = 1 + z * cs[3];
	for (; n > 0; --n) {
		z *= 4;
		cs[3] = (cs[3] + cs[1] * cs[2]) * (real_t)(0.25);
		cs[2] = (cs[1] * cs[1]) * (real_t)(0.5);
		cs[1] = 1 + z * cs[3];
	}
	cs[0] = 1 + z * cs[2];
}


static inline void
stiefel_Gs(
	const real_t s,
	const real_t alpha0,
	real_t Gs[4])
{
	real_t s2 = s * s;
	stumpff_cs(alpha0*s2, Gs);
	Gs[1] *= s;
	Gs[2] *= s2;
	Gs[3] *= s2 * s;
}


static inline real_t
kepler_f(
	const real_t s,
	const real_t r0,
	const real_t sigma0,
	const real_t gamma0,
	const real_t Gs[4])
{
	return r0 * s + sigma0 * Gs[2] + gamma0 * Gs[3];
}


static inline real_t
kepler_fp(
	const real_t r0,
	const real_t sigma0,
	const real_t gamma0,
	const real_t Gs[4])
{
	return r0 + sigma0 * Gs[1] + gamma0 * Gs[2];
}


static inline real_t
kepler_fpp(
	const real_t sigma0,
	const real_t gamma0,
	const real_t Gs[4])
{
	return sigma0 * Gs[0] + gamma0 * Gs[1];
}


static inline void
update_pos_vel(
	const real_t m,
	const real_t r0,
	const real_t inv_r0,
	const real_t inv_r1,
	const real_t sigma0,
	const real_t Gs[4],
	real_t *rx,
	real_t *ry,
	real_t *rz,
	real_t *vx,
	real_t *vy,
	real_t *vz)
{
	// These are not the traditional Gauss f and g functions.
	real_t f = -m * Gs[2] * inv_r0;
	real_t g = r0 * Gs[1] + sigma0 * Gs[2];
	real_t df = -m * Gs[1] * inv_r0 * inv_r1;
	real_t dg = -m * Gs[2] * inv_r1;

	real_t r0x = *rx, r0y = *ry, r0z = *rz;
	real_t v0x = *vx, v0y = *vy, v0z = *vz;

	*rx += (r0x * f + v0x * g);
	*ry += (r0y * f + v0y * g);
	*rz += (r0z * f + v0z * g);
	*vx += (r0x * df + v0x * dg);
	*vy += (r0y * df + v0y * dg);
	*vz += (r0z * df + v0z * dg);
}


static inline real_t
initial_guess(
	const real_t m,
	const real_t r0,
	const real_t dt0,
	const real_t alpha0,
	real_t *S)
{
	real_t s = 0;	// Initial guess for hyperbolic / nearly parabolical orbits.
	real_t dt = dt0;

	if (alpha0 < 0) {
		/*
		** Elliptical orbits: reduce the time-step
		** to a fraction of the orbital period.
		*/
		real_t sqrt_alpha = sqrt(-alpha0);
		real_t T = TWOPI * m / (-alpha0 * sqrt_alpha);
		real_t ratio = fabs(dt0) / T;
		real_t rdiff = ratio - floor(ratio);
		dt = copysign(rdiff * T, dt0);

		/*
		** Initial guess for elliptical / nearly parabolical orbits.
		*/
		real_t inv_r = 2 * alpha0 / (alpha0 * r0 - m);	// == 2/(r0 + a)
		s = dt * inv_r;
	}

	*S = s;
	return dt;
}


static inline int_t
rootfinder(
	const real_t r0,
	const real_t dt0,
	const real_t alpha0,
	const real_t sigma0,
	const real_t gamma0,
	real_t Gs[4],
	real_t *S)
{
	int_t err = -1;
	real_t olds[MAXITER+1] = {NAN};

	real_t s = *S;

	for (uint_t iter = 0; iter < MAXITER; ++iter) {
		stiefel_Gs(s, alpha0, Gs);
		real_t f = kepler_f(s, r0, sigma0, gamma0, Gs) - dt0;
		real_t fp = kepler_fp(r0, sigma0, gamma0, Gs);
		real_t fpp = kepler_fpp(sigma0, gamma0, Gs);

		real_t g = f;
		real_t h = fp;

		#define ORDER 5
		#define ORDER1 (ORDER - 1)
		g *= ORDER;
		real_t b = ORDER1 * h * h - g * fpp;
		h += copysign(sqrt(fabs(ORDER1 * b)), h);

		s = (s * h - g) / h;
		for (uint_t i = 0; i < iter; ++i) {
			if (s == olds[i]) {
				err = 0;
				iter = MAXITER;
				i = iter;
			}
		}
		olds[iter] = s;
	}

	*S = s;
	return err;
}


static inline int_t
__universal_kepler_solver(
	const real_t dt0,
	const real_t m,
	const real_t e2,
	real_t *rx,
	real_t *ry,
	real_t *rz,
	real_t *vx,
	real_t *vy,
	real_t *vz)
{
	real_t r2 = *rx * *rx + *ry * *ry + *rz * *rz;
	if (!(r2 > 0)) return 0;

	r2 += e2;
	real_t inv_r0 = rsqrt(r2);
	real_t r0 = r2 * inv_r0;

	real_t phi0 = 2 * m * inv_r0;
	real_t v0sqr = *vx * *vx + *vy * *vy + *vz * *vz;
	real_t sigma0 = *rx * *vx + *ry * *vy + *rz * *vz;
	real_t alpha0 = v0sqr - phi0;
	real_t gamma0 = alpha0 * r0 + m;
	real_t lagr0 = v0sqr + phi0;
	real_t abs_alpha0 = fabs(alpha0);

	#ifndef CONFIG_USE_OPENCL
	if ((abs_alpha0 < TOLERANCE * lagr0)
			&& (r0 * abs_alpha0 < TOLERANCE * m)) {
		fprintf(stderr, "#---WARNING: Please use higher "
						"floating point precision.\n");
		fprintf(stderr, "#---err flag: \n");
		fprintf(stderr,
			"#   dt0: %a, m: %a, e2: %a,"
			" rx: %a, ry: %a, rz: %a,"
			" vx: %a, vy: %a, vz: %a\n"
			"#   r0: %a, sigma0: %a, v0sqr: %a,"
			" phi0: %a, alpha0: %a, lagr0: %a\n",
			dt0, m, e2,
			*rx, *ry, *rz,
			*vx, *vy, *vz,
			r0, sigma0, v0sqr,
			phi0, alpha0, lagr0);
		fprintf(stderr, "#---\n");
	}
	#endif

	real_t s, Gs[4] = {NAN};
	real_t dt = initial_guess(m, r0, dt0, alpha0, &s);
	int_t err = rootfinder(r0, dt, alpha0, sigma0, gamma0, Gs, &s);

	real_t alpha = alpha0;
	real_t gamma = gamma0;
	if (e2 > 0) {
		real_t inv_r = s / dt;	// time average of 1/r over dt.
		alpha += 2 * m * e2 * inv_r * inv_r * inv_r;
		gamma = alpha * r0 + m;
		err |= rootfinder(r0, dt, alpha, sigma0, gamma, Gs, &s);
	}
	real_t r1 = kepler_fp(r0, sigma0, gamma, Gs);

	update_pos_vel(
		m, r0, inv_r0, 1/r1, sigma0, Gs,
		&(*rx), &(*ry), &(*rz),
		&(*vx), &(*vy), &(*vz));

	#ifndef CONFIG_USE_OPENCL
	if (err != 0) {
		fprintf(stderr, "#---WARNING: Maximum iteration steps "
						"reached in 'rootfinder' function. Trying "
						"again with two steps of size dt0/2.\n");
		fprintf(stderr, "#---err flag: %ld\n", (long)(err));
		fprintf(stderr,
			"#   dt0: %a, m: %a, e2: %a, r0: %a, alpha0: %a,"
			" sigma0: %a, gamma0: %a, v0sqr: %a, phi0: %a,"
			" rx: %a, ry: %a, rz: %a, vx: %a, vy: %a, vz: %a\n",
			dt0, m, e2,	r0, alpha0,
			sigma0, gamma0, v0sqr, phi0,
			*rx, *ry, *rz, *vx, *vy, *vz);
		fprintf(stderr, "#---\n");
	}
	#endif

	if (e2 > 0) {
		real_t r1sqr = *rx * *rx + *ry * *ry + *rz * *rz;
		real_t r1a = sqrt(r1sqr + e2);
		if (fabs(r1a - r1) > exp2(-(real_t)(3)*sizeof(real_t)) * (r1a + r1)) {
			err = -11;
		}
	}

	return err;
}


static inline int_t
_universal_kepler_solver(
	const real_t dt,
	const real_t m,
	const real_t e2,
	real_t *rx,
	real_t *ry,
	real_t *rz,
	real_t *vx,
	real_t *vy,
	real_t *vz)
{
	int_t err;
	uint_t n = 1;

	real_t r0x = *rx, r0y = *ry, r0z = *rz;
	real_t v0x = *vx, v0y = *vy, v0z = *vz;

	for (uint_t iter = 0; iter < MAXITER; ++iter) {
		err = 0;
		*rx = r0x;
		*ry = r0y;
		*rz = r0z;
		*vx = v0x;
		*vy = v0y;
		*vz = v0z;
		for (uint_t i = 0; i < n; ++i) {
			err |= __universal_kepler_solver(
						dt/n, m, e2,
						&(*rx), &(*ry), &(*rz),
						&(*vx), &(*vy), &(*vz));
			if (err != 0) {
				n *= 2;
				i = n;
			}
		}
		if (err == 0) iter = MAXITER;
	}

	return err;
}


static inline int_t
universal_kepler_solver(
	const real_t dt,
	const real_t m,
	const real_t e2,
	const real_t r0x,
	const real_t r0y,
	const real_t r0z,
	const real_t v0x,
	const real_t v0y,
	const real_t v0z,
	real_t *r1x,
	real_t *r1y,
	real_t *r1z,
	real_t *v1x,
	real_t *v1y,
	real_t *v1z)
{
	*r1x = r0x, *r1y = r0y, *r1z = r0z;
	*v1x = v0x, *v1y = v0y, *v1z = v0z;

	int_t err = _universal_kepler_solver(
					dt, m, e2,
					&(*r1x), &(*r1y), &(*r1z),
					&(*v1x), &(*v1y), &(*v1z));

	#ifndef CONFIG_USE_OPENCL
	if (err != 0) {
		fprintf(stderr, "#---ERROR: The solution "
						"have not converged.\n");
		fprintf(stderr, "#---err flag: %ld\n", (long)(err));
		fprintf(stderr,
			"#   dt: %a, m: %a, e2: %a,"
			" r0x: %a, r0y: %a, r0z: %a,"
			" v0x: %a, v0y: %a, v0z: %a\n",
			dt, m, e2,
			r0x, r0y, r0z,
			v0x, v0y, v0z);
		fprintf(stderr, "#---\n");
	}
	#endif

	return err;
}


#endif	// __UNIVERSAL_KEPLER_SOLVER_H__

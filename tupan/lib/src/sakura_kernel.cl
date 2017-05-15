#include "sakura_kernel_common.h"


static inline void
p2p_sakura_kernel_core(
	const real_t dt,
	const int_t flag,
	uint_t lid,
	concat(Sakura_Data, WPT) *jp,
	local concat(Sakura_Data, WGSIZE) *ip)
// flop count: 41 + ??
{
	#pragma unroll
	for (uint_t w = 0; w < WGSIZE; w += NLANES) {
		for (uint_t l = w; l < w + NLANES; ++l) {
			uint_t i = lid^l;
			for (uint_t j = 0; j < WPT; ++j) {
				for (uint_t k = 0; k < 1; ++k) {
					real_t m = ip->m[i] + jp->m[j];
					real_t e2 = ip->e2[i] + jp->e2[j];
					real_t r0x = ip->rx[i] - jp->rx[j];
					real_t r0y = ip->ry[i] - jp->ry[j];
					real_t r0z = ip->rz[i] - jp->rz[j];
					real_t v0x = ip->vx[i] - jp->vx[j];
					real_t v0y = ip->vy[i] - jp->vy[j];
					real_t v0z = ip->vz[i] - jp->vz[j];

					real_t r1x = r0x;
					real_t r1y = r0y;
					real_t r1z = r0z;
					real_t v1x = v0x;
					real_t v1y = v0y;
					real_t v1z = v0z;
					evolve_twobody(
						dt, flag, m, e2,
						r0x, r0y, r0z, v0x, v0y, v0z,
						&r1x, &r1y, &r1z, &v1x, &v1y, &v1z
					);	// flop count: ??

					real_t inv_m = 1 / m;
					real_t drx = r0x - r1x;
					real_t dry = r0y - r1y;
					real_t drz = r0z - r1z;
					real_t dvx = v0x - v1x;
					real_t dvy = v0y - v1y;
					real_t dvz = v0z - v1z;

					real_t imu = ip->m[i] * inv_m;
					jp->drx[j] += imu * drx;
					jp->dry[j] += imu * dry;
					jp->drz[j] += imu * drz;
					jp->dvx[j] += imu * dvx;
					jp->dvy[j] += imu * dvy;
					jp->dvz[j] += imu * dvz;

					real_t jmu = jp->m[j] * inv_m;
					ip->drx[i] += jmu * drx;
					ip->dry[i] += jmu * dry;
					ip->drz[i] += jmu * drz;
					ip->dvx[i] += jmu * dvx;
					ip->dvy[i] += jmu * dvy;
					ip->dvz[i] += jmu * dvz;

					simd_shuff_Sakura_Data(j, jp);
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}


static inline void
sakura_kernel_core(
	const real_t dt,
	const int_t flag,
	uint_t lid,
	concat(Sakura_Data, WPT) *jp,
	local concat(Sakura_Data, WGSIZE) *ip)
// flop count: 28 + ??
{
	#pragma unroll
	for (uint_t w = 0; w < WGSIZE; w += NLANES) {
		for (uint_t l = w; l < w + NLANES; ++l) {
			uint_t i = lid^l;
			for (uint_t j = 0; j < WPT; ++j) {
				for (uint_t k = 0; k < 1; ++k) {
					real_t m = ip->m[i] + jp->m[j];
					real_t e2 = ip->e2[i] + jp->e2[j];
					real_t r0x = ip->rx[i] - jp->rx[j];
					real_t r0y = ip->ry[i] - jp->ry[j];
					real_t r0z = ip->rz[i] - jp->rz[j];
					real_t v0x = ip->vx[i] - jp->vx[j];
					real_t v0y = ip->vy[i] - jp->vy[j];
					real_t v0z = ip->vz[i] - jp->vz[j];

					real_t r1x = r0x;
					real_t r1y = r0y;
					real_t r1z = r0z;
					real_t v1x = v0x;
					real_t v1y = v0y;
					real_t v1z = v0z;
					evolve_twobody(
						dt, flag, m, e2,
						r0x, r0y, r0z, v0x, v0y, v0z,
						&r1x, &r1y, &r1z, &v1x, &v1y, &v1z
					);	// flop count: ??

					real_t inv_m = 1 / m;
					real_t drx = r0x - r1x;
					real_t dry = r0y - r1y;
					real_t drz = r0z - r1z;
					real_t dvx = v0x - v1x;
					real_t dvy = v0y - v1y;
					real_t dvz = v0z - v1z;

					real_t imu = ip->m[i] * inv_m;
					jp->drx[j] += imu * drx;
					jp->dry[j] += imu * dry;
					jp->drz[j] += imu * drz;
					jp->dvx[j] += imu * dvx;
					jp->dvy[j] += imu * dvy;
					jp->dvz[j] += imu * dvz;

//					real_t jmu = jp->m[j] * inv_m;
//					ip->drx[i] += jmu * drx;
//					ip->dry[i] += jmu * dry;
//					ip->drz[i] += jmu * drz;
//					ip->dvx[i] += jmu * dvx;
//					ip->dvy[i] += jmu * dvy;
//					ip->dvz[i] += jmu * dvz;

					simd_shuff_Sakura_Data(j, jp);
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}


kernel void
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
sakura_kernel_rectangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __je2[],
	global const real_t __jrdot[],
	const real_t dt,
	const int_t flag,
	global real_t __idrdot[],
	global real_t __jdrdot[])
{
	local concat(Sakura_Data, WGSIZE) _ip;
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t block = WGSIZE * 1 * WPT;
	for (uint_t jj = block * grp;
				jj < nj;
				jj += block * ngrps) {
		concat(Sakura_Data, WPT) jp = {{{0}}};
		concat(load_Sakura_Data, WPT)(
			&jp, jj + lid, WGSIZE, 1,
			nj, __jm, __je2, __jrdot
		);

		for (uint_t ii = 0;
					ii < ni;
					ii += block)
			if ((ii == jj)
				|| (ii > jj/* && ((ii + jj) / block) % 2 == 0*/)
				|| (ii < jj/* && ((ii + jj) / block) % 2 == 1*/)) {
			for (uint_t ilid = lid;
						ilid < block;
						ilid += WGSIZE * 1) {
				concat(Sakura_Data, 1) ip = {{{0}}};
				concat(load_Sakura_Data, 1)(
					&ip, ii + ilid, WGSIZE, 1,
					ni, __im, __ie2, __irdot
				);
				_ip.m[lid] = ip.m[0];
				_ip.e2[lid] = ip.e2[0];
				_ip.rx[lid] = ip.rx[0];
				_ip.ry[lid] = ip.ry[0];
				_ip.rz[lid] = ip.rz[0];
				_ip.vx[lid] = ip.vx[0];
				_ip.vy[lid] = ip.vy[0];
				_ip.vz[lid] = ip.vz[0];
				_ip.drx[lid] = ip.drx[0];
				_ip.dry[lid] = ip.dry[0];
				_ip.drz[lid] = ip.drz[0];
				_ip.dvx[lid] = ip.dvx[0];
				_ip.dvy[lid] = ip.dvy[0];
				_ip.dvz[lid] = ip.dvz[0];

				if (ii != jj) {
					p2p_sakura_kernel_core(dt, flag, lid, &jp, &_ip);
				} else {
					p2p_sakura_kernel_core(dt, flag, lid, &jp, &_ip);
				}

				ip.drx[0] = -_ip.drx[lid];
				ip.dry[0] = -_ip.dry[lid];
				ip.drz[0] = -_ip.drz[lid];
				ip.dvx[0] = -_ip.dvx[lid];
				ip.dvy[0] = -_ip.dvy[lid];
				ip.dvz[0] = -_ip.dvz[lid];
				concat(store_Sakura_Data, 1)(
					&ip, ii + ilid, WGSIZE, 1,
					ni, __idrdot
				);
			}
		}

		concat(store_Sakura_Data, WPT)(
			&jp, jj + lid, WGSIZE, 1,
			nj, __jdrdot
		);
	}
}


kernel void
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
sakura_kernel_triangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const real_t dt,
	const int_t flag,
	global real_t __idrdot[])
{
	// ------------------------------------------------------------------------
	const uint_t nj = ni;
	global const real_t *__jm = __im;
	global const real_t *__je2 = __ie2;
	global const real_t *__jrdot = __irdot;
	global real_t *__jdrdot = __idrdot;
	// ------------------------------------------------------------------------

	local concat(Sakura_Data, WGSIZE) _ip;
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t block = WGSIZE * 1 * WPT;
	for (uint_t jj = block * grp;
				jj < nj;
				jj += block * ngrps) {
		concat(Sakura_Data, WPT) jp = {{{0}}};
		concat(load_Sakura_Data, WPT)(
			&jp, jj + lid, WGSIZE, 1,
			nj, __jm, __je2, __jrdot
		);

		for (uint_t ii = 0;
					ii < ni;
					ii += block)
			if ((ii == jj)
				|| (ii > jj && ((ii + jj) / block) % 2 == 0)
				|| (ii < jj && ((ii + jj) / block) % 2 == 1)) {
			for (uint_t ilid = lid;
						ilid < block;
						ilid += WGSIZE * 1) {
				concat(Sakura_Data, 1) ip = {{{0}}};
				concat(load_Sakura_Data, 1)(
					&ip, ii + ilid, WGSIZE, 1,
					ni, __im, __ie2, __irdot
				);
				_ip.m[lid] = ip.m[0];
				_ip.e2[lid] = ip.e2[0];
				_ip.rx[lid] = ip.rx[0];
				_ip.ry[lid] = ip.ry[0];
				_ip.rz[lid] = ip.rz[0];
				_ip.vx[lid] = ip.vx[0];
				_ip.vy[lid] = ip.vy[0];
				_ip.vz[lid] = ip.vz[0];
				_ip.drx[lid] = ip.drx[0];
				_ip.dry[lid] = ip.dry[0];
				_ip.drz[lid] = ip.drz[0];
				_ip.dvx[lid] = ip.dvx[0];
				_ip.dvy[lid] = ip.dvy[0];
				_ip.dvz[lid] = ip.dvz[0];

				if (ii != jj) {
					p2p_sakura_kernel_core(dt, flag, lid, &jp, &_ip);
				} else {
					sakura_kernel_core(dt, flag, lid, &jp, &_ip);
				}

				ip.drx[0] = -_ip.drx[lid];
				ip.dry[0] = -_ip.dry[lid];
				ip.drz[0] = -_ip.drz[lid];
				ip.dvx[0] = -_ip.dvx[lid];
				ip.dvy[0] = -_ip.dvy[lid];
				ip.dvz[0] = -_ip.dvz[lid];
				concat(store_Sakura_Data, 1)(
					&ip, ii + ilid, WGSIZE, 1,
					ni, __idrdot
				);
			}
		}

		concat(store_Sakura_Data, WPT)(
			&jp, jj + lid, WGSIZE, 1,
			nj, __jdrdot
		);
	}
}


#include "pnacc_kernel_common.h"


static inline void
p2p_pnacc_kernel_core(
	const CLIGHT clight,
	uint_t lid,
	concat(PNAcc_Data, WPT) *jp,
	local concat(PNAcc_Data, WGSIZE) *ip)
// flop count: 65+24+???
{
	#pragma unroll
	for (uint_t w = 0; w < WGSIZE; w += NLANES) {
		for (uint_t l = w; l < w + NLANES; ++l) {
			uint_t i = lid^l;
			for (uint_t j = 0; j < WPT; ++j) {
				for (uint_t k = 0; k < SIMD; ++k) {
					real_tn ee = ip->e2[i] + jp->e2[j];
					real_tn rx = ip->rx[i] - jp->rx[j];
					real_tn ry = ip->ry[i] - jp->ry[j];
					real_tn rz = ip->rz[i] - jp->rz[j];
					real_tn vx = ip->vx[i] - jp->vx[j];
					real_tn vy = ip->vy[i] - jp->vy[j];
					real_tn vz = ip->vz[i] - jp->vz[j];

					real_tn rr = ee;
					rr += rx * rx;
					rr += ry * ry;
					rr += rz * rz;

					real_tn inv_r = rsqrt(rr);
					real_tn inv_r2 = inv_r * inv_r;

					real_tn im = ip->m[i];
					real_tn im2 = im * im;
					real_tn im_r = im * inv_r;
					real_tn iv2  = ip->vx[i] * ip->vx[i];
							iv2 += ip->vy[i] * ip->vy[i];
							iv2 += ip->vz[i] * ip->vz[i];
					real_tn iv4 = iv2 * iv2;
					real_tn niv  = rx * ip->vx[i];
							niv += ry * ip->vy[i];
							niv += rz * ip->vz[i];
					niv *= inv_r;
					real_tn niv2 = niv * niv;

					real_tn jm = jp->m[j];
					real_tn jm2 = jm * jm;
					real_tn jm_r = jm * inv_r;
					real_tn jv2  = jp->vx[j] * jp->vx[j];
							jv2 += jp->vy[j] * jp->vy[j];
							jv2 += jp->vz[j] * jp->vz[j];
					real_tn jv4 = jv2 * jv2;
					real_tn njv  = rx * jp->vx[j];
							njv += ry * jp->vy[j];
							njv += rz * jp->vz[j];
					njv *= inv_r;
					real_tn njv2 = njv * njv;

					real_tn imjm = im * jm;
					real_tn vv  = vx * vx;
							vv += vy * vy;
							vv += vz * vz;
					real_tn ivjv  = ip->vx[i] * jp->vx[j];
							ivjv += ip->vy[i] * jp->vy[j];
							ivjv += ip->vz[i] * jp->vz[j];
					real_tn nv  = rx * vx;
							nv += ry * vy;
							nv += rz * vz;
					nv *= inv_r;
					real_tn nvnv = nv * nv;
					real_tn nivnjv = niv * njv;

					uint_t order = clight.order;
					real_t inv_c = clight.inv1;

					real_tn ipnA = pnterms_A(jm, jm2, jm_r, jv2, jv4, -njv, njv2,
											 im, im2, im_r, iv2, iv4, -niv, niv2,
											 imjm, inv_r, inv_r2, vv, ivjv,
											 nv, nvnv, nivnjv, order, inv_c);
					real_tn ipnB = pnterms_B(jm, jm2, jm_r, jv2, jv4, -njv, njv2,
											 im, im2, im_r, iv2, iv4, -niv, niv2,
											 imjm, inv_r, inv_r2, vv, ivjv,
											 nv, nvnv, nivnjv, order, inv_c);
					jp->pnax[j] += ipnA * rx + ipnB * vx;
					jp->pnay[j] += ipnA * ry + ipnB * vy;
					jp->pnaz[j] += ipnA * rz + ipnB * vz;

					real_tn jpnA = pnterms_A(im, im2, im_r, iv2, iv4, +niv, niv2,
											 jm, jm2, jm_r, jv2, jv4, +njv, njv2,
											 imjm, inv_r, inv_r2, vv, ivjv,
											 nv, nvnv, nivnjv, order, inv_c);
					real_tn jpnB = pnterms_B(im, im2, im_r, iv2, iv4, +niv, niv2,
											 jm, jm2, jm_r, jv2, jv4, +njv, njv2,
											 imjm, inv_r, inv_r2, vv, ivjv,
											 nv, nvnv, nivnjv, order, inv_c);
					ip->pnax[i] += jpnA * rx + jpnB * vx;
					ip->pnay[i] += jpnA * ry + jpnB * vy;
					ip->pnaz[i] += jpnA * rz + jpnB * vz;

					simd_shuff_PNAcc_Data(j, jp);
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}


static inline void
pnacc_kernel_core(
	const CLIGHT clight,
	uint_t lid,
	concat(PNAcc_Data, WPT) *jp,
	local concat(PNAcc_Data, WGSIZE) *ip)
// flop count: 65+12+???
{
	#pragma unroll
	for (uint_t w = 0; w < WGSIZE; w += NLANES) {
		for (uint_t l = w; l < w + NLANES; ++l) {
			uint_t i = lid^l;
			for (uint_t j = 0; j < WPT; ++j) {
				for (uint_t k = 0; k < SIMD; ++k) {
					real_tn ee = ip->e2[i] + jp->e2[j];
					real_tn rx = ip->rx[i] - jp->rx[j];
					real_tn ry = ip->ry[i] - jp->ry[j];
					real_tn rz = ip->rz[i] - jp->rz[j];
					real_tn vx = ip->vx[i] - jp->vx[j];
					real_tn vy = ip->vy[i] - jp->vy[j];
					real_tn vz = ip->vz[i] - jp->vz[j];

					real_tn rr = ee;
					rr += rx * rx;
					rr += ry * ry;
					rr += rz * rz;

					real_tn inv_r = rsqrt(rr);
					inv_r = (rr > ee) ? (inv_r):(0);
					real_tn inv_r2 = inv_r * inv_r;

					real_tn im = ip->m[i];
					real_tn im2 = im * im;
					real_tn im_r = im * inv_r;
					real_tn iv2  = ip->vx[i] * ip->vx[i];
							iv2 += ip->vy[i] * ip->vy[i];
							iv2 += ip->vz[i] * ip->vz[i];
					real_tn iv4 = iv2 * iv2;
					real_tn niv  = rx * ip->vx[i];
							niv += ry * ip->vy[i];
							niv += rz * ip->vz[i];
					niv *= inv_r;
					real_tn niv2 = niv * niv;

					real_tn jm = jp->m[j];
					real_tn jm2 = jm * jm;
					real_tn jm_r = jm * inv_r;
					real_tn jv2  = jp->vx[j] * jp->vx[j];
							jv2 += jp->vy[j] * jp->vy[j];
							jv2 += jp->vz[j] * jp->vz[j];
					real_tn jv4 = jv2 * jv2;
					real_tn njv  = rx * jp->vx[j];
							njv += ry * jp->vy[j];
							njv += rz * jp->vz[j];
					njv *= inv_r;
					real_tn njv2 = njv * njv;

					real_tn imjm = im * jm;
					real_tn vv  = vx * vx;
							vv += vy * vy;
							vv += vz * vz;
					real_tn ivjv  = ip->vx[i] * jp->vx[j];
							ivjv += ip->vy[i] * jp->vy[j];
							ivjv += ip->vz[i] * jp->vz[j];
					real_tn nv  = rx * vx;
							nv += ry * vy;
							nv += rz * vz;
					nv *= inv_r;
					real_tn nvnv = nv * nv;
					real_tn nivnjv = niv * njv;

					uint_t order = clight.order;
					real_t inv_c = clight.inv1;

					real_tn ipnA = pnterms_A(jm, jm2, jm_r, jv2, jv4, -njv, njv2,
											 im, im2, im_r, iv2, iv4, -niv, niv2,
											 imjm, inv_r, inv_r2, vv, ivjv,
											 nv, nvnv, nivnjv, order, inv_c);
					real_tn ipnB = pnterms_B(jm, jm2, jm_r, jv2, jv4, -njv, njv2,
											 im, im2, im_r, iv2, iv4, -niv, niv2,
											 imjm, inv_r, inv_r2, vv, ivjv,
											 nv, nvnv, nivnjv, order, inv_c);
					jp->pnax[j] += ipnA * rx + ipnB * vx;
					jp->pnay[j] += ipnA * ry + ipnB * vy;
					jp->pnaz[j] += ipnA * rz + ipnB * vz;

//					real_tn jpnA = pnterms_A(im, im2, im_r, iv2, iv4, +niv, niv2,
//											 jm, jm2, jm_r, jv2, jv4, +njv, njv2,
//											 imjm, inv_r, inv_r2, vv, ivjv,
//											 nv, nvnv, nivnjv, order, inv_c);
//					real_tn jpnB = pnterms_B(im, im2, im_r, iv2, iv4, +niv, niv2,
//											 jm, jm2, jm_r, jv2, jv4, +njv, njv2,
//											 imjm, inv_r, inv_r2, vv, ivjv,
//											 nv, nvnv, nivnjv, order, inv_c);
//					ip->pnax[i] += jpnA * rx + jpnB * vx;
//					ip->pnay[i] += jpnA * ry + jpnB * vy;
//					ip->pnaz[i] += jpnA * rz + jpnB * vz;

					simd_shuff_PNAcc_Data(j, jp);
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}


kernel void
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
pnacc_kernel_rectangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
	const uint_t nj,
	global const real_t __jm[],
	global const real_t __je2[],
	global const real_t __jrdot[],
//	const CLIGHT clight,
	constant const CLIGHT * clight,
	global real_t __ipnacc[],
	global real_t __jpnacc[])
{
	local concat(PNAcc_Data, WGSIZE) _ip;
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t block = WGSIZE * SIMD * WPT;
	for (uint_t jj = block * grp;
				jj < nj;
				jj += block * ngrps) {
		concat(PNAcc_Data, WPT) jp = {{{0}}};
		concat(load_PNAcc_Data, WPT)(
			&jp, jj + lid, WGSIZE, SIMD,
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
						ilid += WGSIZE * SIMD) {
				concat(PNAcc_Data, 1) ip = {{{0}}};
				concat(load_PNAcc_Data, 1)(
					&ip, ii + ilid, WGSIZE, SIMD,
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
				_ip.pnax[lid] = ip.pnax[0];
				_ip.pnay[lid] = ip.pnay[0];
				_ip.pnaz[lid] = ip.pnaz[0];

				if (ii != jj) {
					p2p_pnacc_kernel_core(*clight, lid, &jp, &_ip);
				} else {
					p2p_pnacc_kernel_core(*clight, lid, &jp, &_ip);
				}

				ip.pnax[0] = -_ip.pnax[lid];
				ip.pnay[0] = -_ip.pnay[lid];
				ip.pnaz[0] = -_ip.pnaz[lid];
				concat(store_PNAcc_Data, 1)(
					&ip, ii + ilid, WGSIZE, SIMD,
					ni, __ipnacc
				);
			}
		}

		concat(store_PNAcc_Data, WPT)(
			&jp, jj + lid, WGSIZE, SIMD,
			nj, __jpnacc
		);
	}
}


kernel void
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
pnacc_kernel_triangle(
	const uint_t ni,
	global const real_t __im[],
	global const real_t __ie2[],
	global const real_t __irdot[],
//	const CLIGHT clight,
	constant const CLIGHT * clight,
	global real_t __ipnacc[])
{
	// ------------------------------------------------------------------------
	const uint_t nj = ni;
	global const real_t *__jm = __im;
	global const real_t *__je2 = __ie2;
	global const real_t *__jrdot = __irdot;
	global real_t *__jpnacc = __ipnacc;
	// ------------------------------------------------------------------------

	local concat(PNAcc_Data, WGSIZE) _ip;
	uint_t lid = get_local_id(0);
	uint_t grp = get_group_id(0);
	uint_t ngrps = get_num_groups(0);
	uint_t block = WGSIZE * SIMD * WPT;
	for (uint_t jj = block * grp;
				jj < nj;
				jj += block * ngrps) {
		concat(PNAcc_Data, WPT) jp = {{{0}}};
		concat(load_PNAcc_Data, WPT)(
			&jp, jj + lid, WGSIZE, SIMD,
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
						ilid += WGSIZE * SIMD) {
				concat(PNAcc_Data, 1) ip = {{{0}}};
				concat(load_PNAcc_Data, 1)(
					&ip, ii + ilid, WGSIZE, SIMD,
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
				_ip.pnax[lid] = ip.pnax[0];
				_ip.pnay[lid] = ip.pnay[0];
				_ip.pnaz[lid] = ip.pnaz[0];

				if (ii != jj) {
					p2p_pnacc_kernel_core(*clight, lid, &jp, &_ip);
				} else {
					pnacc_kernel_core(*clight, lid, &jp, &_ip);
				}

				ip.pnax[0] = -_ip.pnax[lid];
				ip.pnay[0] = -_ip.pnay[lid];
				ip.pnaz[0] = -_ip.pnaz[lid];
				concat(store_PNAcc_Data, 1)(
					&ip, ii + ilid, WGSIZE, SIMD,
					ni, __ipnacc
				);
			}
		}

		concat(store_PNAcc_Data, WPT)(
			&jp, jj + lid, WGSIZE, SIMD,
			nj, __jpnacc
		);
	}
}


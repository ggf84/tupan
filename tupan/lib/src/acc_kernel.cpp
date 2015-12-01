#include "acc_kernel_common.h"


static inline void
p2p_acc_kernel_core(auto& i, auto& j)
{
	auto rx = i.rx - j.rx;
	auto ry = i.ry - j.ry;
	auto rz = i.rz - j.rz;
	auto e2 = i.e2 + j.e2;
	auto r2 = rx * rx + ry * ry + rz * rz;
	auto inv_r3 = smoothed_inv_r3(r2, e2);	// 5 FLOPs
	{	// i-particle
		auto m_r3 = j.m * inv_r3;
		i.ax -= m_r3 * rx;
		i.ay -= m_r3 * ry;
		i.az -= m_r3 * rz;
	}
	{	// j-particle
		auto m_r3 = i.m * inv_r3;
		j.ax += m_r3 * rx;
		j.ay += m_r3 * ry;
		j.az += m_r3 * rz;
	}
}
// Total flop count: 28


static inline void
p2p_acc_kernel_rectangle(auto i0, auto i1, auto j0, auto j1)
{
	for (auto i = i0; i < i1; ++i) {
		for (auto j = j0; j < j1; ++j) {
			p2p_acc_kernel_core(*i, *j);
		}
	}
}


static inline void
p2p_acc_kernel_triangle(auto i0, auto i1)
{
	for (auto i = i0; i < i1; ++i) {
		for (auto j = i+1; j < i1; ++j) {
			p2p_acc_kernel_core(*i, *j);
		}
	}
}


static inline void
rectangle(auto i0, auto i1, auto j0, auto j1)
{
	constexpr auto threshold = 256;
	auto di = i1 - i0;
	auto dj = j1 - j0;

	if (di > threshold && dj > threshold) {
		auto im = i0 + di / 2;
		auto jm = j0 + dj / 2;

		#pragma omp task
		{ rectangle(i0, im, j0, jm); }
		rectangle(im, i1, jm, j1);
		#pragma omp taskwait

		#pragma omp task
		{ rectangle(i0, im, jm, j1); }
		rectangle(im, i1, j0, jm);
		#pragma omp taskwait
	} else {
		p2p_acc_kernel_rectangle(i0, i1, j0, j1);
	}
}


static inline void
triangle(auto i0, auto i1)
{
	constexpr auto threshold = 256;
	auto di = i1 - i0;

	if (di > threshold) {
		auto im = i0 + di / 2;

		#pragma omp task
		{ triangle(i0, im); }
		triangle(im, i1);
		#pragma omp taskwait

		rectangle(i0, im, im, i1);
	} else {
		p2p_acc_kernel_triangle(i0, i1);
	}
}


void
acc_kernel_rectangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __irx[],
	const real_t __iry[],
	const real_t __irz[],
	const real_t __ie2[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __jrx[],
	const real_t __jry[],
	const real_t __jrz[],
	const real_t __je2[],
	real_t __iax[],
	real_t __iay[],
	real_t __iaz[],
	real_t __jax[],
	real_t __jay[],
	real_t __jaz[])
{
	vector<Acc_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.ax = 0;
		ip.ay = 0;
		ip.az = 0;
		ip.rx = __irx[i];
		ip.ry = __iry[i];
		ip.rz = __irz[i];
		ip.e2 = __ie2[i];
		ip.m = __im[i];
	}

	vector<Acc_Data> jpart{nj};
	for (auto& jp : jpart) {
		auto j = &jp - &jpart[0];
		jp.ax = 0;
		jp.ay = 0;
		jp.az = 0;
		jp.rx = __jrx[j];
		jp.ry = __jry[j];
		jp.rz = __jrz[j];
		jp.e2 = __je2[j];
		jp.m = __jm[j];
	}

	#pragma omp parallel
	#pragma omp single
	rectangle(begin(ipart), end(ipart), begin(jpart), end(jpart));

	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		__iax[i] = ip.ax;
		__iay[i] = ip.ay;
		__iaz[i] = ip.az;
	}

	for (auto& jp : jpart) {
		auto j = &jp - &jpart[0];
		__jax[j] = jp.ax;
		__jay[j] = jp.ay;
		__jaz[j] = jp.az;
	}
}


void
acc_kernel_triangle(
	const uint_t ni,
	const real_t __im[],
	const real_t __irx[],
	const real_t __iry[],
	const real_t __irz[],
	const real_t __ie2[],
	real_t __iax[],
	real_t __iay[],
	real_t __iaz[])
{
	vector<Acc_Data> ipart{ni};
	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		ip.ax = 0;
		ip.ay = 0;
		ip.az = 0;
		ip.rx = __irx[i];
		ip.ry = __iry[i];
		ip.rz = __irz[i];
		ip.e2 = __ie2[i];
		ip.m = __im[i];
	}

	#pragma omp parallel
	#pragma omp single
	triangle(begin(ipart), end(ipart));

	for (auto& ip : ipart) {
		auto i = &ip - &ipart[0];
		__iax[i] = ip.ax;
		__iay[i] = ip.ay;
		__iaz[i] = ip.az;
	}
}


void
acc_kernel(
	const uint_t ni,
	const real_t __im[],
	const real_t __irx[],
	const real_t __iry[],
	const real_t __irz[],
	const real_t __ie2[],
	const uint_t nj,
	const real_t __jm[],
	const real_t __jrx[],
	const real_t __jry[],
	const real_t __jrz[],
	const real_t __je2[],
	real_t __iax[],
	real_t __iay[],
	real_t __iaz[])
{
	for (uint_t i = 0; i < ni; ++i) {
		Acc_IData ip = (Acc_IData){
			.ax = 0,
			.ay = 0,
			.az = 0,
			.rx = __irx[i],
			.ry = __iry[i],
			.rz = __irz[i],
			.e2 = __ie2[i],
			.m = __im[i],
		};

		for (uint_t j = 0; j < nj; ++j) {
			Acc_JData jp = (Acc_JData){
				.rx = __jrx[j],
				.ry = __jry[j],
				.rz = __jrz[j],
				.e2 = __je2[j],
				.m = __jm[j],
			};
			ip = acc_kernel_core(ip, jp);
		}

		__iax[i] = ip.ax;
		__iay[i] = ip.ay;
		__iaz[i] = ip.az;
	}
}


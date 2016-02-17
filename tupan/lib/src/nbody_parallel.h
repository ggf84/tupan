#ifndef __NBODY_PARALLEL_H__
#define __NBODY_PARALLEL_H__

#include <omp.h>


constexpr auto threshold = 256;


static inline void
p2p_interaction(auto i0, auto i1, auto j0, auto j1, auto fn)
{
	for (auto i = i0; i < i1; ++i) {
		for (auto j = j0; j < j1; ++j) {
			fn(*i, *j);
		}
	}
}


static inline void
rectangle(auto i0, auto i1, auto j0, auto j1, auto fn)
{
	const auto di = i1 - i0;
	const auto dj = j1 - j0;

	if (di > threshold && dj > threshold) {
		const auto im = i0 + di / 2;
		const auto jm = j0 + dj / 2;

		#pragma omp task
		{ rectangle(i0, im, j0, jm, fn); }
		rectangle(im, i1, jm, j1, fn);
		#pragma omp taskwait

		#pragma omp task
		{ rectangle(i0, im, jm, j1, fn); }
		rectangle(im, i1, j0, jm, fn);
		#pragma omp taskwait
	} else {
		p2p_interaction(i0, i1, j0, j1, fn);
	}
}


static inline void
triangle(auto i0, auto i1, auto fn)
{
	const auto di = i1 - i0;

	if (di > 1) {
		const auto im = i0 + di / 2;

		if (di > threshold) {
			#pragma omp task
			{ triangle(i0, im, fn); }
			triangle(im, i1, fn);
			#pragma omp taskwait
		} else {
			triangle(i0, im, fn);
			triangle(im, i1, fn);
		}

		rectangle(i0, im, im, i1, fn);
	}
}

#endif // __NBODY_PARALLEL_H__

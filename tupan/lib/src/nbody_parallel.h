#ifndef __NBODY_PARALLEL_H__
#define __NBODY_PARALLEL_H__

#include <omp.h>


constexpr auto threshold = 256;


template<typename I, typename J, typename F>
static inline void
p2p_interaction(I i0, I i1, J j0, J j1, F fn)
{
	for (auto i = i0; i < i1; ++i) {
		for (auto j = j0; j < j1; ++j) {
			fn(*i, *j);
		}
	}
}


template<typename I, typename J, typename F>
static inline void
rectangle(I i0, I i1, J j0, J j1, F fn)
{
	const auto di = i1 - i0;
	const auto dj = j1 - j0;
	if (di < threshold || dj < threshold) {
		return p2p_interaction(i0, i1, j0, j1, fn);
	}

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
}


template<typename I, typename F>
static inline void
triangle(I i0, I i1, F fn)
{
	const auto di = i1 - i0;
	if (di < 2) return;

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

#endif // __NBODY_PARALLEL_H__

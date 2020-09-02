#ifndef LPSQN_P_H
#define LPSQN_P_H
#include <cstddef>

namespace lp {

inline void copy(double * __restrict__ x, double const * __restrict__ y,
                 size_t const dim) noexcept {
  for(size_t i = 0; i < dim; ++i, ++x, ++y)
    *x = *y;
}

inline void vec_diff
(double const * __restrict__ x, double const * __restrict__ y,
 double * __restrict__ res, size_t const n) noexcept {
  for(size_t i = 0; i < n; ++i, ++x, ++y, ++res)
    *res = *x - *y;
}

inline double vec_dot(double const *x, size_t const n) noexcept {
  double out(0.);
  for(size_t i = 0; i < n; ++i, ++x)
    out += *x * *x;
  return out;
}

inline double vec_dot
(double const * __restrict__ x, double const * __restrict__ y,
 size_t const n) noexcept {
  double out(0.);
  for(size_t i = 0; i < n; ++i, ++x, ++y)
    out += *x * *y;
  return out;
}

/***
 computes b <- b + Xx where is is a n x n matrix and x is n-dimensional vector
 */
inline void mat_vec_dot
(double const *__restrict__ X, double const * __restrict__ x,
 double * __restrict__ res, size_t const n) noexcept {
  for(size_t j = 0; j < n; ++j, ++x){
    double * r = res;
    for(size_t i = 0; i < n; ++i, ++X, ++r)
      *r += *X * *x;
  }
}

/***
 computes b <- b + Xx where b and x are seperated into an nb1 and bn2
 dimensional vector.
 */
inline void mat_vec_dot
(double const *__restrict__ X, double const * __restrict__ x1,
 double const * __restrict__ x2, double * const __restrict__ r1,
 double * const __restrict__ r2,
 size_t const n1, size_t const n2) noexcept {
  for(size_t j = 0; j < n1; ++j, ++x1){
    {
      double * r1i = r1;
      for(size_t i = 0; i < n1; ++i, ++X, ++r1i)
        *r1i += *X * *x1;
    }

    double * r2i = r2;
    for(size_t i = 0; i < n2; ++i, ++X, ++r2i)
      *r2i += *X * *x1;
  }

  for(size_t j = 0; j < n2; ++j, ++x2){
    {
      double * r1i = r1;
      for(size_t i = 0; i < n1; ++i, ++X, ++r1i)
        *r1i += *X * *x2;
    }

    double * r2i = r2;
    for(size_t i = 0; i < n2; ++i, ++X, ++r2i)
      *r2i += *X * *x2;
  }
}

/***
 performs a rank one update X <- X + scal * x.x^T.
 */
inline void rank_one_update
(double * __restrict__ X, double const * __restrict__ x,
 double const scal, size_t const n)
noexcept {
  double const * xj = x;
  for(size_t j = 0; j < n; ++j, ++xj){
    double const * xi = x;
    for(size_t i = 0; i < j; ++i, ++xi){
      double const val = scal * *xj * *xi;
      *(X + i     + j * n) += val;
      *(X + i * n + j    ) += val;
    }
    *(X + j + j * n) += scal * *xj * *xj;
  }
}
} // namespace lp

#endif

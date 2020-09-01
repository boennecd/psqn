#ifndef LPSQN_P_H
#define LPSQN_P_H

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

inline double vec_dot(double const *x, double const *y,
                      size_t const n) noexcept {
  double out(0.);
  for(size_t i = 0; i < n; ++i, ++x, ++y)
    out += *x * *y;
  return out;
}

/***
 computes b = Xx where is is a n x n matrix and x is n-dimensional vector
 */
inline void mat_vec_dot
(double const *__restrict__ X, double const * __restrict__ x,
 double * __restrict__ res, size_t const n) noexcept {
  for(size_t i = 0; i < n; ++i, ++x){
    double * r = res;
    for(size_t i = 0; i < n; ++i, ++X, ++r)
      *r += *X * *x;
  }
}

inline void rank_one_update
(double * __restrict__ B, double const * __restrict__ x,
 double const scal, size_t const n)
noexcept {
  // TODO: check this function
  double const * xj = x;
  double * bi = B;
  for(size_t j = 0; j < n; ++j, ++xj){
    double const * xi = x;
    double * bj = B + j;
    for(size_t i = 0; i < n - j; ++i, ++xi, ++bi, bj += n){
      double const val = scal * *xj * *xi;
      *bi = val;
      *bj = val;
    }
  }
}
} // namespace lp

#endif

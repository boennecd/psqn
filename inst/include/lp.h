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
 computes b <- b + Xx where is is a n x n symmetric matrix containing only
 the upper triangular and x is a n-dimensional vector.
 */
inline void mat_vec_dot
(double const * __restrict__  X, double const * const __restrict__ x,
 double * const __restrict__ res, size_t const n) noexcept {
  double const * xj = x;
  double * rj = res;
  for(size_t j = 0; j < n; ++j, ++xj, ++rj){
    double const *xi = x;
    double * ri = res;

    for(size_t i = 0L; i < j; ++i, ++X, ++ri, ++xi){
      *ri += *X * *xj;
      *rj += *X * *xi;
    }
    *rj += *X++ * *xi;
  }
}

/** util class which jumps from one memory location to another. */
template<class T>
class sep_mem {
  T *       cur,
    * const d1_end,
    * const d2;
public:
  sep_mem(T * d1, T * d2, size_t const n1):
  cur(n1 > 0 ? d1 : d2), d1_end(d1 + n1), d2(d2) { }

  inline sep_mem& operator++() noexcept {
    if(++cur != d1_end)
      return *this;

    cur = d2;
    return *this;
  }

  inline operator T*() const noexcept {
    return cur;
  }
};

/***
 computes b <- b + Xx where b and x are seperated into an nb1 and bn2
 dimensional vector.
 */
inline void mat_vec_dot
(double const *__restrict__ X, double const * __restrict__ x1,
 double const * __restrict__ x2, double * const __restrict__ r1,
 double * const __restrict__ r2,
 size_t const n1, size_t const n2) noexcept {
  sep_mem<double const> xj(x1, x2, n1);
  sep_mem<double>       rj(r1, r2, n1);
  size_t const n = n1 + n2;
  for(size_t j = 0; j < n; ++j, ++xj, ++rj){
    sep_mem<double const> xi(x1, x2, n1);
    sep_mem<double>       ri(r1, r2, n1);

    for(size_t i = 0L; i < j; ++i, ++X, ++ri, ++xi){
      *ri += *X * *xj;
      *rj += *X * *xi;
    }
    *rj += *X++ * *xi;
  }
}

/***
 performs a rank one update X <- X + scal * x.x^T where X is a symmetric
 matrix contaning only the upper triangular.
 */
inline void rank_one_update
(double * __restrict__ X, double const * __restrict__ x,
 double const scal, size_t const n)
noexcept {
  double const * xj = x;
  for(size_t j = 0; j < n; ++j, ++xj){
    double const * xi = x;
    for(size_t i = 0; i < j; ++i, ++xi, ++X){
      *X += scal * *xj * *xi;
    }
    *X++ += scal * *xj * *xj;
  }
}
} // namespace lp

#endif

#ifndef LPSQN_P_H
#define LPSQN_P_H
#include <algorithm>
#include "constant.h"
#include "psqn-misc.h"
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef PSQN_W_LAPACK
// TODO: need to deal with the possible underscore in Fotran definitions
#include <R_ext/RS.h>
#endif

namespace lp {
using PSQN::psqn_uint;

inline void copy(double * PSQN_RESTRICT x, double const * PSQN_RESTRICT y,
                 psqn_uint const dim) noexcept {
  std::copy(y, y + dim, x);
}

inline void vec_diff
(double const * PSQN_RESTRICT x, double const * PSQN_RESTRICT y,
 double * PSQN_RESTRICT res, psqn_uint const n) noexcept {
  for(psqn_uint i = 0; i < n; ++i, ++x, ++y, ++res)
    *res = *x - *y;
}

template<bool DoParallel>
double vec_dot(double const *x, psqn_uint const n) noexcept {
  return std::accumulate(x, x + n, 0., [](double const res, double xi){
    return res + xi * xi;
  });
}

// template<>
// inline double vec_dot<true>(double const *x, psqn_uint const n) noexcept {
//   double out(0.);
// #ifdef _OPENMP
// #pragma omp parallel for reduction(+:out)
// #endif
//   for(psqn_uint i = 0; i < n; ++i)
//     out += x[i] * x[i];
//   return out;
// }

template<bool DoParallel>
double vec_dot
(double const * PSQN_RESTRICT x, double const * PSQN_RESTRICT y,
 psqn_uint const n) noexcept {
  return std::inner_product(x, x + n, y, 0.);
}

// template<>
// inline double vec_dot<true>
// (double const * PSQN_RESTRICT x, double const * PSQN_RESTRICT y,
//  psqn_uint const n) noexcept {
//   double out(0.);
// #ifdef _OPENMP
// #pragma omp parallel for reduction(+:out)
// #endif
//   for(psqn_uint i = 0; i < n; ++i)
//     out += x[i] * y[i];
//   return out;
// }

/**
 computes b <- b + Xx where is is a n x n symmetric matrix containing only
 the upper triangular and x is a n-dimensional vector.
 */
inline void mat_vec_dot
(double const * PSQN_RESTRICT  X, double const * const PSQN_RESTRICT x,
 double * const PSQN_RESTRICT res, psqn_uint const n) noexcept {
  double const * xj = x;
  double * rj = res;
  for(psqn_uint j = 0; j < n; ++j, ++xj, ++rj){
    double const *xi = x;
    double * ri = res;

    for(psqn_uint i = 0L; i < j; ++i, ++X, ++ri, ++xi){
      *ri += *X * *xj;
      *rj += *X * *xi;
    }
    *rj += *X++ * *xi;
  }
}

/**
 computes b <- b + Xx[idx] where X is a q x q symmetric matrix
 containing only the upper triangular and x is a n-dimensional vector (n greater
 than q) and idx is a q-dimensional vector of indices in the range of 1 to n.
 */
inline void mat_vec_dot
(double const * PSQN_RESTRICT  X, double const * const PSQN_RESTRICT x,
 double * const PSQN_RESTRICT res, psqn_uint const n, psqn_uint const *idx) noexcept {
  psqn_uint const * idx_j = idx;
  double * res_j = res;
  for(psqn_uint j = 0; j < n; ++j, ++idx_j, ++res_j){
    psqn_uint const * idx_i = idx;
    double * res_i = res;

    for(psqn_uint i = 0L; i < j; ++i, ++X, ++idx_i, ++res_i){
      *res_i += *X * x[*idx_j];
      *res_j += *X * x[*idx_i];
    }
    *res_j += *X++ * x[*idx_i];
  }
}

/**
 computes b <- b + Xx where b and x are separated into an nb1 and bn2
 dimensional vector.
 */
inline void mat_vec_dot
(double const * PSQN_RESTRICT X, double const * PSQN_RESTRICT x1,
 double const * PSQN_RESTRICT x2, double * const PSQN_RESTRICT r1,
 double * const PSQN_RESTRICT r2,
 psqn_uint const n1, psqn_uint const n2) noexcept {
  /*
   Write X as the matrix
   [C_11, C_21^T,
   C_21, C_22]

   Then we handle the C_11 part, then the C_21 and C_21^T and then the last block
   */

  for(psqn_uint j = 0; j < n1; ++j){
    for(psqn_uint i = 0; i < j; ++i, ++X){
      r1[i] += *X * x1[j];
      r1[j] += *X * x1[i];
    }
    r1[j] += *X++ * x1[j];
  }

  {
    double const * X_block{X};
    for(psqn_uint j = 0; j < n2; ++j, X_block += j)
      for(psqn_uint i = 0; i < n1; ++i, ++X_block){
        r1[i] += *X_block * x2[j];
        r2[j] += *X_block * x1[i];
      }
  }
  {
    double const * X_block{X + n1};
    for(psqn_uint j = 0; j < n2; ++j, X_block += n1){
      for(psqn_uint i = 0; i < j; ++i, ++X_block){
        r2[i] += *X_block * x2[j];
        r2[j] += *X_block * x2[i];
      }
      r2[j] += *X_block++ * x2[j];
    }
  }
}

/**
 computes b <- b + Xx where b and x are separated into an nb1 and bn2
 dimensional vector but excluding the first n1 x n1 block of X.

 X is assumed to be symmetric and we only store the upper triangular part.
 */
inline void mat_vec_dot_excl_first
(double const * PSQN_RESTRICT X, double const * PSQN_RESTRICT x1,
 double const * PSQN_RESTRICT x2, double * const PSQN_RESTRICT r1,
 double * const PSQN_RESTRICT r2,
 psqn_uint const n1, psqn_uint const n2) noexcept {
  /*
    Write X as the matrix
     [C_11, C_21^T,
      C_21, C_22]

   Then we handle the C_21 and C_21^T parts first and then the last block
   */

  X += (n1 * (n1 + 1)) / 2; // never needed
  {
    double const * X_block{X};
    for(psqn_uint j = 0; j < n2; ++j, X_block += j)
      for(psqn_uint i = 0; i < n1; ++i, ++X_block){
        r1[i] += *X_block * x2[j];
        r2[j] += *X_block * x1[i];
      }
  }
  {
    double const * X_block{X + n1};
    for(psqn_uint j = 0; j < n2; ++j, X_block += n1){
      for(psqn_uint i = 0; i < j; ++i, ++X_block){
        r2[i] += *X_block * x2[j];
        r2[j] += *X_block * x2[i];
      }
      r2[j] += *X_block++ * x2[j];
    }
  }
}

/**
 performs a rank one update X <- X + scal * x.x^T where X is a symmetric
 matrix containing only the upper triangular.
 */
inline void rank_one_update
(double * PSQN_RESTRICT X, double const * PSQN_RESTRICT x,
 double const scal, psqn_uint const n)
noexcept {
  double const * xj = x;
  for(psqn_uint j = 0; j < n; ++j, ++xj){
    double const * xi = x;
    for(psqn_uint i = 0; i <= j; ++i, ++xi, ++X)
      *X += scal * *xj * *xi;
  }
}

/***
 performs the update
   H <- (I - cx.y^T).H.(I - cy.x^T) + cx.x^T
      = H - cx.y^T.H - cH.y.x^T + c^2x.(y^T.H.y).x^T + cx.x^T
 where X is a symmetric matrix contaning only the upper triangular.
 */
inline void bfgs_update
  (double * PSQN_RESTRICT X, double const * PSQN_RESTRICT x,
   double const * PSQN_RESTRICT H_y, double const y_H_y,
   double const scal, psqn_uint const n)
  noexcept {
  double const * xj = x,
             * H_yj = H_y;
  double const y_H_y_p_scal = scal * (scal * y_H_y + 1);
  for(psqn_uint j = 0; j < n; ++j, ++xj, ++H_yj){
    double const * xi = x,
               * H_yi = H_y;
    for(psqn_uint i = 0; i <= j; ++i, ++xi, ++H_yi, ++X){
      *X += y_H_y_p_scal * *xj * *xi;
      *X -= scal * (*H_yj * *xi + *H_yi * *xj);
    }
  }
}

// Kahan summation algorithm
inline void Kahan(double &sum, double &comp, double const new_val) noexcept {
#ifdef PSQN_USE_KAHAN
  double const y = new_val - comp,
               t = sum + y;
  comp = (t - sum) - y;
  sum = t;
#else
  sum += new_val;
#endif
}

/***
 Kahan summation algorithm where the summation and the compensation lies in
 consecutive memory starting with the summation.
 */
inline void Kahan(double * sum_n_comp, double const new_val) noexcept {
  Kahan(*sum_n_comp, sum_n_comp[1], new_val);
}

/***
 * Computes the Cholesky decomposition of a n x n symmetric matrix A.
 * The function uses LAPACK's dpotrf routine. The result is also stored as the
 * n(n + 1)/2 upper triangle.
 *
 * If the factorization fails then the factorization is formed for
 *
 *    A + (epsilon + g) I
 *
 * where epsilon is 1e-3 and g is |A_min| where A_min is the smallest element
 * of the diagonal of A if A_min < 0. Otherwise, g is zero.
 *
 * If this fails, than epsilon <- epsilon * 10 is attempted. This repeat up
 * to a given number of times until the factorization succeed. If all of this
 * fails, the factorization is set to the absolute value of the diagonal
 * elements of A plus 1e-3.
 *
 * working memory of size n x n is required
 */
void setup_precondition_chol(double const *A, double *out, int const n,
                             double *wrk);

/***
 * given the output from setup_precondition_chol and a n dimensional vector x,
 * the fucntion solves
 *
 *   A.y = x
 *
 * or the solution with A replaced by A + (epsilon * 10^k + g) I.
 */
void precondition_chol_solve(double const *A_chol, double *x, int const n);

#ifdef PSQN_W_LAPACK
extern "C" {
  void F77_NAME(dpotrf)
  (const char*, const int*, double*, const int*, int*, size_t);

  void F77_NAME(dtpsv)
    (const char*, const char*, const char*, const int*,
    const double*, double*, const int*, size_t, size_t, size_t);
}

inline void setup_precondition_chol
  (double const *A, double *out, int const n, double *wrk){
  // find the smallest diagonal element
  double Amin{std::numeric_limits<double>::max()};
  {
    double const *a{A};
    for(int i = 0; i < n; ++i, a += n + 1)
      Amin = std::min(Amin, *a);
  }

  // attempt to compute the factorization
  size_t const n_ele = n * n;
  constexpr double epsilon_start{1e-3},
                   epsilon_mult{10};
  double epsilon{epsilon_start / epsilon_mult};
  int info{Amin > 0 ? 0 : 1};
  Amin = Amin < 0 ? std::abs(Amin) : 0; // maybe flip the sign

  for(unsigned i = 0; i < 10; ++i){
    if(info > 0)
      epsilon *= epsilon_mult;

    // setup the working memory object
    std::copy(A, A + n_ele, wrk);
    if(info > 0){
      // add to the diagonal
      double *w{wrk};
      for(int i = 0; i < n; ++i, w += n + 1)
        *w += Amin + epsilon;
    }

    F77_CALL(dpotrf)("U", &n, wrk, &n, &info, 1);
    if(info == 0)
      break;
  }

  if(info != 0){
    // set the factorization to be diagonal matrix
    std::fill(wrk, wrk + n_ele, 0);

    double const *a{A};
    double *w{wrk};
    for(int i = 0; i < n; ++i, w += n + 1, a += n + 1)
      *w += std::sqrt(std::abs(*a) + epsilon_start);
  }

  // copy the result
  double *o{out};
  double const *w{wrk};
  for(int j = 0; j < n; ++j, o += j, w += n)
    std::copy(w, w + j + 1, o);
}

inline void precondition_chol_solve
  (double const *A_chol, double *x, int const n){
  int const incx{1L};
  F77_CALL(dtpsv)("U", "T", "N", &n, A_chol, x, &incx, 1, 1, 1);
  F77_CALL(dtpsv)("U", "N", "N", &n, A_chol, x, &incx, 1, 1, 1);
}
#else // #ifdef PSQN_W_LAPACK

inline void setup_precondition_chol
  (double const *A, double *out, int const n, double *wrk){
  throw std::runtime_error("not compiled with LACPACK");
}

inline void precondition_chol_solve
  (double const *A_chol, double *x, int const n){
  throw std::runtime_error("not compiled with LACPACK");
}

#endif


} // namespace lp

#endif

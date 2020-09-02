#include "lp.h"
#include <testthat.h>

context("testing lp namespace") {
  test_that("vec_diff works") {
    /*
     x <- c(-0.63, 0.18, -0.84, 1.6)
     y <- c(0.33, -0.82, 0.49, 0.74)
     dput(x - y)
     */
    constexpr size_t const n(4L);
    constexpr double const x[4] = { -0.63, 0.18, -0.84, 1.6 },
                           y[4] = { 0.33, -0.82, 0.49, 0.74 },
                           ex[4] = { -0.96, 1, -1.33, 0.86 };
    double res[4];
    lp::vec_diff(x, y, res, n);

    for(size_t i = 0; i < n; ++i)
      expect_true(std::abs(res[i] - ex[i]) < 1e-8);
  }

  test_that("mat_vec_dot works (no seperation)") {
    /*
     X <- c(0.58, -2.21, 0.82, -1.99, -0.48, -2.21, 1.12, 0.59, 0.62, 0.42, 0.82, 0.59, 0.92, -0.06, 1.36, -1.99, 0.62, -0.06, -0.16, -0.1, -0.48, 0.42, 1.36, -0.1, 0.39)
     x <- c(-0.05, -1.38, -0.41, -0.39, -0.06)
     dput(matrix(X, 5) %*% x)
     dput(X[upper.tri(matrix(X, 5), TRUE)])
     */
    constexpr size_t const n(5L);
    constexpr double const X[(n * (n + 1l)) / 2L] = { 0.58, -2.21, 1.12, 0.82, 0.59, 0.92, -1.99, 0.62, -0.06, -0.16, -0.48, 0.42, 1.36, -0.1, 0.39 },
                           x[n]                   = { -0.05, -1.38, -0.41, -0.39, -0.06 },
                          ex[n]                   = { 3.4895, -1.944, -1.2906, -0.6631, -1.0976 };
    double res[n] = { 0, 0, 0, 0, 0 };

    lp::mat_vec_dot(X, x, res, n);

    for(size_t i = 0; i < n; ++i)
      expect_true(std::abs(res[i] - ex[i]) < 1e-8);
  }

  test_that("mat_vec_dot works (seperation)") {
    /*
     X <- c(0.58, -2.21, 0.82, -1.99, -0.48, -2.21, 1.12, 0.59, 0.62, 0.42, 0.82, 0.59, 0.92, -0.06, 1.36, -1.99, 0.62, -0.06, -0.16, -0.1, -0.48, 0.42, 1.36, -0.1, 0.39)
     x <- c(-1.82, 0.44, 1.09, 0.05, -0.23, 1.23, 0.11, 0.91, 0.03, 1.48)
     s <- c(-0.39, -0.36, 0.71, -0.02, -1.62, 1.45, -0.85, 0.43, -1.34, 0.11)

     idx <- c(1:3, 1:2 + 3L + 5L)
     s[idx] <- s[idx] + matrix(X, 5) %*% x[idx]
     dput(s)
     dput(X[upper.tri(matrix(X, 5), TRUE)])
     */
    constexpr size_t const n1(3L),
                           n2(2L),
                            n = n1 + n2,
                          xtr(5L);
    constexpr double const X [(n * (n + 1)) / 2L] = { 0.58, -2.21, 1.12, 0.82, 0.59, 0.92, -1.99, 0.62, -0.06, -0.16, -0.48, 0.42, 1.36, -0.1, 0.39 },
                           x [n + xtr]            = { -1.82, 0.44, 1.09, 0.05, -0.23, 1.23, 0.11, 0.91, 0.03, 1.48 },
                           ex[n + xtr]            = { -2.2943, 5.4383, 2.491, -0.02, -1.62, 1.45, -0.85, 0.43, 2.3364, 3.225 };
    double res[n + xtr] = { -0.39, -0.36, 0.71, -0.02, -1.62, 1.45, -0.85, 0.43, -1.34, 0.11 };

    lp::mat_vec_dot(X, x, x + n1 + xtr, res, res + n1 + xtr, n1, n2);
    for(size_t i = 0; i < n + xtr; ++i)
      expect_true(std::abs(res[i] - ex[i]) < 1e-8);
  }

  test_that("rank_one_update works") {
    /*
     X <- c(0.58, -2.21, 0.82, -1.99, -0.48, -2.21, 1.12, 0.59, 0.62, 0.42, 0.82, 0.59, 0.92, -0.06, 1.36, -1.99, 0.62, -0.06, -0.16, -0.1, -0.48, 0.42, 1.36, -0.1, 0.39)
     x <- c(-0.05, -1.38, -0.41, -0.39, -0.06)
     O <- matrix(X, 5) + x %o% x * 2.3
     dput(O[upper.tri(O, TRUE)])
     dput(X[upper.tri(O, TRUE)])
     */
    constexpr size_t const n(5L),
                      n_upper = (n * (n + 1L)) / 2L;
    double X[n_upper] = { 0.58, -2.21, 1.12, 0.82, 0.59, 0.92, -1.99, 0.62, -0.06, -0.16, -0.48, 0.42, 1.36, -0.1, 0.39 };
    constexpr double const x[n]       = { -0.05, -1.38, -0.41, -0.39, -0.06 },
                          ex[n_upper] = { 0.58575, -2.0513, 5.50012, 0.86715, 1.89134, 1.30663, -1.94515, 1.85786, 0.30777, 0.18983, -0.4731, 0.61044, 1.41658, -0.04618, 0.39828 };
    lp::rank_one_update(X, x, 2.3, n);

    for(size_t i = 0; i < n_upper; ++i)
      expect_true(std::abs(X[i] - ex[i]) < 1e-8);
  }
}

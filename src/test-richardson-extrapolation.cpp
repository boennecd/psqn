#include <testthat.h>
#include <richardson-extrapolation.h>
#include <vector>
#include <cmath>

context("testing numerical differentation") {
  test_that("works with a multivariate function") {
    auto func = [](double const x, double *out){
      out[0] = exp(2 * x);
      out[1] = std::sin(3 * x);
    };

    using comp_obj = PSQN::richardson_extrapolation<decltype(func)>;

    double res[2];
    constexpr unsigned order{6};
    double const tol{
      std::pow(std::numeric_limits<double>::epsilon(), 3./5.)};
    std::vector<double> wk_mem(comp_obj::n_wk_mem(2, order));
    constexpr double x{1.5};
    double const f1{exp(2 * x) * 2},
                 f2{std::cos(3 * x) * 3};

    comp_obj(func, order, wk_mem.data(), 1e-4, 2, tol, 2)(x, res);
    expect_true(std::abs(res[0] - f1) < 10 * std::abs(f1) * tol);
    expect_true(std::abs(res[1] - f2) < 10 * std::abs(f2) * tol);

    comp_obj(func, order, wk_mem.data(), 1e-4, 4, tol, 2)(x, res);
    expect_true(std::abs(res[0] - f1) < 10 * std::abs(f1) * tol);
    expect_true(std::abs(res[1] - f2) < 10 * std::abs(f2) * tol);

    comp_obj(func, order, wk_mem.data(), 1, 4, tol, 2)(x, res);
    expect_true(std::abs(res[0] - f1) < 10 * std::abs(f1) * tol);
    expect_true(std::abs(res[1] - f2) < 10 * std::abs(f2) * tol);
  }
}

#include "psqn.h"
#include "psqn-reporter.h"

using namespace Rcpp;

/**
 simple wrapper for an R function which takes three argument.

 Caution: it is much faster but is not gaurded against errors:
   https://stackoverflow.com/a/37846827/5861244
 */
class simple_R_func3 {
  SEXP fn, env;

public:
  simple_R_func3(SEXP fn, SEXP env):
  fn(fn), env(env) { }

  SEXP operator()(SEXP a1, SEXP a2, SEXP a3) const {
    SEXP R_fcall, out;
    PROTECT(R_fcall = Rf_lang4(fn, a1, a2, a3));
    PROTECT(out = Rf_eval(R_fcall, env));
    UNPROTECT(2);
    return out;
  }
};

/** same as simple_R_func3 but with only one argument */
class simple_R_func1 {
  SEXP fn, env;

public:
  simple_R_func1(SEXP fn, SEXP env):
  fn(fn), env(env) { }

  SEXP operator()(SEXP a1) const {
    SEXP R_fcall, out;
    PROTECT(R_fcall = Rf_lang2(fn, a1));
    PROTECT(out = Rf_eval(R_fcall, env));
    UNPROTECT(2);
    return out;
  }
};

class r_worker {
  simple_R_func3 f;
  IntegerVector f_idx;
  LogicalVector mutable scomp_grad = LogicalVector(1L);
  size_t const g_dim, p_dim,
               n_ele = g_dim + p_dim;

  NumericVector mutable par = NumericVector(g_dim + p_dim);

public:
  r_worker(SEXP func, int iarg, SEXP rho):
  f(func, rho),
  f_idx(([&](){
    IntegerVector out(1L);
    out[0] = iarg + 1L;
    return out;
  })()),
  g_dim(([&](){
    NumericVector dum(0);
    scomp_grad[0] = false;
    SEXP res =  f(f_idx, dum, scomp_grad);
    if(!Rf_isInteger(res) or !Rf_isVector(res) or Rf_xlength(res) != 2L)
      throw std::invalid_argument(
          "fn returns invalid lengths with zero length par");

    return *INTEGER(res);
  })()),
  p_dim(([&](){
    NumericVector dum(0);
    scomp_grad[0] = false;
    SEXP res =  f(f_idx, dum, scomp_grad);
    if(!Rf_isInteger(res) or !Rf_isVector(res) or Rf_xlength(res) != 2L)
      throw std::invalid_argument(
          "fn returns invalid lengths with zero length par");

    return *(INTEGER(res) + 1L);
  })())
  { };

  size_t global_dim() const {
    return g_dim;
  };
  size_t private_dim() const {
    return p_dim;
  }

  double func(double const *point) const {
    for(size_t j = 0; j < n_ele; ++j, ++point)
      par[j] = *point;
    scomp_grad[0] = false;
    SEXP res =  f(f_idx, par, scomp_grad);

    if(!Rf_isReal(res) or !Rf_isVector(res) or Rf_xlength(res) != 1L)
      throw std::invalid_argument(
          "fn returns invalid output with comp_grad = FALSE");

    return Rf_asReal(res);
  }

  double grad
    (double const * __restrict__ point, double * __restrict__ gr) const {
    for(size_t j = 0; j < n_ele; ++j, ++point)
      par[j] = *point;
    scomp_grad[0] = true;
    SEXP res =  f(f_idx, par, scomp_grad);
    CharacterVector what("grad");
    SEXP gr_val = Rf_getAttrib(res, what);

    if(!Rf_isReal(res) or !Rf_isVector(res) or Rf_xlength(res) != 1L or
         Rf_isNull(gr_val) or !Rf_isReal(gr_val) or
         static_cast<size_t>(Rf_xlength(gr_val)) != n_ele)
      throw std::invalid_argument(
          "fn returns invalid output with comp_grad = TRUE");

    lp::copy(gr, REAL(gr_val), n_ele);
    return Rf_asReal(res);
  };

  virtual bool thread_safe() const {
    return false;
  };
};

List wrap_optim_info(NumericVector par_res, PSQN::optim_info res){
  NumericVector counts = NumericVector::create(
    res.n_eval, res.n_grad,  res.n_cg);
  counts.names() = CharacterVector::create("function", "gradient", "n_cg");

  int const info = static_cast<int>(res.info);
  return List::create(
    _["par"] = par_res, _["value"] = res.value, _["info"] = info,
      _["counts"] = counts, _["convergence"] = info >= 0L);
}

//' Partially Separable Function Optimization
//'
//' @description
//' Optimization for specially structured partially separable function.
//' See \code{vignette("psqn", package = "psqn")} for details.
//'
//' @param par Initial values for the parameters.
//' @param fn Function to compute the element functions and their
//' derivatives.
//' @param n_ele_func Number of element functions.
//' @param rel_eps Relative convergence threshold.
//' @param n_threads Number of threads to use.
//' @param max_it Maximum number of iterations.
//' @param c1,c2 Tresholds for the Wolfe condition.
//' @param use_bfgs Logical for whether to use BFGS updates or SR1 updates.
//' @param trace Integer where larger values gives more information during the
//' optimization.
//' @param cg_tol threshold for conjugate gradient method.
//' @param strong_wolfe \code{TRUE} if the strong Wolfe condition should be used.
//' @param env enviroment to evaluate fn in. \code{NULL} yields the global
//' enviroment.
//'
//' @export
// [[Rcpp::export]]
List psqn
  (NumericVector par, SEXP fn, unsigned const n_ele_func,
   double const rel_eps = .00000001,
   unsigned const max_it = 100L, unsigned const n_threads = 1L,
   double const c1 = .0001, double const c2 = .9,
   bool const use_bfgs = true, int const trace = 0L,
   double const cg_tol = .5, bool const strong_wolfe = true,
   SEXP env = R_NilValue){
  if(n_ele_func < 1L)
    throw std::invalid_argument("psqn: n_ele_func < 1L");

  if(Rf_isNull(env))
    env = Environment::global_env();
  if(!Rf_isEnvironment(env))
    throw std::invalid_argument("psqn: env is not an enviroment");
  if(!Rf_isFunction(fn))
    throw std::invalid_argument("psqn: fn is not a function");

  std::vector<r_worker> funcs;
  funcs.reserve(n_ele_func);
  for(size_t i = 0; i < n_ele_func; ++i)
    funcs.emplace_back(fn, i, env);

  PSQN::optimizer<r_worker, PSQN::R_reporter> optim(funcs, n_threads);

  // check that we pass a parameter value of the right length
  if(optim.n_par != static_cast<size_t>(par.size()))
    throw std::invalid_argument("psqn: invalid parameter size");

  NumericVector par_arg = clone(par);
  optim.set_n_threads(n_threads);
  auto res = optim.optim(&par_arg[0], rel_eps, max_it, c1, c2,
                         use_bfgs, trace, cg_tol, strong_wolfe);

  return wrap_optim_info(par_arg, res);
}


class r_worker_bfgs : public PSQN::problem {
  simple_R_func1 f, g;
  size_t const n_ele;
  NumericVector par = NumericVector(n_ele);

public:
  r_worker_bfgs(SEXP f, SEXP g, size_t const n_ele, SEXP env):
  f(f, env), g(g, env), n_ele(n_ele) { }

  size_t size() const {
    return n_ele;
  }

  double func(double const *val){
    lp::copy(&par[0], val, n_ele);
    SEXP res = f(par);
    if(!Rf_isReal(res) or !Rf_isVector(res) or Rf_xlength(res) != 1L)
      throw std::invalid_argument("fn returns invalid output");

    return Rf_asReal(res);
  }

  double grad(double * __restrict__ const val,
              double * __restrict__       gr){
    lp::copy(&par[0], val, n_ele);

    SEXP res = g(par);
    CharacterVector what("value");
    SEXP func_val = Rf_getAttrib(res, what);

    if(!Rf_isReal(res) or !Rf_isVector(res) or
         static_cast<size_t>(Rf_xlength(res)) != n_ele or
         Rf_isNull(func_val) or !Rf_isReal(func_val) or
         Rf_xlength(func_val) != 1L)
      throw std::invalid_argument("gr returns invalid output");

    lp::copy(gr, REAL(res), n_ele);
    return Rf_asReal(func_val);
  }
};

//' BFGS Implementation Used Internally in the psqn Package
//'
//' @inheritParams psqn
//' @param fn Function to evaluate the function to be minimized.
//' @param gr Gradient of \code{fn}. Should return the function value as an
//' attribute called \code{"value"}.
//' @export
//'
//' @examples
//' # declare function and gradient from the example from help(optim)
//' fn <- function(x) {
//'   x1 <- x[1]
//'   x2 <- x[2]
//'   100 * (x2 - x1 * x1)^2 + (1 - x1)^2
//' }
//' gr <- function(x) {
//'   x1 <- x[1]
//'   x2 <- x[2]
//'   c(-400 * x1 * (x2 - x1 * x1) - 2 * (1 - x1),
//'      200 *      (x2 - x1 * x1))
//' }
//' 
//' # we need a different function for the method in this package
//' gr_psqn <- function(x) {
//'   x1 <- x[1]
//'   x2 <- x[2]
//'   out <- c(-400 * x1 * (x2 - x1 * x1) - 2 * (1 - x1),
//'             200 *      (x2 - x1 * x1))
//'   attr(out, "value") <- 100 * (x2 - x1 * x1)^2 + (1 - x1)^2
//'   out
//' }
//' 
//' # we get the same
//' optim    (c(-1.2, 1), fn, gr, method = "BFGS")
//' psqn_bfgs(c(-1.2, 1), fn, gr_psqn)
//' 
//' # compare the computation time
//' system.time(replicate(1000,
//'                       optim    (c(-1.2, 1), fn, gr, method = "BFGS")))
//' system.time(replicate(1000,
//'                       psqn_bfgs(c(-1.2, 1), fn, gr_psqn)))
// [[Rcpp::export]]
List psqn_bfgs
  (NumericVector par, SEXP fn, SEXP gr,
   double const rel_eps = .00000001, size_t const max_it = 100,
   double const c1 = .0001, double const c2 = .9, int const trace = 0L,
   SEXP env = R_NilValue){
  if(Rf_isNull(env))
    env = Environment::global_env();
  if(!Rf_isEnvironment(env))
    throw std::invalid_argument("psqn_bfgs: env is not an enviroment");
  if(!Rf_isFunction(fn))
    throw std::invalid_argument("psqn_bfgs: fn is not a function");
  if(!Rf_isFunction(gr))
    throw std::invalid_argument("psqn_bfgs: gr is not a function");

  r_worker_bfgs problem(fn, gr, par.size(), env);

  NumericVector par_res = clone(par);
  auto const out = PSQN::bfgs<PSQN::R_reporter>
    (problem, &par_res[0], rel_eps, max_it, c1, c2, trace);

  return wrap_optim_info(par_res, out);
}

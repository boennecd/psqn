#include "psqn.h"
#include "psqn-reporter.h"

using namespace Rcpp;

class r_worker {
  Function f;
  IntegerVector f_idx;
  LogicalVector mutable scomp_grad = LogicalVector(1L);
  size_t const g_dim, p_dim,
               n_ele = g_dim + p_dim;

  NumericVector mutable par = NumericVector(g_dim + p_dim);

public:
  r_worker(Function func, int iarg):
  f(func),
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
    SEXP gr_val = Rf_getAttrib(res,  what);

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
//'
//' @export
// [[Rcpp::export]]
List psqn
  (NumericVector par, Function fn, unsigned const n_ele_func,
   double const rel_eps = .00000001, unsigned const max_it = 100L,
   unsigned const n_threads = 1L,
   double const c1 = .0001, double const c2 = .9,
   bool const use_bfgs = true, int const trace = 0L,
   double const cg_tol = .1){
  if(n_ele_func < 1L)
    throw std::invalid_argument("optim_mlogit: n_ele_func < 1L");

  std::vector<r_worker> funcs;
  funcs.reserve(n_ele_func);
  for(size_t i = 0; i < n_ele_func; ++i)
    funcs.emplace_back(fn, i);

  PSQN::optimizer<r_worker, PSQN::R_reporter> optim(funcs, n_threads);

  // check that we pass a parameter value of the right length
  if(optim.n_par != static_cast<size_t>(par.size()))
    throw std::invalid_argument("optim_mlogit: invalid parameter size");

  NumericVector par_arg = clone(par);
  optim.set_n_threads(n_threads);
  auto res = optim.optim(&par_arg[0], rel_eps, max_it, c1, c2,
                         use_bfgs, trace, cg_tol);
  NumericVector counts = NumericVector::create(
    res.n_eval, res.n_grad,  res.n_cg);
  counts.names() = CharacterVector::create("function", "gradient", "n_cg");

  return List::create(
    _["par"] = par_arg, _["value"] = res.value, _["info"] = res.info,
      _["counts"] = counts, _["convergence"] = res.info == 0L);
}

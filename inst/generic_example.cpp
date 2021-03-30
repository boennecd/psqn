// see `mlogit-ex.cpp` for an example with more comments

// we will use OpenMP to perform the computation in parallel
// [[Rcpp::plugins(openmp, cpp11)]]

// [[Rcpp::depends(psqn)]]
#include "psqn.h"
#include "psqn-reporter.h"

#include <Rcpp.h>

using namespace Rcpp;

class generic_example final : public PSQN::element_function_generic {
  /// number of argument to this element function;
  size_t const n_args_val;
  /// indices of the element function parameters
  std::unique_ptr<size_t[]> indices_array;
  /// y point
  double const y;

public:
  generic_example(List data):
  n_args_val(as<IntegerVector>(data["indices"]).size()),
  indices_array(([&]() -> std::unique_ptr<size_t[]> {
    IntegerVector indices = as<IntegerVector>(data["indices"]);
    std::unique_ptr<size_t[]> out(new size_t[n_args_val]);
    for(size_t i = 0; i < n_args_val; ++i)
      out[i] = indices[i];
    return out;
  })()),
  y(as<double>(data["y"]))
  { }

  // we need to make a copy constructor because of the unique_ptr
  generic_example(generic_example const &other):
  n_args_val(other.n_args_val),
  indices_array(([&]() -> std::unique_ptr<size_t[]> {
    std::unique_ptr<size_t[]> out(new size_t[n_args_val]);
    for(size_t i = 0; i < n_args_val; ++i)
      out[i] = other.indices_array[i];
    return out;
  })()),
  y(other.y) { }

  size_t n_args() const {
    return n_args_val;
  }

  size_t const * indices() const {
    return indices_array.get();
  }

  double func(double const *point) const {
    double sum(0.);
    for(size_t i = 0; i < n_args_val; ++i)
      sum += point[i];
    return -y * sum + std::exp(sum);
  }

  double grad
  (double const * point, double * gr) const {
    double sum(0.);
    for(size_t i = 0; i < n_args_val; ++i)
      sum += point[i];
    double const exp_sum = std::exp(sum),
                    fact = -y + exp_sum;
    for(size_t i = 0; i < n_args_val; ++i)
      gr[i] = fact;

    return -y * sum + std::exp(sum);
  }

  bool thread_safe() const {
    return true;
  }
};

using generic_opt =
  PSQN::optimizer_generic<generic_example, PSQN::R_reporter,
                          PSQN::R_interrupter>;

// [[Rcpp::export]]
SEXP get_generic_ex_obj(List data, unsigned const max_threads){
  size_t const n_elem_funcs = data.size();
  std::vector<generic_example> funcs;
  funcs.reserve(n_elem_funcs);
  for(auto dat : data)
    funcs.emplace_back(List(dat));

  // create an XPtr to the object we will need
  XPtr<generic_opt>ptr(new generic_opt(funcs, max_threads));

  // return the pointer to be used later
  return ptr;
}

// [[Rcpp::export]]
List optim_generic_ex
  (NumericVector val, SEXP ptr, double const rel_eps, unsigned const max_it,
   unsigned const n_threads, double const c1,
   double const c2, bool const use_bfgs = true, int const trace = 0L,
   double const cg_tol = .5, bool const strong_wolfe = true,
   size_t const max_cg = 0L, int const pre_method = 1L){
  XPtr<generic_opt> optim(ptr);

  // check that we pass a parameter value of the right length
  if(optim->n_par != static_cast<size_t>(val.size()))
    throw std::invalid_argument("optim_generic_ex: invalid parameter size");

  NumericVector par = clone(val);
  optim->set_n_threads(n_threads);
  auto res = optim->optim(&par[0], rel_eps, max_it, c1, c2,
                          use_bfgs, trace, cg_tol, strong_wolfe, max_cg,
                          static_cast<PSQN::precondition>(pre_method));
  NumericVector counts = NumericVector::create(
    res.n_eval, res.n_grad,  res.n_cg);
  counts.names() = CharacterVector::create("function", "gradient", "n_cg");

  int const info = static_cast<int>(res.info);
  return List::create(
    _["par"] = par, _["value"] = res.value, _["info"] = info,
      _["counts"] = counts,
      _["convergence"] =  res.info == PSQN::info_code::converged);
}

// [[Rcpp::export]]
double eval_generic_ex(NumericVector val, SEXP ptr, unsigned const n_threads){
  XPtr<generic_opt> optim(ptr);

  // check that we pass a parameter value of the right length
  if(optim->n_par != static_cast<size_t>(val.size()))
    throw std::invalid_argument("eval_generic_ex: invalid parameter size");

  optim->set_n_threads(n_threads);
  return optim->eval(&val[0], nullptr, false);
}

// [[Rcpp::export]]
NumericVector grad_generic_ex(NumericVector val, SEXP ptr,
                              unsigned const n_threads){
  XPtr<generic_opt> optim(ptr);

  // check that we pass a parameter value of the right length
  if(optim->n_par != static_cast<size_t>(val.size()))
    throw std::invalid_argument("grad_generic_ex: invalid parameter size");

  NumericVector grad(val.size());
  optim->set_n_threads(n_threads);
  grad.attr("value") = optim->eval(&val[0], &grad[0], true);

  return grad;
}

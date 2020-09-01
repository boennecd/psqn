// [[Rcpp::depends(psqn)]]
#include "psqn.h"

// we use RcppArmadillo to simplify the code
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;

/// simple function to avoid copying a vector. You can ignore this.
inline arma::vec vec_no_cp(double const * x, size_t const n_ele){
  return arma::vec(const_cast<double *>(x), n_ele, false);
}

/***
 implements the element function for a given cluster. The class must provide
 the member functions which we provide here.

 We do not need to inherit from the element_function class but we can do it
 to ensure that we have implemented all the member functions.
 */
class m_logit_func final : public PSQN::element_function {
  /// design matrices
  arma::mat const X, Z;
  /// outcomes
  arma::vec const y;
  /// inverse covariance matrix
  arma::mat const Sigma_inv;

public:
  m_logit_func(List data):
  X        (as<arma::mat>(data["X"        ])),
  Z        (as<arma::mat>(data["Z"        ])),
  y        (as<arma::vec>(data["y"        ])),
  Sigma_inv(as<arma::mat>(data["Sigma_inv"])) { }

  /// dimension of the global parameters
  size_t shared_dim() const {
    return X.n_rows;
  }
  /// dimension of the private parameters
  size_t own_dim() const {
    return Z.n_rows;
  }

  /***
   computes the element function.
   @param point point to compute function at.
   */
  double func(double const *point) const {
    arma::vec const beta = vec_no_cp(point           , X.n_rows),
                       u = vec_no_cp(point + X.n_rows, Z.n_rows),
                     eta = X.t() * beta + Z.t() * u;

    double out(0);
    for(size_t i = 0; i < y.n_elem; ++i)
      out -= y[i] * eta[i] - log(1 + exp(eta[i]));

    out += arma::as_scalar(u.t() * Sigma_inv * u) * .5;

    return out;
  }

  /***
   computes the element function and its gradient.
   @param point point to compute function at.
   @param gr gradient vector with respect to global and private parameters.
   */
  double grad
    (double const * __restrict__ point, double * __restrict__ gr) const {
    arma::vec const beta = vec_no_cp(point           , X.n_rows),
                       u = vec_no_cp(point + X.n_rows, Z.n_rows),
                     eta = X.t() * beta + Z.t() * u;

    // create objects to write to for the gradient
    std::fill(gr, gr + beta.n_elem + u.n_elem, 0.);
    arma::vec dbeta(gr              , beta.n_elem, false),
              du   (gr + beta.n_elem, u.n_elem   , false);

    double out(0);
    for(size_t i = 0; i < y.n_elem; ++i){
      double const exp_eta = exp(eta[i]),
                   d_eta   = y[i] - exp_eta / (1 + exp_eta);
      out -= y[i] * eta[i] - log(1 + exp_eta);
      dbeta -= d_eta * X.col(i);
      du    -= d_eta * Z.col(i);
    }
    du += Sigma_inv * u;

    return out;
  }
};

/***
 creates a pointer to an object which can.
 @param data list with data for each element function.
 */
// [[Rcpp::export]]
SEXP get_mlogit_optimizer(List data){
  size_t const n_elem_funcs = data.size();
  std::vector<m_logit_func> funcs;
  funcs.reserve(n_elem_funcs);
  for(auto dat : data)
    funcs.emplace_back(List(dat));

  // create an XPtr to the pointer object we will need
  XPtr<PSQN::optimizer<m_logit_func> >
    ptr(new PSQN::optimizer<m_logit_func>(funcs));

  // return the pointer to be used later
  return ptr;
}

/***
 evaluates the partially separable function.
 @param val vector with global and private parameters to evaluate the
 function at.
 @param ptr returned object from get_mlogit_optimizer.
 */
// [[Rcpp::export]]
double eval_mlogit(NumericVector val, SEXP ptr){
  XPtr<PSQN::optimizer<m_logit_func> > optim(ptr);

  // check that we pass a parameter value of the right length.
  if(optim->n_par != static_cast<size_t>(val.size()))
    throw std::invalid_argument("eval_mlogit: invalid parameter size");

  return optim->eval(&val[0], nullptr, false);
}

/***
 evaluates the gradient of a partially separable function.
 @param val vector with global and private parameters to evaluate the
 function at.
 @param ptr returned object from get_mlogit_optimizer.
 */
// [[Rcpp::export]]
NumericVector grad_mlogit(NumericVector val, SEXP ptr){
  XPtr<PSQN::optimizer<m_logit_func> > optim(ptr);

  // check that we pass a parameter value of the right length.
  if(optim->n_par != static_cast<size_t>(val.size()))
    throw std::invalid_argument("eval_mlogit: invalid parameter size");

  NumericVector grad(val.size());
  optim->eval(&val[0], &grad[0], true);
  return grad;
}

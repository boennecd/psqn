#ifndef PSQN_H
#define PSQN_H
#include <vector>
#include <memory>
#include "lp.h"
#include <algorithm>

namespace PSQN {
/***
 virtual base class which computes an element function and its gradient. The
 virtual class is mainly used as a check to ensure that all the member
 fucntions are implemented.
 */
class element_function {
public:
  /// dimension of the global parameters
  virtual size_t shared_dim() const = 0;
  /// dimension of the private parameters
  virtual size_t own_dim() const = 0;

  /***
   computes the element function.
   @param point point to compute function at.
   */
  virtual double func(double const *point) const = 0;

  /***
   computes the element function and its gradient.
   @param point point to compute function at.
   @param gr gradient vector with respect to global and private parameters.
   */
  virtual double grad
    (double const * __restrict__ point, double * __restrict__ gr)
    const = 0;

  virtual ~element_function() = default;
};

template<class EFunc>
class optimizer {
  /***
   worker class to hold an element function and the element function''s
   Hessian approximation.
   */
  class worker {
    /// logical for whether it the first call
    bool first_call = true;

  public:
    /// the element function for this worker
    EFunc const func;
    /// number of elements
    size_t const n_ele = func.shared_dim() + func.own_dim();
    /// memory for the Hessian approximation
    double * const __restrict__ B;
    /// memory for the gradient
    double * const __restrict__ gr = B + n_ele * n_ele;
    /// memory for the old gradient
    double * const __restrict__ gr_old = gr + n_ele;
    /// memory for the old value
    double * const __restrict__ x_old = gr_old + n_ele;
    /// memory for the current value
    double * const __restrict__ x_new = x_old + n_ele;
    /// indices of first set of own parameters
    size_t const par_start;

    worker(EFunc &&func, double * mem, size_t const par_start):
      func(func), B(mem), par_start(par_start) {
      std::fill(B, B + n_ele * n_ele, 0.);
      // set diagonal entries to one
      double *b = B;
      for(size_t i = 0; i < n_ele; ++i, b += n_ele + 1)
        *b = 1.;
    }

    /***
     computes the element function and possibly its gradient.

     @param shared values for the shared parameters
     @param own values for own parameters.
     @param comp_grad logical for whether to compute the gradient
     */
    double operator()
      (double const * __restrict__ shared, double const * __restrict__ own,
       bool const comp_grad){
      // copy values
      size_t const d_shared = func.shared_dim(),
                   d_own    = func.own_dim();

      lp::copy(x_new           , shared, d_shared);
      lp::copy(x_new + d_shared, own   , d_own);

      if(!comp_grad)
        return func.func(x_new);

      return func.grad(x_new, gr);
    }

    /***
     save the current parameter values and gradient in order to do update
     the Hessian approximation.
     */
    void record() noexcept {
      lp::copy(x_old , static_cast<double const*>(x_new), n_ele);
      lp::copy(gr_old, static_cast<double const*>(gr   ), n_ele);
    }

    /***
     updates the Hessian approximation. Assumes that the () operator and
     record have been called.
     */
    void update_Hes(){
      // differences in parameters and gradient
      // TODO: avoid the memory allocation
      std::unique_ptr<double[]> s(new double[n_ele]),
                                y(new double[n_ele]),
                              wrk(new double[n_ele]);

      lp::vec_diff(x_new, x_old , s.get(), n_ele);
      lp::vec_diff(gr   , gr_old, y.get(), n_ele);

      double const s_y = lp::vec_dot(y.get(), s.get(), n_ele);
      if(first_call){
        first_call = false;
        // make update as on page 143
        double const scal = s_y / lp::vec_dot(y.get(), n_ele);
        double *b = B;
        for(size_t i = 0; i < n_ele * n_ele; ++i, ++b)
          *b *= scal;
      }

      // perform BFGS update
      std::fill(wrk.get(), wrk.get() + n_ele, 0.);
      lp::mat_vec_dot(B, s.get(), wrk.get(), 1.);
      double const scal = -lp::vec_dot(s.get(), wrk.get(), n_ele);
      lp::rank_one_update(B, wrk.get(), scal, n_ele);
      lp::rank_one_update(B, y.get(), s_y, n_ele);
    }
  };

public:
  /// dimension of the shared parameters
  size_t const shared_dim;
  /// total number of parameters
  size_t const n_par;

private:
  /// size of the allocated working memory
  size_t const n_mem;
  /// working memory
  std::unique_ptr<double[]> mem =
    std::unique_ptr<double[]>(new double[n_mem]);
  /// element functions
  std::vector<worker> funcs;

public:
  /***
   takes in a vector with element functions and constructs the optimizer.
   The members are moved out of the vector.
   */
  optimizer(std::vector<EFunc> &funcs_in):
  shared_dim(([&](){
    if(funcs_in.size() < 1L)
      throw std::invalid_argument(
          "optimizer<EFunc>::optimizer: no functions supplied");
    return funcs_in[0].shared_dim();
  })()),
  n_par(([&](){
    size_t out(shared_dim);
    for(auto &f : funcs_in)
      out += f.own_dim();
    return out;
  })()),
  n_mem(([&](){
    size_t out(0L);
    for(auto &f : funcs_in){
      if(f.shared_dim() != shared_dim)
        throw std::invalid_argument(
            "optimizer<EFunc>::optimizer: shared_dim differs");
      size_t const own_dim = f.own_dim();
      out += (own_dim + shared_dim) * (4L + (own_dim + shared_dim));
    }
    return out;
  })()),
  funcs(([&](){
    std::vector<worker> out;
    size_t const n_ele(funcs_in.size());
    out.reserve(funcs_in.size());

    double * mem_ptr = mem.get();
    size_t i_start(shared_dim);
    for(size_t i = 0; i < n_ele; ++i){
      worker new_func(std::move(funcs_in[i]), mem_ptr, i_start);
      out.emplace_back(std::move(new_func));
      mem_ptr += out.back().n_ele * (4L + out.back().n_ele);
      i_start += out.back().func.own_dim();
    }

    return out;
  })()) { }


  /***
   evalutes the partially separable and also the gradient if requested.
   @param val pointer to the value to evaluate the function at.
   @param gr pointer to store gradient in.
   @param comp_grad boolean for whether to compute the gradient.
   */
  double eval(double const * val, double * __restrict__ gr,
              bool const comp_grad){
    double out(0.);
    size_t const n_funcs = funcs.size();
    for(size_t i = 0; i < n_funcs; ++i){
      auto &f = funcs[i];
      out += f(val, val + f.par_start, comp_grad);
    }

    if(comp_grad){
      std::fill(gr, gr + shared_dim, 0.);
      for(size_t i = 0; i < n_funcs; ++i){
        auto const &f = funcs[i];
        for(size_t j = 0; j < shared_dim; ++j)
          *(gr + j) += *(f.gr + j);

        size_t const own = f.func.own_dim();
        lp::copy(gr + f.par_start, f.gr + shared_dim, own);
      }
    }

    return out;
  }
};

} // namespace PSQN

#endif

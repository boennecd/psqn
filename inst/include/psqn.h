#ifndef PSQN_H
#define PSQN_H
#include <vector>
#include <array>
#include <memory>
#include "lp.h"
#include <algorithm>
#include <limits>
#include "constant.h"
#include <cmath>
#include "intrapolate.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace PSQN {
using std::abs;
using std::sqrt;

/***
 virtual base class which computes an element function and its gradient. The
 virtual class is mainly used as a check to ensure that all the member
 fucntions are implemented.
 */
class element_function {
public:
  /// dimension of the global parameters
  virtual size_t global_dim() const = 0;
  /// dimension of the private parameters
  virtual size_t private_dim() const = 0;

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

  /***
   returns true if the member functions are thread-safe.
   */
  virtual bool thread_safe() const = 0;

  virtual ~element_function() = default;
};

struct dummy_reporter {
  /**
   reporting after conjugate gradient method.

   @param trace info level passed by the user.
   @param iteration itteration number starting at zero.
   @param n_cg number of conjugate gradient iterations.
   @param successful true of conjugate gradient method succeeded.
   */
  static void cg(int const trace, size_t const iteration,
                 size_t const n_cg, bool const successful) { }

  /**
   reporting during line search.

   @param trace info level passed by the user.
   @param a_old lower value in zoom or previous value if not zoom.
   @param a_new new value to test.
   @param f_new function value at a_new.
   @param is_zoom true if the call is from the zoom function.
   @param d_new derivative at a_new.
   @param a_high upper value in zoom.
   */
  static void line_search_inner
  (int const trace, double const a_old, double const a_new,
   double const f_new, bool const is_zoom, double const d_new,
   double const a_high) { }

  /**
   reporting after line search.

   @param trace info level passed by the user.
   @param iteration itteration number starting at zero.
   @param n_eval number of function evaluations.
   @param n_grad number of gradient evaluations.
   @param fval_old old function value.
   @param fval new function value.
   @param successful true of conjugate gradient method succeeded.
   @param step_size found step size.
   @param new_x new parameter value.
   @param n_global number of global parameters.
   */
  static void line_search
  (int const trace, size_t const iteration, size_t const n_eval,
   size_t const n_grad, double const fval_old,
   double const fval, bool const successful, double const step_size,
   double const *new_x, size_t const n_global) { }
};

/***
 the reporter class can be used to report results during the optimization.
 */
template<class EFunc, class Reporter = dummy_reporter>
class optimizer {
  /***
   worker class to hold an element function and the element function''s
   Hessian approximation.
   */
  class worker {
    /// logical for whether it the first call
    bool first_call = true;
    /// logical for whether this is the first gradient evaluation
    bool first_grad = true;

  public:
    /// the element function for this worker
    EFunc const func;
    /// number of elements
    size_t const n_ele = func.global_dim() + func.private_dim();
    /// memory for the Hessian approximation
    double * const __restrict__ B;
    /// memory for the gradient
    double * const __restrict__ gr = B + (n_ele * (n_ele + 1)) / 2L;
    /// memory for the old gradient
    double * const __restrict__ gr_old = gr + n_ele;
    /// memory for the old value
    double * const __restrict__ x_old = gr_old + n_ele;
    /// memory for the current value
    double * const __restrict__ x_new = x_old + n_ele;
    /// indices of first set of private parameters
    size_t const par_start;
    /// bool for whether to use BFGS or SR1 updates
    bool use_bfgs = true;

  private:
    /***
     save the current parameter values and gradient in order to do update
     the Hessian approximation.
     */
    void record() noexcept {
      lp::copy(x_old , static_cast<double const*>(x_new), n_ele);
      lp::copy(gr_old, static_cast<double const*>(gr   ), n_ele);
    }

  public:
    /***
     resets the Hessian approximation.
     */
    void reset() noexcept {
      first_call = true;
      first_grad = true;

      std::fill(B, B + n_ele * n_ele, 0.);
      // set diagonal entries to one
      double *b = B;
      for(size_t i = 0; i < n_ele; ++i, b += i + 1)
        *b = 1.;
    }

    worker(EFunc &&func, double * mem, size_t const par_start):
      func(func), B(mem), par_start(par_start) {
      reset();
    }

    /***
     computes the element function and possibly its gradient.

     @param global values for the global parameters
     @param vprivate values for private parameters.
     @param comp_grad logical for whether to compute the gradient
     */
    double operator()
      (double const * __restrict__ global,
       double const * __restrict__ vprivate, bool const comp_grad){
      // copy values
      size_t const d_global  = func.global_dim(),
                   d_private = func.private_dim();

      lp::copy(x_new           , global  , d_global);
      lp::copy(x_new + d_global, vprivate, d_private);

      if(!comp_grad)
        return func.func(static_cast<double const *>(x_new));

      double const out =  func.grad(
        static_cast<double const *>(x_new), gr);

      if(!first_grad)
        return out;

      first_grad = false;
      record();
      return out;
    }

    /***
     updates the Hessian approximation. Assumes that the () operator have
     been called at-least twice at different values.
     @param wmem working memory to use.
     */
    void update_Hes(double * const wmem){
      // differences in parameters and gradient
      double * const __restrict__ s   = wmem,
             * const __restrict__ y   = s + n_ele,
             * const __restrict__ wrk = y + n_ele;

      lp::vec_diff(x_new, x_old , s, n_ele);
      lp::vec_diff(gr   , gr_old, y, n_ele);

      if(use_bfgs){
        double const s_y = lp::vec_dot(y, s, n_ele);
        if(first_call){
          first_call = false;
          // make update on page 143
          double const scal = lp::vec_dot(y, n_ele) / s_y;
          double *b = B;
          for(size_t i = 0; i < n_ele; ++i, b += i + 1)
            *b = scal;
        }

        // perform BFGS update
        std::fill(wrk, wrk + n_ele, 0.);
        lp::mat_vec_dot(B, s, wrk, n_ele);
        double const s_B_s = lp::vec_dot(s, wrk, n_ele);

        lp::rank_one_update(B, wrk, -1. / s_B_s, n_ele);

        if(s_y < .2 * s_B_s){
          // damped BFGS
          double const theta = .8 * s_B_s / (s_B_s - s_y);
          double *yi = y,
                 *wi = wrk;
          for(size_t i = 0; i < n_ele; ++i, ++yi, ++wi)
            *yi = theta * *yi + (1 - theta) * *wi;
          double const s_r = lp::vec_dot(y, s, n_ele);
          lp::rank_one_update(B, y, 1. / s_r, n_ele);

        } else
          // regular BFGS
          lp::rank_one_update(B, y, 1. / s_y, n_ele);

      } else {
        if(first_call){
          first_call = false;
          // make update on page 143
          double const scal =
            lp::vec_dot(y, n_ele) / lp::vec_dot(y, s, n_ele);
          double *b = B;
          for(size_t i = 0; i < n_ele; ++i, b += i + 1)
            *b = scal;
        }

        /// maybe perform SR1
        std::fill(wrk, wrk + n_ele, 0.);
        lp::mat_vec_dot(B, s, wrk, n_ele);
        for(size_t i = 0; i < n_ele; ++i){
          *(wrk + i) *= -1;
          *(wrk + i) += *(y + i);
        }
        double const s_w = lp::vec_dot(s, wrk, n_ele),
                  s_norm = sqrt(abs(lp::vec_dot(s, n_ele))),
                wrk_norm = sqrt(abs(lp::vec_dot(wrk, n_ele)));
        constexpr double const r = 1e-8;
        if(abs(s_w) > r * s_norm * wrk_norm)
          lp::rank_one_update(B, wrk, 1. / s_w, n_ele);

      }

      record();
    }
  };

public:
  /// dimension of the global parameters
  size_t const global_dim;
  /// true if the element functions are thread-safe
  bool const is_ele_func_thread_safe;
  /// total number of parameters
  size_t const n_par;

private:
  /***
   size of the allocated working memory. The first element is needed for
   the workers. The second element is needed during the computation for the
   master thread. The third element is number required per thread.
  */
  std::array<size_t, 3L> const n_mem;
  /// maximum number of threads to use
  size_t const max_threads;
  /// working memory
  std::unique_ptr<double[]> mem =
    std::unique_ptr<double[]>(
        new double[n_mem[0] + n_mem[1] + max_threads * n_mem[2]]);
  /// pointer to temporary memory to use on the master thread
  double * const temp_mem = mem.get() + n_mem[0];
  /// pointer to temporray memory to be used by the threads
  double * const temp_thread_mem = temp_mem + n_mem[1];
  /// element functions
  std::vector<worker> funcs;
  /// number of function evaluations
  size_t n_eval = 0L;
  /// number of gradient evaluations
  size_t n_grad = 0L;
  /// number of iterations of conjugate gradient
  size_t n_cg = 0L;

  /***
   reset the counters for the number of evaluations
   */
  void reset_counters() {
    n_eval = 0L;
    n_grad = 0L;
    n_cg = 0L;
  }

  /// returns the thread number.
  int get_thread_num() const noexcept {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 1L;
#endif
  }

  /// returns working memory for this thread
  double * get_thread_mem() const noexcept {
    return temp_thread_mem + get_thread_num() * n_mem[2];
  }

  /// number of threads to use
  int n_threads = 1L;

public:
  /// set the number of threads to use
  void set_n_threads(size_t const n_threads_new) noexcept {
#ifdef _OPENMP
    n_threads = std::max(
      static_cast<size_t>(1L), std::min(n_threads_new, max_threads));
    omp_set_num_threads(n_threads);
    omp_set_dynamic(0L);
#endif
  }

  /***
   takes in a vector with element functions and constructs the optimizer.
   The members are moved out of the vector.
   @param funcs_in vector with element functions.
   @param max_threads maximum number of threads to use.
   */
  optimizer(std::vector<EFunc> &funcs_in, size_t const max_threads):
  global_dim(([&](){
    if(funcs_in.size() < 1L)
      throw std::invalid_argument(
          "optimizer<EFunc>::optimizer: no functions supplied");
    return funcs_in[0].global_dim();
  })()),
  is_ele_func_thread_safe(funcs_in[0].thread_safe()),
  n_par(([&](){
    size_t out(global_dim);
    for(auto &f : funcs_in)
      out += f.private_dim();
    return out;
  })()),
  n_mem(([&](){
    size_t out(0L),
           max_priv(0L);
    for(auto &f : funcs_in){
      if(f.global_dim() != global_dim)
        throw std::invalid_argument(
            "optimizer<EFunc>::optimizer: global_dim differs");
      if(f.thread_safe() != is_ele_func_thread_safe)
        throw std::invalid_argument(
            "optimizer<EFunc>::optimizer: thread_safe differs");
      size_t const private_dim = f.private_dim(),
                   n_ele       = private_dim + global_dim;
      if(max_priv < private_dim)
        max_priv = private_dim;

      out += n_ele * 4L + (n_ele * (n_ele + 1L)) / 2L;
    }

    constexpr size_t const mult = cacheline_size() / sizeof(double),
                       min_size = 2L * mult;

    size_t thread_mem = std::max(global_dim + max_priv, min_size);
    thread_mem = std::max(thread_mem, 2L * global_dim);
    thread_mem = (thread_mem + mult - 1L) / mult;
    thread_mem *= mult;

    std::array<size_t, 3L> ret = { out, 3L * n_par, thread_mem };
    return ret;
  })()),
  max_threads(max_threads > 0 ? max_threads : 1L),
  funcs(([&](){
    std::vector<worker> out;
    size_t const n_ele(funcs_in.size());
    out.reserve(funcs_in.size());

    double * mem_ptr = mem.get();
    size_t i_start(global_dim);
    for(size_t i = 0; i < n_ele; ++i){
      worker new_func(std::move(funcs_in[i]), mem_ptr, i_start);
      out.emplace_back(std::move(new_func));
      size_t const n_ele = out.back().n_ele;
      mem_ptr += n_ele * 4L + (n_ele * (n_ele + 1L)) / 2L;
      i_start += out.back().func.private_dim();
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
    if(comp_grad)
      n_grad++;
    else
      n_eval++;

    size_t const n_funcs = funcs.size();
    auto serial_version = [&](){
      double out(0.);
      for(size_t i = 0; i < n_funcs; ++i){
        auto &f = funcs[i];
        out += f(val, val + f.par_start, comp_grad);
      }

      if(comp_grad){
        std::fill(gr, gr + global_dim, 0.);
        for(size_t i = 0; i < n_funcs; ++i){
          auto const &f = funcs[i];
          for(size_t j = 0; j < global_dim; ++j)
            *(gr + j) += *(f.gr + j);

          size_t const iprivate = f.func.private_dim();
          lp::copy(gr + f.par_start, f.gr + global_dim, iprivate);
        }
      }

      return out;
    };

    if(n_threads < 2L or !is_ele_func_thread_safe)
      return serial_version();

#ifdef _OPENMP
    double out(0.);
#pragma omp parallel num_threads(n_threads)
  {
    double * v_mem = get_thread_mem(),
           * r_mem = v_mem + global_dim;
    lp::copy(v_mem, val, global_dim);
    if(comp_grad)
      std::fill(r_mem, r_mem + global_dim, 0.);

    double thread_terms(0.);
#pragma omp for schedule(static) nowait
    for(size_t i = 0; i < n_funcs; ++i){
      auto &f = funcs[i];
      thread_terms += f(v_mem, val + f.par_start, comp_grad);

      if(comp_grad){
        // update global
        double *lhs = r_mem;
        double const *rhs = f.gr;
        for(size_t j = 0; j < global_dim; ++j, ++lhs, ++rhs)
          *lhs += *rhs;

        // update private
        lp::copy(gr + f.par_start, f.gr + global_dim, f.func.private_dim());
      }
    }

    if(comp_grad)
#pragma omp single
      std::fill(gr, gr + global_dim, 0.);

    // add to global parameters
#pragma omp for ordered schedule(static, 1)
    for (int t = 0; t < omp_get_num_threads(); t++){
#pragma omp ordered
      {
        out += thread_terms;
        if(comp_grad)
          for(size_t i = 0; i < global_dim; ++i)
            *(gr + i) += *(r_mem + i);
      }
    }
  }

    return out;
#else
    return serial_version();
#endif
  }

  /***
   computes y <- y + B.x where B is the current Hessian approximation.
   @param val vector on the right-hand side.
   @param res output vector on the left-hand side.
   ***/
  void B_vec(double const * const __restrict__ val,
             double * const __restrict__ res) const noexcept {
    size_t const n_funcs = funcs.size();

    // the serial version
    auto serial_version = [&](){
      for(size_t i = 0; i < n_funcs; ++i){
        auto &f = funcs[i];
        size_t const iprivate = f.func.private_dim(),
          private_offset = f.par_start;

        lp::mat_vec_dot(f.B, val, val + private_offset, res,
                        res + private_offset, global_dim, iprivate);
      }
    };

    if(n_threads < 2L){
      serial_version();
      return;
    }

#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
    {
    double * v_mem = get_thread_mem(),
           * r_mem = v_mem + global_dim;
    lp::copy(v_mem, val, global_dim);
    std::fill(r_mem, r_mem + global_dim, 0.);

#pragma omp for schedule(static) nowait
    for(size_t i = 0; i < n_funcs; ++i){
      auto &f = funcs[i];
      size_t const iprivate = f.func.private_dim(),
             private_offset = f.par_start;

      lp::mat_vec_dot(f.B, v_mem, val + private_offset, r_mem,
                      res + private_offset, global_dim, iprivate);
    }

    // add to global parameters
#pragma omp for ordered schedule(static, 1)
    for (int t = 0; t < omp_get_num_threads(); t++){
#pragma omp ordered
      {
        for(size_t i = 0; i < global_dim; ++i)
          *(res + i) += *(r_mem + i);
      }
    }
    }
#else
    serial_version();
#endif
  }

  /***
    conjugate gradient method. Solves B.y = x where B is the Hessian
    approximation.
    @param tol convergence threshold.
   */
  bool conj_grad(double const * __restrict__ x, double * __restrict__ y,
                 double const tol){
    double * __restrict__ r   = temp_mem,
           * __restrict__ p   = r + n_par,
           * __restrict__ B_p = p + n_par;

    // setup before first iteration
    std::fill(y, y + n_par, 0.);
    for(size_t i = 0; i < n_par; ++i){
      *(r + i) = -*(x + i);
      *(p + i) =  *(x + i);
    }

    double old_r_dot = lp::vec_dot(r, n_par);
    for(size_t i = 0; i < n_par; ++i){
      ++n_cg;
      std::fill(B_p, B_p + n_par, 0.);
      B_vec(p, B_p);

      double const p_B_p = lp::vec_dot(p, B_p, n_par);
      if(p_B_p <= 0){
        // negative curvature. Thus, exit
        if(i < 1L)
          // set output to be the gradient
          for(size_t j = 0; j < n_par; ++j)
            *(y + j) = *(x + j);

        break;
      }
      double const alpha = old_r_dot / p_B_p;

      for(size_t j = 0; j < n_par; ++j){
        *(y + j) += alpha * *(p   + j);
        *(r + j) += alpha * *(B_p + j);
      }

      double const r_dot = lp::vec_dot(r, n_par);
      if(sqrt(abs(r_dot)) < tol)
        break;

      double const beta = r_dot / old_r_dot;
      old_r_dot = r_dot;
      for(size_t j = 0; j < n_par; ++j){
        *(p + j) *= beta;
        *(p + j) -= *(r + j);
      }
    }

    return true;
  }

  /***
   performs line search to satisfy the Wolfe condition.
   @param f0 value of the functions at the current value.
   @param x0 value the function is evaluted.
   @param gr0 value of the current gradient.
   @param dir direction to search in.
   @param fnew the function value at the found solution.
   @param c1,c2 tresholds for Wolfe condition.
   @param strong_wolfe true if the strong Wolfe condition should be used.

   x0 and gr0 contains the new value and gradient on return. The method
   returns false if the line search fails.
   */
  bool line_search(
      double const f0, double * __restrict__ x0, double * __restrict__ gr0,
      double * __restrict__ dir, double &fnew, double const c1,
      double const c2, bool const strong_wolfe, int const trace){
    double * const x_mem = temp_mem;

    // declare 1D functions
    auto psi = [&](double const alpha){
      for(size_t i = 0; i < n_par; ++i)
        *(x_mem + i) = *(x0 + i) + alpha * *(dir + i);

      return eval(x_mem, nullptr, false);
    };

    // returns the function value and the gradient
    auto dpsi = [&](double const alpha){
      for(size_t i = 0; i < n_par; ++i)
        *(x_mem + i) = *(x0 + i) + alpha * *(dir + i);

      fnew = eval(x_mem, gr0, true);
      return lp::vec_dot(gr0, dir, n_par);
    };

    // the above at alpha = 0
    double const dpsi_zero = lp::vec_dot(gr0, dir, n_par);
    if(dpsi_zero > 0){
      // not a descent direction! Go the other way
      for(double * d = dir; d != dir + n_par; ++d)
        *d *= -1;
      return line_search(f0, x0, gr0, dir, fnew, c1, c2, strong_wolfe,
                         trace);
    }

    constexpr double const NaNv  = std::numeric_limits<double>::quiet_NaN();

    auto zoom = [&](double a_low, double a_high, intrapolate &inter){
      double f_low = psi(a_low);
      for(size_t i = 0; i < 25L; ++i){
        double const ai = inter.get_value(a_low, a_high),
                     fi = psi(ai);
        inter.update(ai, fi);
        Reporter::line_search_inner(trace, a_low, ai, fi, true,
                                    NaNv, a_high);

        if(fi > f0 + c1 * ai * dpsi_zero or fi >= f_low){
          a_high = ai;
          continue;
        }

        double const dpsi_i = dpsi(ai);
        Reporter::line_search_inner(trace, a_low, ai, fi, true,
                                    dpsi_i, a_high);
        double const test_val = strong_wolfe ? abs(dpsi_i) : -dpsi_i;
        if(test_val <= - c2 * dpsi_zero)
          return true;

        if(dpsi_i * (a_high - a_low) >= 0.)
          a_high = a_low;

        a_low = ai;
        f_low = fi;
      }

      return false;
    };

    double fold(f0),
         a_prev(0),
             ai(.5);
    bool found_ok_prev = false;
    for(size_t i = 0; i < 25L; ++i){
      ai *= 2;
      double const fi = psi(ai);
      Reporter::line_search_inner(trace, a_prev, ai, fi, false,
                                  NaNv, NaNv);
      if(!std::isfinite(fi)){
        // handle inf/nan case
        ai /= 4;
        continue;
      }

      if(fi > f0 + c1 * ai * dpsi_zero or (found_ok_prev and fi > fold)){
        intrapolate inter(f0, dpsi_zero, ai, fi);
        bool const out = zoom(a_prev, ai, inter);
        lp::copy(x0, x_mem, n_par);
        return out;
      }

      double const dpsi_i = dpsi(ai);
      Reporter::line_search_inner(trace, a_prev, ai, fi, false,
                                  dpsi_i, NaNv);

      double const test_val = strong_wolfe ? abs(dpsi_i) : -dpsi_i;
      if(test_val <= - c2 * dpsi_zero){
        lp::copy(x0, x_mem, n_par);
        return true;
      }

      if(dpsi_i >= 0){
        intrapolate inter = ([&](){
          if(found_ok_prev){
            // we have two values that we can use
            intrapolate out(f0, dpsi_zero, a_prev, fold);
            out.update(ai, fi);
            return out;
          }

          return intrapolate(f0, dpsi_zero, ai, fi);
        })();
        bool const out = zoom(ai, a_prev, inter);
        lp::copy(x0, x_mem, n_par);
        return out;
      }

      found_ok_prev = true;
      a_prev = ai;
      fold = fi;
    }

    return false;
  }

  struct optim_info {
    double value;
    int info;
    size_t n_eval, n_grad, n_cg;
  };


  /***
   optimizes the partially separable function.
   @param val pointer to starting value. Set to the final estimate at the
   end.
   @param rel_eps relative convergence threshold.
   @param max_it maximum number of iterations.
   @param c1,c2 tresholds for Wolfe condition.
   @param use_bfgs bool for whether to use BFGS updates or SR1 updates.
   @param trace integer with info level passed to reporter.
   @param cg_tol threshold for conjugate gradient method.
   @param strong_wolfe true if the strong Wolfe condition should be used.
   */
  optim_info optim
    (double * val, double const rel_eps, size_t const max_it,
     double const c1, double const c2,
     bool const use_bfgs = true, int const trace = 0,
     double const cg_tol = .5, bool const strong_wolfe = true){
    reset_counters();
    for(auto &f : funcs){
      f.reset();
      f.use_bfgs = use_bfgs;
    }

    std::unique_ptr<double[]> gr(new double[n_par]),
                             dir(new double[n_par]);

    // evaluate the gradient at the current value
    double fval = eval(val, gr.get(), true);

    int info = -1L;
    for(size_t i = 0; i < max_it; ++i){
      double const fval_old = fval,
                     gr_nom = sqrt(abs(lp::vec_dot(gr.get(), n_par))),
                 cg_tol_use = std::min(cg_tol, gr_nom) * gr_nom;
      if(!conj_grad(gr.get(), dir.get(), cg_tol_use)){
        info = -2L;
        Reporter::cg(trace, i, n_cg, false);
        break;
      }

      Reporter::cg(trace, i, n_cg, true);
      for(double * d = dir.get(); d != dir.get() + n_par; ++d)
        *d *= -1;

      double const x1 = *val;
      if(!line_search(fval_old, val, gr.get(), dir.get(), fval, c1, c2,
                      strong_wolfe, trace)){
        info = -3L;
        Reporter::line_search
          (trace, i, n_eval, n_grad, fval_old, fval, false,
           std::numeric_limits<double>::quiet_NaN(),
           const_cast<double const *>(val), global_dim);
        break;
      }

      Reporter::line_search
        (trace, i, n_eval, n_grad, fval_old, fval, true,
         (*val - x1) / *dir.get(), const_cast<double const *>(val),
         global_dim);

      bool const has_converged =
        abs(fval - fval_old) < rel_eps * (abs(fval_old) + rel_eps);
      if(has_converged){
        info = 0L;
        break;
      }

      // update the Hessian and take another iteation
      for(auto &f : funcs)
        f.update_Hes(temp_mem);
    }

    return { fval, info, n_eval, n_grad, n_cg };
  }

  /***
   returns the current Hessian approximation.
   */
  void get_hess(double * const __restrict__ hess) const {
    // TODO: make an implementation which returns a sparse Hessian
    std::fill(hess, hess + n_par * n_par, 0.);

    size_t private_offset(global_dim);
    for(auto &f : funcs){
      size_t const iprivate = f.func.private_dim();

      auto get_i = [&](size_t const i, size_t const j){
        size_t const ii = std::min(i, j),
                     jj = std::max(j, i);

        return ii + (jj * (jj + 1L)) / 2L;
      };

      double const * const b = f.B;
      {
        double *h1 = hess,
               *h2 = hess + private_offset;
        for(size_t j = 0; j < global_dim;
            ++j, h1 += n_par, h2 += n_par){
          for(size_t i = 0; i < global_dim; ++i)
            *(h1 + i) += *(b + get_i(i             , j));
          for(size_t i = 0; i < iprivate; ++i)
            *(h2 + i) += *(b + get_i(i + global_dim, j));
        }
      }

      double *h1 = hess + private_offset * n_par,
             *h2 = h1 + private_offset;
      for(size_t j = 0; j < iprivate;
          ++j, h1 += n_par, h2 += n_par){
        for(size_t i = 0; i < global_dim; ++i)
          *(h1 + i) += *(b + get_i(i             , j + global_dim));
        for(size_t i = 0; i < iprivate; ++i)
          *(h2 + i) += *(b + get_i(i + global_dim, j + global_dim));
      }

      private_offset += iprivate;
    }
  }
};
} // namespace PSQN

#endif

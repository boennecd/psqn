#ifndef PSQN_H
#define PSQN_H
#include <vector>
#include <array>
#include <memory>
#include "lp.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include "constant.h"
#include <cmath>
#include "intrapolate.h"
#include "psqn-misc.h"
#include "psqn-bfgs.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace PSQN {
using std::abs;
using std::sqrt;

inline void throw_no_eigen_error(){
#ifndef PSQN_USE_EIGEN
  throw std::logic_error("The code was compilled wihtout having defined 'PSQN_USE_EIGEN' before including the psqn header files. Define 'PSQN_USE_EIGEN' and include Eigen or RcppEigen");
#endif
}

/***
 virtual base class for the element functions.
 */
class base_worker {
  /// logical for whether it is the first call
  bool first_call = true;

public:
  /// number of elements
  psqn_uint const n_ele;
  /// memory for the Hessian approximation
  double * const PSQN_RESTRICT B;
  /// memory for the gradient
  double * const PSQN_RESTRICT gr = B + (n_ele * (n_ele + 1)) / 2L;
  /// memory for the old gradient
  double * const PSQN_RESTRICT gr_old = gr + n_ele;
  /// memory for the old value
  double * const PSQN_RESTRICT x_old = gr_old + n_ele;
  /// memory for the current value
  double * const PSQN_RESTRICT x_new = x_old + n_ele;
  /// bool for whether to use BFGS or SR1 updates
  bool use_bfgs = true;

  base_worker(psqn_uint const n_ele, double * mem): n_ele(n_ele), B(mem) { }
  virtual ~base_worker() = default;

  /***
   save the current parameter values and gradient in order to do update
   the Hessian approximation.
   */
  void record() noexcept {
    lp::copy(x_old , static_cast<double const*>(x_new), n_ele);
    lp::copy(gr_old, static_cast<double const*>(gr   ), n_ele);
  }

  /***
   resets the Hessian approximation.
   */
  void reset() noexcept {
    first_call = true;

    std::fill(B, B + n_ele * n_ele, 0.);
    // set diagonal entries to one
    double *b = B;
    for(psqn_uint i = 0; i < n_ele; ++i, b += i + 1)
      *b = 1.;
  }

  /***
   updates the Hessian approximation. Assumes that x_new, x_old, gr, and gr_new
   have all been set.
   @param wmem working memory to use.
   */
  void update_Hes(double * const wmem){
    // differences in parameters and gradient
    double * const PSQN_RESTRICT s   = wmem,
           * const PSQN_RESTRICT y   = s + n_ele,
           * const PSQN_RESTRICT wrk = y + n_ele;

    lp::vec_diff(x_new, x_old , s, n_ele);

    bool all_unchanged(true);
    for(psqn_uint i = 0; i < n_ele; ++i)
      if(abs(s[i]) > abs(x_new[i]) *
         std::numeric_limits<double>::epsilon() * 100){
        all_unchanged = false;
        break;
      }

    if(!all_unchanged){
      lp::vec_diff(gr, gr_old, y, n_ele);

      if(use_bfgs){
        double const s_y = lp::vec_dot(y, s, n_ele);
        if(first_call){
          first_call = false;
          // make update on page 143
          double const scal = lp::vec_dot(y, n_ele) / s_y;
          double *b = B;
          for(psqn_uint i = 0; i < n_ele; ++i, b += i + 1)
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
          for(psqn_uint i = 0; i < n_ele; ++i, ++yi, ++wi)
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
          for(psqn_uint i = 0; i < n_ele; ++i, b += i + 1)
            *b = scal;
        }

        /// maybe perform SR1
        std::fill(wrk, wrk + n_ele, 0.);
        lp::mat_vec_dot(B, s, wrk, n_ele);
        for(psqn_uint i = 0; i < n_ele; ++i){
          wrk[i] *= -1;
          wrk[i] += y[i];
        }
        double const s_w = lp::vec_dot(s, wrk, n_ele),
          s_norm = sqrt(abs(lp::vec_dot(s, n_ele))),
          wrk_norm = sqrt(abs(lp::vec_dot(wrk, n_ele)));
        constexpr double const r = 1e-8;
        if(abs(s_w) > r * s_norm * wrk_norm)
          lp::rank_one_update(B, wrk, 1. / s_w, n_ele);

      }
    } else
      // essentially no change in the input. Reset the Hessian
      // approximation
      reset();

    record();
  }
};

/***
 virtual base class which computes an element function and its gradient. The
 virtual class is mainly used as a check to ensure that all the member
 functions are implemented.
 */
class element_function {
public:
  /// dimension of the global parameters
  virtual psqn_uint global_dim() const = 0;
  /// dimension of the private parameters
  virtual psqn_uint private_dim() const = 0;

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
    (double const * PSQN_RESTRICT point, double * PSQN_RESTRICT gr)
    const = 0;

  /***
   returns true if the member functions are thread-safe.
   */
  virtual bool thread_safe() const = 0;

  virtual ~element_function() = default;
};

/***
 default class which can be replaced to do prior computation before
 calling the methods on the element function class. This allows one to save
 some computation.
 */
template<class EFunc>
struct default_caller {
  default_caller(std::vector<EFunc const*>&) { }

  /**
   method that is called prior to calling eval_func or eval_grad. The method
   will only be called by the master thread.

   @param val pointer to the value to evaluate the function at.
   @param comp_grad true if eval_grad is going to be called.
   */
  inline void setup(double const *val, bool const comp_grad) { }

  template<class T>
  inline double eval_func(T &obj, double const * val) {
    return obj.func(val);
  }

  template<class T>
  inline double eval_grad(T &obj, double const * val, double *gr) {
    return obj.grad(val, gr);
  }
};

/***
 template class to perform parts of the optimization that is common to
 different particular optimizers. The reporter class can be used to report
 results during the optimization.
 */
template<class OptT, class Reporter = dummy_reporter>
class optimizer_internals {
  OptT * const opt_obj = nullptr;
  double * const temp_mem = nullptr;
  psqn_uint const n_par = 0L;
public:
  /// number of function evaluations
  psqn_uint n_eval = 0L;
  /// number of gradient evaluations
  psqn_uint n_grad = 0L;
  /// number of iterations of conjugate gradient
  psqn_uint n_cg = 0L;

  /***
   reset the counters for the number of evaluations
   */
  void reset_counters() {
    n_eval = 0L;
    n_grad = 0L;
    n_cg = 0L;
  }

  optimizer_internals(OptT *opt_obj):
    opt_obj(opt_obj),
    temp_mem(opt_obj ? opt_obj->get_internals_mem() : nullptr),
    n_par(opt_obj ? opt_obj->n_par : 0L)
    { }

  /***
   evaluates the partially separable and also the gradient if requested.
   @param val pointer to the value to evaluate the function at.
   @param gr pointer to store gradient in.
   @param comp_grad boolean for whether to compute the gradient.
   */
  double eval(double const * val, double * PSQN_RESTRICT gr,
              bool const comp_grad){
    if(comp_grad)
      n_grad++;
    else
      n_eval++;
    return opt_obj->eval(val, gr, comp_grad);
  }

  /***
    conjugate gradient method with diagonal preconditioning. Solves B.y = x
    where B is the Hessian approximation.
    @param tol convergence threshold.
    @param max_cg maximum number of conjugate gradient iterations.
    @param trace controls the amount of tracing information.
    @param pre_method preconditioning method.
   */
  bool conj_grad(double const * PSQN_RESTRICT x, double * PSQN_RESTRICT y,
                 double tol, psqn_uint const max_cg,
                 int const trace, precondition const pre_method){
    double const x_nom = sqrt(abs(lp::vec_dot(x, n_par))),
               tol_use = std::min(tol, sqrt(x_nom));

    if(pre_method == precondition::choleksy){
#ifdef PSQN_USE_EIGEN
      Eigen::SparseMatrix<double> &sparse_B_mat =
        opt_obj->get_hess_sparse_ref();
      Eigen::ConjugateGradient
        <Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper,
         Eigen::IncompleteCholesky<double, Eigen::Lower | Eigen::Upper> > cg;

      // setup the cg object
      cg.analyzePattern(sparse_B_mat);
      cg.factorize(sparse_B_mat);
      if(cg.preconditioner().info() == Eigen::ComputationInfo::NumericalIssue)
        return false;

      cg.setMaxIterations(max_cg);
      cg.setTolerance(tol_use);

      // compute and return
      Eigen::VectorXd rhs(n_par), lhs(n_par);
      for(psqn_uint i = 0; i < n_par; ++i)
        rhs[i] = x[i];

      lhs = cg.solve(rhs);
      // the plus one is almost always right as Eigen only increments the
      // iteration counter if the convergence criteria does not pass (so one is)
      // "missing"
      n_cg += cg.iterations() + 1L;

      Reporter::cg_it(trace, cg.iterations() + 1L, max_cg,
                      cg.error(), tol_use);
      for(psqn_uint i = 0; i < n_par; ++i)
        y[i] = lhs[i];
      return true;
#else
      throw_no_eigen_error();
      return false;
#endif
    }

    // eigen does not include this factor
    double const eps = tol_use * x_nom;

    double * PSQN_RESTRICT r       = temp_mem,
           * PSQN_RESTRICT p       = r      + n_par,
           * PSQN_RESTRICT B_p     = p      + n_par,
           * PSQN_RESTRICT v       = B_p    + n_par,
           * PSQN_RESTRICT B_diag  = v      + n_par;
    bool const do_pre = pre_method == precondition::diag;

    // setup before first iteration
    if(do_pre){
      opt_obj->get_diag(B_diag);
      double *b = B_diag;
      for(psqn_uint i = 0; i < n_par; ++i, ++b)
        *b = 1. / *b; // want to use multiplication rather than division
    }

    auto diag_solve = [&](double       * PSQN_RESTRICT vy,
                          double const * PSQN_RESTRICT vx) -> void {
      double * di = B_diag;
      for(psqn_uint i = 0; i < n_par; ++i)
        *vy++ = *vx++ * *di++;
    };

    std::fill(y, y + n_par, 0.);
    for(psqn_uint i = 0; i < n_par; ++i){
      r[i] = -x[i];
      if(!do_pre)
        p[i] = x[i];
    }

    if(do_pre){
      diag_solve(v, r);
      for(psqn_uint i = 0; i < n_par; ++i)
        p[i] = -v[i];
    }

    auto get_r_v_dot = [&]() -> double {
      return do_pre ? lp::vec_dot(r, v, n_par) : lp::vec_dot(r, n_par);
    };

    auto comp_B_vec = opt_obj->get_comp_B_vec();

    double old_r_v_dot = get_r_v_dot();
    for(psqn_uint i = 0; i < max_cg; ++i){
      ++n_cg;
      std::fill(B_p, B_p + n_par, 0.);
      comp_B_vec(p, B_p);

      double const p_B_p = lp::vec_dot(p, B_p, n_par);
      if(p_B_p <= 0){
        // negative curvature. Thus, exit
        if(i < 1L)
          // set output to be the gradient
          for(psqn_uint j = 0; j < n_par; ++j)
            y[j] = x[j];

        break;
      }
      double const alpha = old_r_v_dot / p_B_p;

      for(psqn_uint j = 0; j < n_par; ++j){
        y[j] += alpha *   p[j];
        r[j] += alpha * B_p[j];
      }

      if(do_pre)
        diag_solve(v, r);
      double const r_v_dot = get_r_v_dot(),
                   t_val   = do_pre ? sqrt(abs(lp::vec_dot(r, n_par))) :
                                      sqrt(abs(r_v_dot));
      Reporter::cg_it(trace, i, max_cg, t_val, eps);
      if(t_val < eps)
        break;

      double const beta = r_v_dot / old_r_v_dot;
      old_r_v_dot = r_v_dot;
      for(psqn_uint j = 0; j < n_par; ++j){
        p[j] *= beta;
        p[j] -= do_pre ? v[j] : r[j];
      }
    }

    return true;
  }

  /***
   performs line search to satisfy the Wolfe condition.
   @param f0 value of the functions at the current value.
   @param x0 value the function is evaluated.
   @param gr0 value of the current gradient.
   @param dir direction to search in.
   @param fnew the function value at the found solution.
   @param c1,c2 thresholds for Wolfe condition.
   @param strong_wolfe true if the strong Wolfe condition should be used.
   @param trace controls the amount of tracing information.

   x0 and gr0 contains the new value and gradient on return. The method
   returns false if the line search fails.
   */
  bool line_search(
      double const f0, double * PSQN_RESTRICT x0, double * PSQN_RESTRICT gr0,
      double * PSQN_RESTRICT dir, double &fnew, double const c1,
      double const c2, bool const strong_wolfe, int const trace){
    double * const x_mem = temp_mem;

    // declare 1D functions
    auto psi = [&](double const alpha) -> double {
      for(psqn_uint i = 0; i < n_par; ++i)
        x_mem[i] = x0[i] + alpha * dir[i];

      return eval(x_mem, nullptr, false);
    };

    // returns the function value and the gradient
    auto dpsi = [&](double const alpha) -> double {
      for(psqn_uint i = 0; i < n_par; ++i)
        x_mem[i] = x0[i] + alpha * dir[i];

      fnew = eval(x_mem, gr0, true);
      return lp::vec_dot(gr0, dir, n_par);
    };

    // the above at alpha = 0
    double const dpsi_zero = lp::vec_dot(gr0, dir, n_par);
    if(dpsi_zero > 0)
      // not a descent direction
      return false;

    constexpr psqn_uint const max_it = 20L;
    static double const NaNv = std::numeric_limits<double>::quiet_NaN();
    auto zoom =
      [&](double a_low, double a_high, intrapolate &inter) -> bool {
        double f_low = psi(a_low);
        for(psqn_uint i = 0; i < max_it; ++i){
          double const ai = inter.get_value(a_low, a_high),
                       fi = psi(ai);
          if(!std::isfinite(fi)){
            // this is very bad! We do not know in which direction to go.
            // We try to go in a lower direction
            if(a_low < a_high)
              a_high = ai;
            else
              a_low = ai;
            continue;
          }

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
    bool found_ok_prev = false,
         failed_once   = false;
    double mult = 2;
    for(psqn_uint i = 0; i < max_it; ++i){
      ai *= mult;
      double fi = psi(ai);
      Reporter::line_search_inner(trace, a_prev, ai, fi, false,
                                  NaNv, NaNv);
      if(!std::isfinite(fi)){
        // handle inf/nan case
        failed_once = true;
        mult = .5;

        if(!found_ok_prev)
          // no valid previous value yet to use
          continue;
        else {
          // the previous value was ok. Use that one
          fi = fold;
          ai = a_prev;
        }
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

      if(failed_once and fi < f0){
        // effectively just line search
        lp::copy(x0, x_mem, n_par);
        return false;
      }

      if(dpsi_i >= 0){
        intrapolate inter = ([&]() -> intrapolate {
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
};

/***
 The reporter class can be used to report results during the optimization.
 */
template<class EFunc, class Reporter = dummy_reporter,
         class interrupter = dummy_interrupter,
         class T_caller = default_caller<EFunc> >
class optimizer {
  /***
   worker class to hold an element function and the element function's
   Hessian approximation.
   */
  class worker final : public base_worker {
  public:
    /// the element function for this worker
    EFunc const func;
    /// indices of first set of private parameters
    psqn_uint const par_start;

    worker(EFunc &&func, double * mem, psqn_uint const par_start):
      base_worker(func.global_dim() + func.private_dim(), mem),
      func(func), par_start(par_start) {
      reset();
    }

    /***
     computes the element function and possibly its gradient.

     @param global values for the global parameters
     @param vprivate values for private parameters.
     @param comp_grad logical for whether to compute the gradient
     @param call_obj object to evaluate the function or gradient given an
                     EFunc object.
     */
    double operator()
      (double const * PSQN_RESTRICT global,
       double const * PSQN_RESTRICT vprivate, bool const comp_grad,
       T_caller &call_obj){
      // copy values
      psqn_uint const d_global  = func.global_dim(),
                      d_private = func.private_dim();

      lp::copy(x_new           , global  , d_global);
      lp::copy(x_new + d_global, vprivate, d_private);

      if(!comp_grad)
        return call_obj.eval_func(func, static_cast<double const *>(x_new));

      double const out = call_obj.eval_grad(
        func, static_cast<double const *>(x_new), gr);

      return out;
    }
  };

  /**
   class to optimize one set of private parameters given the global
   parameters. */
  class sub_problem final : public problem {
    worker &w;
    double const * g_val;
    psqn_uint const p_dim = w.func.private_dim(),
                    g_dim = w.func.global_dim();
    T_caller &call_obj;

  public:
    sub_problem(worker &w, double const *g_val, T_caller &call_obj):
    w(w), g_val(g_val), call_obj(call_obj) { }

    psqn_uint size() const {
      return p_dim;
    }
    double func(double const *val){
      return w(g_val, val, false, call_obj);
    }
    double grad(double const * PSQN_RESTRICT val,
                double       * PSQN_RESTRICT gr){
      double const out = w(g_val, val, true, call_obj);
      for(psqn_uint i = 0; i < p_dim; ++i)
        gr[i] = w.gr[i + g_dim];
      return out;
    }
  };

public:
  /// dimension of the global parameters
  psqn_uint const global_dim;
  /// true if the element functions are thread-safe
  bool const is_ele_func_thread_safe;
  /// total number of parameters
  psqn_uint const n_par;

private:
  /***
   size of the allocated working memory. The first element is needed for
   the workers. The second element is needed during the computation for the
   master thread. The third element is number required per thread.
  */
  std::array<std::size_t, 3L> const n_mem;
  /// maximum number of threads to use
  psqn_uint const max_threads;
  /// working memory
  std::unique_ptr<double[]> mem =
    std::unique_ptr<double[]>(
        new double[n_mem[0] + n_mem[1] + max_threads * n_mem[2]]);
  /// pointer to temporary memory to use on the master thread
  double * const temp_mem = mem.get() + n_mem[0];
  /// pointer to temporary memory to be used by the threads
  double * const temp_thread_mem = temp_mem + n_mem[1];
  /// element functions
  std::vector<worker> funcs;
  /// object to do computations prior to evaluating the element functions.
  T_caller caller = T_caller(([&](std::vector<worker> &fs) -> T_caller {
    std::vector<EFunc const*> ele_funcs;
    size_t n_ele_funcs = fs.size();
    ele_funcs.reserve(n_ele_funcs);
    for(auto &f : fs)
      ele_funcs.emplace_back(&f.func);
    return T_caller(ele_funcs);
  })(funcs));

  /// returns the thread number.
  int get_thread_num() const noexcept {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0L;
#endif
  }

  /// returns working memory for this thread
  double * get_thread_mem(int const thread_num) const noexcept {
    return temp_thread_mem + thread_num * n_mem[2];
  }

  double * get_thread_mem() const noexcept {
    return get_thread_mem(get_thread_num());
  }

  /// number of threads to use
  int n_threads = 1L;

  // inner optimizer to use
  std::unique_ptr<optimizer_internals<optimizer, Reporter> > opt_internals;
  friend class optimizer_internals<optimizer, Reporter>;

  // function need for optimizer_internals to get temporary memory
  double * get_internals_mem() {
    // need memory for B_start!
    return temp_mem + (global_dim * (global_dim + 1)) / 2;
  }

  // class to compute B.x for optimizer_internals
  class comp_B_vec_obj {
    bool is_first_call = true;
    double * B_start;
    optimizer<EFunc, Reporter, interrupter, T_caller> &obj;
  public:
    comp_B_vec_obj(double * mem,
                   optimizer<EFunc, Reporter, interrupter, T_caller> &obj):
    B_start(mem), obj(obj) { }


    void operator()(double const * const PSQN_RESTRICT val,
                    double * const PSQN_RESTRICT res) noexcept {
      obj.B_vec(val, res, B_start, is_first_call);
      is_first_call = false;
    }
  };

  inline comp_B_vec_obj get_comp_B_vec() {
    return comp_B_vec_obj(temp_mem, *this);
  }

public:
  /// set the number of threads to use
  void set_n_threads(psqn_uint const n_threads_new) noexcept {
#ifdef _OPENMP
    n_threads = std::max(
      static_cast<psqn_uint>(1L), std::min(n_threads_new, max_threads));
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
  optimizer(std::vector<EFunc> &funcs_in, psqn_uint const max_threads):
  global_dim(([&]() -> psqn_uint {
    if(funcs_in.size() < 1L)
      throw std::invalid_argument(
          "optimizer<EFunc>::optimizer: no functions supplied");
    return funcs_in[0].global_dim();
  })()),
  is_ele_func_thread_safe(funcs_in[0].thread_safe()),
  n_par(([&]() -> psqn_uint {
    psqn_uint out(global_dim);
    for(auto &f : funcs_in)
      out += f.private_dim();
    return out;
  })()),
  n_mem(([&]() -> std::array<std::size_t, 3L> {
    std::size_t out(0L),
           max_priv(0L);
    for(auto &f : funcs_in){
      if(f.global_dim() != global_dim)
        throw std::invalid_argument(
            "optimizer<EFunc>::optimizer: global_dim differs");
      if(f.thread_safe() != is_ele_func_thread_safe)
        throw std::invalid_argument(
            "optimizer<EFunc>::optimizer: thread_safe differs");
      std::size_t const private_dim = f.private_dim(),
                        n_ele       = private_dim + global_dim;
      if(max_priv < private_dim)
        max_priv = private_dim;

      out += n_ele * 4L + (n_ele * (n_ele + 1L)) / 2L;
    }

    constexpr std::size_t const mult = cacheline_size() / sizeof(double),
                            min_size = 2L * mult;

    std::size_t thread_mem = std::max<std::size_t>(
      3L * (global_dim + max_priv), min_size);
    thread_mem = (thread_mem + mult - 1L) / mult;
    thread_mem *= mult;

    std::size_t master_mem(5L * n_par);
    master_mem += (global_dim * (global_dim + 1L)) / 2L;

    std::array<std::size_t, 3L> ret {out, master_mem, thread_mem };
    return ret;
  })()),
  max_threads(max_threads > 0 ? max_threads : 1L),
  funcs(([&]() -> std::vector<worker> {
    std::vector<worker> out;
    psqn_uint const n_ele(funcs_in.size());
    out.reserve(funcs_in.size());

    double * mem_ptr = mem.get();
    psqn_uint i_start(global_dim);
    for(psqn_uint i = 0; i < n_ele; ++i){
      out.emplace_back(std::move(funcs_in[i]), mem_ptr, i_start);
      psqn_uint const n_ele = out.back().n_ele;
      mem_ptr += n_ele * 4L + (n_ele * (n_ele + 1L)) / 2L;
      i_start += out.back().func.private_dim();
    }

    return out;
  })()) {
    opt_internals.reset(new optimizer_internals<optimizer, Reporter>(this));
  }

  /***
   evaluates the partially separable function and also the gradient if
   requested.
   @param val pointer to the value to evaluate the function at.
   @param gr pointer to store gradient in.
   @param comp_grad boolean for whether to compute the gradient.
   */
  double eval(double const * val, double * PSQN_RESTRICT gr,
              bool const comp_grad){
    caller.setup(val, comp_grad);

    psqn_uint const n_funcs = funcs.size();
    auto serial_version = [&]() -> double {
      double out(0.);
      for(psqn_uint i = 0; i < n_funcs; ++i){
        auto &f = funcs[i];
        out += f(val, val + f.par_start, comp_grad, caller);
      }

      if(comp_grad){
        std::fill(gr, gr + global_dim, 0.);
        for(psqn_uint i = 0; i < n_funcs; ++i){
          auto const &f = funcs[i];
          for(psqn_uint j = 0; j < global_dim; ++j)
            gr[j] += f.gr[j];

          psqn_uint const iprivate = f.func.private_dim();
          lp::copy(gr + f.par_start, f.gr + global_dim, iprivate);
        }
      }

      return out;
    };

    if(n_threads < 2L or !is_ele_func_thread_safe)
      return serial_version();

#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
    {
    double * r_mem = get_thread_mem(),
           * v_mem =
               r_mem + global_dim + 1L /* leave 1 ele for func value*/;
    lp::copy(v_mem, val, global_dim);
    if(comp_grad)
      std::fill(r_mem, r_mem + global_dim, 0.);

    double &thread_terms = *(r_mem + global_dim);
    thread_terms = 0;
#pragma omp for schedule(static)
    for(psqn_uint i = 0; i < n_funcs; ++i){
      auto &f = funcs[i];
      thread_terms += f(v_mem, val + f.par_start, comp_grad, caller);

      if(comp_grad){
        // update global
        double *lhs = r_mem;
        double const *rhs = f.gr;
        for(psqn_uint j = 0; j < global_dim; ++j, ++lhs, ++rhs)
          *lhs += *rhs;

        // update private
        lp::copy(gr + f.par_start, f.gr + global_dim, f.func.private_dim());
      }
    }
    }

    if(comp_grad)
      std::fill(gr, gr + global_dim, 0.);

    // add to global parameters
    double out(0.);
    for (int t = 0; t < n_threads; t++){
      double const *r_mem = get_thread_mem(t);
      if(comp_grad)
        for(psqn_uint i = 0; i < global_dim; ++i)
          gr[i] += r_mem[i];
      out += r_mem[global_dim];
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
   @param B_start memory with the [global_dim] x [global_dim] part of B.
   @param comp_B_start true if B_start should be computed.
   ***/
  void B_vec(double const * const PSQN_RESTRICT val,
             double * const PSQN_RESTRICT res,
             double * const PSQN_RESTRICT B_start,
             bool const comp_B_start) const noexcept {
    psqn_uint const n_funcs = funcs.size();

    // aggregate the first part of B if needed
    if(comp_B_start){
      psqn_uint const B_sub_ele = (global_dim * (global_dim + 1)) / 2;
      std::fill(B_start, B_start + B_sub_ele, 0);
      for(psqn_uint i = 0; i < n_funcs; ++i){
        auto &f = funcs[i];
        double * b = B_start;
        double const *b_inc = f.B;
        for(psqn_uint j = 0; j < B_sub_ele; ++j)
          *b++ += *b_inc++;
      }
    }

    // compute the first part
    lp::mat_vec_dot(B_start, val, res, global_dim);

    // the serial version
    auto serial_version = [&]() -> void {
      for(psqn_uint i = 0; i < n_funcs; ++i){
        auto &f = funcs[i];
        psqn_uint const iprivate = f.func.private_dim(),
                  private_offset = f.par_start;

        lp::mat_vec_dot_excl_first(f.B, val, val + private_offset, res,
                                   res + private_offset, global_dim,
                                   iprivate);
      }
    };

    if(n_threads < 2L){
      serial_version();
      return;
    }

#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
    {
    double * r_mem = get_thread_mem(),
           * v_mem = r_mem + global_dim;
    lp::copy(v_mem, val, global_dim);
    std::fill(r_mem, r_mem + global_dim, 0.);

#pragma omp for schedule(static)
    for(psqn_uint i = 0; i < n_funcs; ++i){
      auto &f = funcs[i];
      psqn_uint const iprivate = f.func.private_dim(),
                private_offset = f.par_start;

      lp::mat_vec_dot_excl_first(f.B, v_mem, val + private_offset, r_mem,
                                 res + private_offset, global_dim,
                                 iprivate);
    }
    }

    // add to global parameters
    for (int t = 0; t < n_threads; t++){
      double const *r_mem = get_thread_mem(t);
      for(psqn_uint i = 0; i < global_dim; ++i)
        res[i] += r_mem[i];
    }
#else
    serial_version();
#endif
  }

  /**
    computes the diagonal of B.
   */
  void get_diag(double * x){
    std::fill(x, x + global_dim, 0.);
    double * PSQN_RESTRICT x_priv = x + global_dim;

    for(psqn_uint i = 0; i < funcs.size(); ++i){
      auto &f = funcs[i];
      psqn_uint const iprivate = f.func.private_dim();

      // add to the global parameters
      double const * b_diag = f.B;
      psqn_uint j = 0L;
      for(; j <            global_dim; ++j, b_diag += j + 1)
        x[j] += *b_diag;

      for(; j < iprivate + global_dim; ++j, b_diag += j + 1)
        *x_priv++ = *b_diag;
    }
  }

  /***
    conjugate gradient method with diagonal preconditioning. Solves B.y = x
    where B is the Hessian approximation.
    @param tol convergence threshold.
    @param max_cg maximum number of conjugate gradient iterations.
    @param trace controls the amount of tracing information.
    @param pre_method preconditioning method.
   */
  bool conj_grad(double const * PSQN_RESTRICT x, double * PSQN_RESTRICT y,
                 double const tol, psqn_uint const max_cg,
                 int const trace, precondition const pre_method){
    return opt_internals->conj_grad(x, y, tol, max_cg, trace, pre_method);
  }

  /***
   performs line search to satisfy the Wolfe condition.
   @param f0 value of the functions at the current value.
   @param x0 value the function is evaluated.
   @param gr0 value of the current gradient.
   @param dir direction to search in.
   @param fnew the function value at the found solution.
   @param c1,c2 thresholds for Wolfe condition.
   @param strong_wolfe true if the strong Wolfe condition should be used.
   @param trace controls the amount of tracing information.

   x0 and gr0 contains the new value and gradient on return. The method
   returns false if the line search fails.
   */
  bool line_search(
      double const f0, double * PSQN_RESTRICT x0, double * PSQN_RESTRICT gr0,
      double * PSQN_RESTRICT dir, double &fnew, double const c1,
      double const c2, bool const strong_wolfe, int const trace){
    return opt_internals->line_search(f0, x0, gr0, dir, fnew, c1, c2,
                                      strong_wolfe, trace);
  }

  /***
   optimizes the partially separable function.
   @param val pointer to starting value. Set to the final estimate at the
   end.
   @param rel_eps relative convergence threshold.
   @param max_it maximum number of iterations.
   @param c1,c2 thresholds for Wolfe condition.
   @param use_bfgs bool for whether to use BFGS updates or SR1 updates.
   @param trace integer with info level passed to reporter.
   @param cg_tol threshold for conjugate gradient method.
   @param strong_wolfe true if the strong Wolfe condition should be used.
   @param max_cg maximum number of conjugate gradient iterations in each
   iteration. Use zero if there should not be a limit.
   @param pre_method preconditioning method.
   */
  optim_info optim
    (double * val, double const rel_eps, psqn_uint const max_it,
     double const c1, double const c2,
     bool const use_bfgs = true, int const trace = 0,
     double const cg_tol = .5, bool const strong_wolfe = true,
     psqn_uint const max_cg = 0,
     precondition const pre_method = precondition::diag){
    opt_internals->reset_counters();
    for(auto &f : funcs){
      f.reset();
      f.use_bfgs = use_bfgs;
    }

    std::unique_ptr<double[]> gr(new double[n_par]),
                             dir(new double[n_par]);

    // evaluate the gradient at the current value
    double fval = eval(val, gr.get(), true);
    for(auto &f : funcs)
      f.record();

    info_code info = info_code::max_it_reached;
    int n_line_search_fail = 0;
    for(psqn_uint i = 0; i < max_it; ++i){
      if(i % 10 == 0)
        if(interrupter::check_interrupt()){
          info = info_code::user_interrupt;
          break;
        }

      double const fval_old = fval;
      if(!conj_grad(gr.get(), dir.get(), cg_tol,
                    max_cg < 1 ? n_par : max_cg, trace,
                    pre_method)){
        info = info_code::conjugate_gradient_failed;
        Reporter::cg(trace, i, opt_internals->n_cg, false);
        break;
      } else
        Reporter::cg(trace, i, opt_internals->n_cg, true);

      for(double * d = dir.get(); d != dir.get() + n_par; ++d)
        *d *= -1;

      double const x1 = *val;
      if(!line_search(fval_old, val, gr.get(), dir.get(), fval, c1, c2,
                      strong_wolfe, trace)){
        info = info_code::line_search_failed;
        Reporter::line_search
          (trace, i, opt_internals->n_eval, opt_internals->n_grad, fval_old,
           fval, false, std::numeric_limits<double>::quiet_NaN(),
           const_cast<double const *>(val), global_dim);
        if(++n_line_search_fail > 2)
          break;
      } else{
        n_line_search_fail = 0;
        Reporter::line_search
          (trace, i, opt_internals->n_eval, opt_internals->n_grad, fval_old, fval,
           true, (*val - x1) / *dir.get(), const_cast<double const *>(val),
           global_dim);

      }

      bool const has_converged =
        abs(fval - fval_old) < rel_eps * (abs(fval_old) + rel_eps);
      if(has_converged){
        info = info_code::converged;
        break;
      }

      // update the Hessian and take another iteration
      if(n_line_search_fail < 1){
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif
        {
          double * const tmp_mem_use = get_thread_mem();
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
          for(psqn_uint i = 0; i < funcs.size(); ++i)
            funcs[i].update_Hes(tmp_mem_use);
        }
      } else
        for(psqn_uint i = 0; i < funcs.size(); ++i){
          funcs[i].reset();
          funcs[i].record();
        }
    }

    return { fval, info, opt_internals->n_eval, opt_internals->n_grad,
             opt_internals->n_cg };
  }

  /***
   returns the current Hessian approximation.
   */
  void get_hess(double * const PSQN_RESTRICT hess) const {
    std::fill(hess, hess + n_par * n_par, 0.);

    psqn_uint private_offset(global_dim);
    for(auto &f : funcs){
      psqn_uint const iprivate = f.func.private_dim();

      auto get_i = [&](psqn_uint const i, psqn_uint const j) -> psqn_uint {
        psqn_uint const ii = std::min(i, j),
                        jj = std::max(j, i);

        return ii + (jj * (jj + 1L)) / 2L;
      };

      double const * const b = f.B;
      {
        double *h1 = hess,
               *h2 = hess + private_offset;
        for(psqn_uint j = 0; j < global_dim;
            ++j, h1 += n_par, h2 += n_par){
          for(psqn_uint i = 0; i < global_dim; ++i)
            h1[i] += b[get_i(i             , j)];
          for(psqn_uint i = 0; i < iprivate; ++i)
            h2[i] += b[get_i(i + global_dim, j)];
        }
      }

      double *h1 = hess + private_offset * n_par,
             *h2 = h1 + private_offset;
      for(psqn_uint j = 0; j < iprivate;
          ++j, h1 += n_par, h2 += n_par){
        for(psqn_uint i = 0; i < global_dim; ++i)
          h1[i] += b[get_i(i             , j + global_dim)];
        for(psqn_uint i = 0; i < iprivate; ++i)
          h2[i] += b[get_i(i + global_dim, j + global_dim)];
      }

      private_offset += iprivate;
    }
  }

  /***
   optimizes the private parameters given the global parameters.
   @param val pointer to starting value. Set to the final estimate at the
   end.
   @param rel_eps relative convergence threshold.
   @param max_it maximum number of iterations.
   @param c1,c2 thresholds for Wolfe condition.

   If you supply a non-default T_caller class, T_caller.setup is only called
   on the starting values. Thus, this function only yields valid results if
   this is what is expected.
   */
  double optim_priv
  (double * val, double const rel_eps, psqn_uint const max_it,
   double const c1, double const c2){
    double out(0.);
    caller.setup(val, true);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads) reduction(+:out) if(is_ele_func_thread_safe)
#endif
    for(psqn_uint i = 0; i < funcs.size(); ++i){
      auto &f = funcs[i];
      sub_problem prob(f, val, caller);
      double * const p_val = val + f.par_start;

      auto const opt_out = bfgs(prob, p_val, rel_eps, max_it, c1, c2, 0L);
      out += opt_out.value;
    }

    return out;
  }

  /**
   returns a pointers to the element function objects.  This may be useful if
   the element functions hold data.
   */
  std::vector<EFunc const *> get_ele_funcs() const {
    std::vector<EFunc const *> out;
    out.reserve(funcs.size());
    for(auto &o : funcs)
      out.emplace_back(&o.func);
    return out;
  }

#ifdef PSQN_USE_EIGEN
private:
  /**
   the sparse matrix we will fill. We keep this as a member to avoid repeated
   memory allocations.
   */
  Eigen::SparseMatrix<double> sparse_B_mat;
  /// vector of triplets we use to fill sparse_B_mat
  std::vector<Eigen::Triplet<double> > sparse_B_mat_triplets;

  /// fills the sparse_B_mat
  inline void fill_sparse_B_mat(){
    // fill in the triplets
    sparse_B_mat_triplets.clear();

    // fill in the data
    {
      psqn_uint private_offset(0L);

      // object to store the first block
      psqn_uint const B_sub_ele = (global_dim * (global_dim + 1)) / 2;
      std::fill(temp_mem, temp_mem + B_sub_ele, 0);

      for(auto const &f : funcs){
        // fill in the first block
        double const *b_inc = f.B;
        {
          double * b = temp_mem;
          for(psqn_uint j = 0; j < B_sub_ele; ++j)
            *b++ += *b_inc++;
        }

        psqn_uint const iprivate = f.func.private_dim();
        for(psqn_uint j = global_dim; j < global_dim + iprivate; ++j){
          psqn_uint i = 0;
          for(; i < global_dim; ++i){
            psqn_uint const j_shift = j + private_offset;
            if(i < j)
              sparse_B_mat_triplets.emplace_back(i      , j_shift, *b_inc  );
            sparse_B_mat_triplets.emplace_back  (j_shift, i      , *b_inc++);
          }

          for(; i <= j; ++i){
            psqn_uint const i_shift = i + private_offset,
                            j_shift = j + private_offset;
            if(i < j)
              sparse_B_mat_triplets.emplace_back(i_shift, j_shift, *b_inc  );
            sparse_B_mat_triplets.emplace_back  (j_shift, i_shift, *b_inc++);
          }
        }

        private_offset += iprivate;
      }

      // fill the global parameter part
      double * b = temp_mem;
      for(psqn_uint j = 0; j < global_dim; ++j)
        for(psqn_uint i = 0; i <= j; ++i){
          if(i < j)
            sparse_B_mat_triplets.emplace_back(i, j, *b  );
          sparse_B_mat_triplets.emplace_back  (j, i, *b++);
        }
    }

    sparse_B_mat.resize(n_par, n_par);
    sparse_B_mat.setFromTriplets(
      sparse_B_mat_triplets.begin(), sparse_B_mat_triplets.end());
  }

  Eigen::SparseMatrix<double> & get_hess_sparse_ref() {
    fill_sparse_B_mat();
    return sparse_B_mat;
  }

#endif // PSQN_USE_EIGEN

public:
  /***
   returns the current Hessian approximation as sparse matrix.
   */
#ifdef PSQN_USE_EIGEN
  Eigen::SparseMatrix<double> get_hess_sparse() {
    fill_sparse_B_mat();
    return sparse_B_mat;
  }
#else
  void get_hess_sparse() {
    throw_no_eigen_error();
  }
#endif // PSQN_USE_EIGEN
};

/***
 same as element_function but for the generic interface.
 */
class element_function_generic {
public:
  /// indices of the element function
  virtual psqn_uint const * indices() const = 0;
  /// number of element in induces
  virtual psqn_uint n_args() const = 0;

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
    (double const * PSQN_RESTRICT point, double * PSQN_RESTRICT gr)
    const = 0;

  /***
   returns true if the member functions are thread-safe.
   */
  virtual bool thread_safe() const = 0;

  virtual ~element_function_generic() = default;
};

/***
 similar to optimizer but for generic partially separable functions. This has
 some extra overhead.
 */
template<class EFunc, class Reporter = dummy_reporter,
         class interrupter = dummy_interrupter,
         class T_caller = default_caller<EFunc> >
class optimizer_generic {
  /***
   worker class to hold an element function and the element function's
   Hessian approximation.
   */
  class worker final : public base_worker {
  public:
    /// the element function for this worker
    EFunc const func;
    /// number of argument to func
    psqn_uint const n_args = func.n_args();
    /// indices of the arguments to pass to func
    psqn_uint const * indices() const {
      return func.indices();
    }

    worker(EFunc &&func, double * mem):
      base_worker(func.n_args(), mem),
      func(func) {
      reset();
    }

    /***
     computes the element function and possibly its gradient.

     @param point values of all parameters at which to evalute the element
                  function.
     @param comp_grad logical for whether to compute the gradient
     @param call_obj object to evaluate the function or gradient given an
                     EFunc object.
     */
    double operator()
      (double const * PSQN_RESTRICT point, bool const comp_grad,
       T_caller &call_obj){
      // copy values
      psqn_uint const * idx_i = indices();
      for(psqn_uint i = 0; i < n_args; ++i, ++idx_i)
        x_new[i] = point[*idx_i];

      if(!comp_grad)
        return call_obj.eval_func(func, static_cast<double const *>(x_new));

      double const out = call_obj.eval_grad(
        func, static_cast<double const *>(x_new), gr);

      return out;
    }
  };

public:
  /// true if the element functions are thread-safe
  bool const is_ele_func_thread_safe;
  /// total number of parameters
  psqn_uint const n_par;

private:
  /***
   size of the allocated working memory. The first element is needed for
   the workers. The second element is needed during the computation for the
   master thread. The third element is number required per thread.
  */
  std::array<std::size_t, 3L> const n_mem;
  /// maximum number of threads to use
  psqn_uint const max_threads;
  /// working memory
  std::unique_ptr<double[]> mem =
    std::unique_ptr<double[]>(
        new double[n_mem[0] + n_mem[1] + max_threads * n_mem[2]]);
  /// pointer to temporary memory to use on the master thread
  double * const temp_mem = mem.get() + n_mem[0];
  /// pointer to temporary memory to be used by the threads
  double * const temp_thread_mem = temp_mem + n_mem[1];
  /// element functions
  std::vector<worker> funcs;
  /// object to do computations prior to evaluating the element functions.
  T_caller caller = T_caller(([&](std::vector<worker> &fs) -> T_caller {
    std::vector<EFunc const*> ele_funcs;
    ele_funcs.reserve(fs.size());
    for(auto &f : fs)
      ele_funcs.emplace_back(&f.func);
    return T_caller(ele_funcs);
  })(funcs));

  /// returns the thread number.
  int get_thread_num() const noexcept {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0L;
#endif
  }

  /// returns working memory for this thread
  double * get_thread_mem(int const thread_num) const noexcept {
    return temp_thread_mem + thread_num * n_mem[2];
  }

  double * get_thread_mem() const noexcept {
    return get_thread_mem(get_thread_num());
  }

  /// number of threads to use
  int n_threads = 1L;

  // inner optimizer to use
  std::unique_ptr<optimizer_internals<optimizer_generic, Reporter> >
    opt_internals;
  friend class optimizer_internals<optimizer_generic, Reporter>;

  // function need for optimizer_internals to get temporary memory
  double * get_internals_mem() {
    return temp_mem;
  }

  // class to compute B.x for optimizer_internals
  class comp_B_vec_obj {
    optimizer_generic<EFunc, Reporter, interrupter, T_caller> &obj;
  public:
    comp_B_vec_obj
    (optimizer_generic<EFunc, Reporter, interrupter, T_caller> &obj):
    obj(obj) { }

    void operator()(double const * const PSQN_RESTRICT val,
                    double * const PSQN_RESTRICT res) noexcept {
      obj.B_vec(val, res);
    }
  };

  inline comp_B_vec_obj get_comp_B_vec() {
    return comp_B_vec_obj(*this);
  }

public:
  /// set the number of threads to use
  void set_n_threads(psqn_uint const n_threads_new) noexcept {
#ifdef _OPENMP
    n_threads = std::max(
      static_cast<psqn_uint>(1L), std::min(n_threads_new, max_threads));
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
  optimizer_generic(std::vector<EFunc> &funcs_in, psqn_uint const max_threads):
  is_ele_func_thread_safe(funcs_in[0].thread_safe()),
  n_par(([&]() -> psqn_uint {
    psqn_uint out(0L);
    for(auto &f : funcs_in){
      psqn_uint const * idx_i = f.indices();
      for(psqn_uint i = 0; i < f.n_args(); ++i, ++idx_i)
        if(*idx_i > out)
          out = *idx_i;
    }
    return out + 1L;
  })()),
  n_mem(([&]() -> std::array<std::size_t, 3L> {
    std::size_t out(0L),
           max_args(0L);
    for(auto &f : funcs_in){
      if(f.thread_safe() != is_ele_func_thread_safe)
        throw std::invalid_argument(
            "optimizer_generic<EFunc>::optimizer: thread_safe differs");
      std::size_t const n_args = f.n_args();
      if(max_args < n_args)
        max_args = n_args;

      out += n_args * 4L + (n_args * (n_args + 1L)) / 2L;
    }

    constexpr std::size_t const mult = cacheline_size() / sizeof(double),
                            min_size = 2L * mult;
    std::size_t const n_extra = std::min<std::size_t>(2L, max_args);

    std::size_t thread_mem = std::max<std::size_t>(
      2L * n_par + n_extra, min_size);
    thread_mem = std::max<std::size_t>(thread_mem, 3L * max_args);
    thread_mem = (thread_mem + mult - 1L) / mult;
    thread_mem *= mult;

    size_t const master_mem(5L * n_par);

    std::array<std::size_t, 3L> ret { out, master_mem, thread_mem };
    return ret;
  })()),
  max_threads(max_threads > 0 ? max_threads : 1L),
  funcs(([&]() -> std::vector<worker> {
    std::vector<worker> out;
    psqn_uint const n_ele(funcs_in.size());
    out.reserve(funcs_in.size());

    double * mem_ptr = mem.get();
    for(psqn_uint i = 0; i < n_ele; ++i){
      out.emplace_back(std::move(funcs_in[i]), mem_ptr);
      psqn_uint const n_args = out.back().n_args;
      mem_ptr += n_args * 4L + (n_args * (n_args + 1L)) / 2L;
    }

    return out;
  })()) {
    opt_internals.reset(
      new optimizer_internals<optimizer_generic, Reporter>(this));
  }

  /***
   evaluates the partially separable function and also the gradient if
   requested.
   @param val pointer to the value to evaluate the function at.
   @param gr pointer to store gradient in.
   @param comp_grad boolean for whether to compute the gradient.
   */
  double eval(double const * val, double * PSQN_RESTRICT gr,
              bool const comp_grad){
    caller.setup(val, comp_grad);

    psqn_uint const n_funcs = funcs.size();
    auto serial_version = [&]() -> double {
      double out(0.), out_comp(0.);
      for(psqn_uint i = 0; i < n_funcs; ++i){
        auto &f = funcs[i];
        lp::Kahan(out, out_comp, f(val, comp_grad, caller));
      }

      if(comp_grad){
        // add the gradients
        double * comp_mem = get_thread_mem();
        std::fill(gr      , gr       + n_par, 0.);
        std::fill(comp_mem, comp_mem + n_par, 0.);

        for(auto const &f : funcs){
          psqn_uint const * idx_j = f.indices();
          for(psqn_uint j = 0; j < f.n_args; ++j, ++idx_j)
            lp::Kahan(gr[*idx_j], comp_mem[*idx_j], f.gr[j]);
        }
      }

      return out;
    };

    if(n_threads < 2L or !is_ele_func_thread_safe)
      return serial_version();

#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
    {
    double * gr_mem = get_thread_mem();
    if(comp_grad)
      std::fill(gr_mem, gr_mem + 2L * n_par, 0.);

    double &th_out  = *(gr_mem + 2L * n_par),
           &th_comp = *(gr_mem + 2L * n_par + 1L);
    th_out = 0;
    th_comp = 0;
#pragma omp for schedule(static)
    for(psqn_uint i = 0; i < n_funcs; ++i){
      auto &f = funcs[i];
      lp::Kahan(th_out, th_comp, f(val, comp_grad, caller));

      if(comp_grad){
        // add the gradient terms
        psqn_uint const * idx_j = f.indices();
        for(psqn_uint j = 0; j < f.n_args; ++j, ++idx_j)
          lp::Kahan(gr_mem + 2L * *idx_j, f.gr[j]);
      }
    }
    }

    // aggregate the results and possibly add to the gradient
    std::unique_ptr<double*[]> r_mem(new double*[n_threads]);
    for (int t = 0; t < n_threads; t++)
      r_mem[t] = get_thread_mem(t);

    double out(0.), out_comp(0.);
    {
      psqn_uint const inc = 2L * n_par;
      for(int t = 0; t < n_threads; t++){
        out      += r_mem[t][inc];
        out_comp += r_mem[t][inc + 1L];
      }
    }

    if(comp_grad){
      // reset
      std::fill(gr, gr + n_par, 0.);

      // fill the gradient. See
      //    https://stackoverflow.com/a/18016809/5861244
      for(psqn_uint i = 0; i < n_par; ++i){
        double val(0.), val_comp(0.);
        for(int t = 0; t < n_threads; t++){
          val      += *r_mem[t]++;
          val_comp += *r_mem[t]++;
        }

        gr[i] += val - val_comp;
      }
    }

    return out - out_comp;
#else
    return serial_version();
#endif
  }

  /***
   computes y <- y + B.x where B is the current Hessian approximation.
   @param val vector on the right-hand side.
   @param res output vector on the left-hand side.
   ***/
  void B_vec(double const * const PSQN_RESTRICT val,
             double * const PSQN_RESTRICT res) const noexcept {
    psqn_uint const n_funcs = funcs.size();

    // the serial version
    auto serial_version = [&]() -> void {
      double * const cmp_mem = get_thread_mem(),
             * const tmp_mem = cmp_mem + n_par;
      std::fill(cmp_mem, cmp_mem + n_par, 0.);

      for(psqn_uint i = 0; i < n_funcs; ++i){
        auto &f = funcs[i];

        std::fill(tmp_mem, tmp_mem + f.n_args, 0.);
        lp::mat_vec_dot(f.B, val, tmp_mem, f.n_args, f.indices());

        psqn_uint const * idx_j = f.indices();
        for(psqn_uint j = 0; j < f.n_args; ++j, ++idx_j)
          lp::Kahan(res[*idx_j], cmp_mem[*idx_j], tmp_mem[j]);
      }
    };

    if(n_threads < 2L){
      serial_version();
      return;
    }

#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
    {
    double * const gr_mem  = get_thread_mem(),
           * const tmp_mem = gr_mem + 2L * n_par;
    std::fill(gr_mem, gr_mem + 2L * n_par, 0.);

#pragma omp for schedule(static)
    for(psqn_uint i = 0; i < n_funcs; ++i){
      auto &f = funcs[i];
      std::fill(tmp_mem, tmp_mem + f.n_args, 0.);
      lp::mat_vec_dot(f.B, val, tmp_mem, f.n_args, f.indices());

      psqn_uint const * idx_j = f.indices();
      for(psqn_uint j = 0; j < f.n_args; ++j, ++idx_j)
        lp::Kahan(gr_mem + 2L * *idx_j, tmp_mem[j]);
    }
    }

    // fill the result. See
    //    https://stackoverflow.com/a/18016809/5861244
    std::unique_ptr<double*[]> r_mem(new double*[n_threads]);
    for (int t = 0; t < n_threads; t++)
      r_mem[t] = get_thread_mem(t);

    for(psqn_uint i = 0; i < n_par; ++i){
      double val(0.), val_comp(0.);
      for(int t = 0; t < n_threads; t++){
        val      += *r_mem[t]++;
        val_comp += *r_mem[t]++;
      }

      res[i] += val - val_comp;
    }

#else
    serial_version();
#endif
  }

  /**
    computes the diagonal of B.
   */
  void get_diag(double * x){
    std::fill(x, x + n_par, 0.);

    for(psqn_uint i = 0; i < funcs.size(); ++i){
      auto &f = funcs[i];

      // add the diagonal entries
      double const * b_diag = f.B;
      psqn_uint const n_args = f.n_args;
      psqn_uint const * idx_j = f.indices();
      for(psqn_uint j = 0; j < n_args; ++j, b_diag += j + 1, ++idx_j)
        x[*idx_j] += *b_diag;
    }
  }

  /***
    conjugate gradient method with diagonal preconditioning. Solves B.y = x
    where B is the Hessian approximation.
    @param tol convergence threshold.
    @param max_cg maximum number of conjugate gradient iterations.
    @param trace controls the amount of tracing information.
    @param pre_method preconditioning method.
   */
  bool conj_grad(double const * PSQN_RESTRICT x, double * PSQN_RESTRICT y,
                 double const tol, psqn_uint const max_cg,
                 int const trace, precondition const pre_method){
    return opt_internals->conj_grad(x, y, tol, max_cg, trace, pre_method);
  }

  /***
   performs line search to satisfy the Wolfe condition.
   @param f0 value of the functions at the current value.
   @param x0 value the function is evaluated.
   @param gr0 value of the current gradient.
   @param dir direction to search in.
   @param fnew the function value at the found solution.
   @param c1,c2 thresholds for Wolfe condition.
   @param strong_wolfe true if the strong Wolfe condition should be used.
   @param trace controls the amount of tracing information.

   x0 and gr0 contains the new value and gradient on return. The method
   returns false if the line search fails.
   */
  bool line_search(
      double const f0, double * PSQN_RESTRICT x0, double * PSQN_RESTRICT gr0,
      double * PSQN_RESTRICT dir, double &fnew, double const c1,
      double const c2, bool const strong_wolfe, int const trace){
    return opt_internals->line_search(f0, x0, gr0, dir, fnew, c1, c2,
                                      strong_wolfe, trace);
  }

  /***
   optimizes the partially separable function.
   @param val pointer to starting value. Set to the final estimate at the
   end.
   @param rel_eps relative convergence threshold.
   @param max_it maximum number of iterations.
   @param c1,c2 thresholds for Wolfe condition.
   @param use_bfgs bool for whether to use BFGS updates or SR1 updates.
   @param trace integer with info level passed to reporter.
   @param cg_tol threshold for conjugate gradient method.
   @param strong_wolfe true if the strong Wolfe condition should be used.
   @param max_cg maximum number of conjugate gradient iterations in each
   iteration. Use zero if there should not be a limit.
   @param pre_method preconditioning method.
   */
  optim_info optim
    (double * val, double const rel_eps, psqn_uint const max_it,
     double const c1, double const c2,
     bool const use_bfgs = true, int const trace = 0,
     double const cg_tol = .5, bool const strong_wolfe = true,
     psqn_uint const max_cg = 0,
     precondition const pre_method = precondition::diag){
    opt_internals->reset_counters();
    for(auto &f : funcs){
      f.reset();
      f.use_bfgs = use_bfgs;
    }

    std::unique_ptr<double[]> gr(new double[n_par]),
                             dir(new double[n_par]);

    // evaluate the gradient at the current value
    double fval = eval(val, gr.get(), true);
    for(auto &f : funcs)
      f.record();

    info_code info = info_code::max_it_reached;
    int n_line_search_fail = 0;
    for(psqn_uint i = 0; i < max_it; ++i){
      if(i % 10 == 0)
        if(interrupter::check_interrupt()){
          info = info_code::user_interrupt;
          break;
        }

      double const fval_old = fval;
      if(!conj_grad(gr.get(), dir.get(), cg_tol,
                    max_cg < 1 ? n_par : max_cg, trace,
                    pre_method)){
        info = info_code::conjugate_gradient_failed;
        Reporter::cg(trace, i, opt_internals->n_cg, false);
        break;
      } else
        Reporter::cg(trace, i, opt_internals->n_cg, true);

      for(double * d = dir.get(); d != dir.get() + n_par; ++d)
        *d *= -1;

      double const x1 = *val;
      if(!line_search(fval_old, val, gr.get(), dir.get(), fval, c1, c2,
                      strong_wolfe, trace)){
        info = info_code::line_search_failed;
        Reporter::line_search
          (trace, i, opt_internals->n_eval, opt_internals->n_grad, fval_old,
           fval, false, std::numeric_limits<double>::quiet_NaN(),
           const_cast<double const *>(val), 0L);
        if(++n_line_search_fail > 2)
          break;
      } else{
        n_line_search_fail = 0;
        Reporter::line_search
          (trace, i, opt_internals->n_eval, opt_internals->n_grad, fval_old, fval,
           true, (*val - x1) / *dir.get(), const_cast<double const *>(val), 0L);

      }

      bool const has_converged =
        abs(fval - fval_old) < rel_eps * (abs(fval_old) + rel_eps);
      if(has_converged){
        info = info_code::converged;
        break;
      }

      // update the Hessian and take another iteration
      if(n_line_search_fail < 1){
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif
        {
          double * const tmp_mem_use = get_thread_mem();
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
          for(psqn_uint i = 0; i < funcs.size(); ++i)
            funcs[i].update_Hes(tmp_mem_use);
        }
      } else
        for(psqn_uint i = 0; i < funcs.size(); ++i){
          funcs[i].reset();
          funcs[i].record();
        }
    }

    return { fval, info, opt_internals->n_eval, opt_internals->n_grad,
             opt_internals->n_cg };
  }

  /**
   returns a pointers to the element function objects. This may be useful if
   the element functions hold data.
   */
  std::vector<EFunc const *> get_ele_funcs() const {
    std::vector<EFunc const *> out;
    out.reserve(funcs.size());
    for(auto &o : funcs)
      out.emplace_back(&o.func);
    return out;
  }

  /***
   returns the current Hessian approximation.
   */
  void get_hess(double * const PSQN_RESTRICT hess) const {
    std::fill(hess, hess + n_par * n_par, 0.);

    for(auto &f : funcs){
      psqn_uint const n_args = f.n_args;
      double const * B = f.B;
      psqn_uint const * const idx = f.indices();
      for(psqn_uint j = 0; j < n_args; ++j) {
        psqn_uint const col_offset = idx[j] * n_par;
        for(psqn_uint i = 0; i <= j; ++i){
          if(i < j)
            hess[idx[i]         + col_offset] += *B;
          hess  [idx[i] * n_par + idx[j]    ] += *B++;
        }
      }
    }
  }

#ifdef PSQN_USE_EIGEN
private:
  /**
   the sparse matrix we will fill. We keep this as a member to avoid repeated
   memory allocations.
   */
  Eigen::SparseMatrix<double> sparse_B_mat;
  /// vector of triplets we use to fill sparse_B_mat
  std::vector<Eigen::Triplet<double> > sparse_B_mat_triplets;

  /// fills the sparse_B_mat
  inline void fill_sparse_B_mat(){
    // fill in the triplets
    sparse_B_mat_triplets.clear();

    // fill in the data
    for(auto &f : funcs){
      psqn_uint const n_args = f.n_args;
      double const * B = f.B;
      psqn_uint const * const idx = f.indices();
      for(psqn_uint j = 0; j < n_args; ++j)
        for(psqn_uint i = 0; i <= j; ++i){
          if(i < j)
            sparse_B_mat_triplets.emplace_back(idx[i], idx[j], *B   );
          sparse_B_mat_triplets  .emplace_back(idx[j], idx[i], *B++);
        }
    }

    sparse_B_mat.resize(n_par, n_par);
    sparse_B_mat.setFromTriplets(
      sparse_B_mat_triplets.begin(), sparse_B_mat_triplets.end());
  }

  Eigen::SparseMatrix<double> & get_hess_sparse_ref() {
    fill_sparse_B_mat();
    return sparse_B_mat;
  }

#endif // PSQN_USE_EIGEN

public:
  /***
   returns the current Hessian approximation as sparse matrix.
   */
#ifdef PSQN_USE_EIGEN
  Eigen::SparseMatrix<double> get_hess_sparse() {
    fill_sparse_B_mat();
    return sparse_B_mat;
  }
#else
  void get_hess_sparse() {
    throw_no_eigen_error();
  }
#endif // PSQN_USE_EIGEN
};

} // namespace PSQN

#endif

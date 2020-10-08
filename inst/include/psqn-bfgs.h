#ifndef PSQN_BFGS_H
#define PSQN_BFGS_H
#include <cstddef>
#include "psqn-misc.h"
#include "memory.h"
#include "lp.h"
#include "intrapolate.h"
#include <memory>
#include <algorithm>

namespace PSQN {
using std::abs;
using std::sqrt;

/** base problem class to pass to optimization method. */
class problem {
public:
  /** returns the number of parameters. */
  virtual size_t size() const = 0;
  /** returns evalutes the function at val. */
  virtual double func(double const *val) = 0;
  /** evaluates the function and compute the gradient. */
  virtual double grad(double * __restrict__ const val,
                      double * __restrict__       gr) = 0;
  virtual ~problem() = default;
};

/**
 minimizes a function.
 @param prob problem with function to be minimized.
 @param val starting value. Result on return.
 @param rel_eps relative convergence threshold.
 @param max_it maximum number of iterations.
 @param c1,c2 tresholds for Wolfe condition.
 @param strong_wolfe true if the strong Wolfe condition should be used.
 @param trace controls the amount of tracing information.
 */
template<class Reporter = dummy_reporter>
optim_info bfgs(
    problem &prob, double *val, double const rel_eps = .00000001,
    size_t const max_it = 100, double const c1 = .0001,
    double const c2 = .9, int const trace = 0L){
  // allocate the memory we need
  /* non-const due to
   *    https://www.mail-archive.com/gcc-bugs@gcc.gnu.org/msg531670.html */
  size_t n_ele = prob.size();
  std::unique_ptr<double[]>
    mem(new double[7 * n_ele + (n_ele * (n_ele + 1)) / 2]);
  double * __restrict__ const v_old  = mem.get(),
         * __restrict__ const gr     = v_old  + n_ele,
         * __restrict__ const gr_old = gr     + n_ele,
         * __restrict__ const s      = gr_old + n_ele,
         * __restrict__ const y      = s      + n_ele,
         * __restrict__ const wrk    = y      + n_ele,
         * __restrict__ const dir    = wrk    + n_ele,
         * __restrict__ const H      = dir    + n_ele;

  // initialize
  bool first_call(true);
  auto reset = [&](){
    std::fill(H, H + (n_ele * (n_ele + 1)) / 2, 0.);
    // set diagonals to 1
    double * h = H;
    for(size_t i = 0; i < n_ele; ++i, h += i + 1)
      *h = 1.;
    first_call = true;
  };
  reset();
  size_t n_eval(0),
         n_grad(0);
  double fval = prob.grad(val, gr);
  n_grad++;
  // declare lambda function to record parameter value and gradient
  auto record = [&](){
    lp::copy(v_old , val, n_ele);
    lp::copy(gr_old, gr , n_ele);
  };
  record();
  info_code info = info_code::max_it_reached;

  // declare lambda function to perform the BFGS update
  auto bfgs_update = [&](){
    lp::vec_diff(val, v_old , s, n_ele);

    // check if there is any changes in the input
    bool all_unchanged(true);
    for(size_t i = 0; i < n_ele; ++i)
      if(abs(s[i]) > abs(val[i]) *
          std::numeric_limits<double>::epsilon() * 100){
        all_unchanged = false;
        break;
      }

    if(!all_unchanged){
      lp::vec_diff(gr , gr_old, y, n_ele);

      double const s_y = lp::vec_dot(y, s, n_ele);
      if(s_y > 0){
        // TODO: implement damped BFGS?
        if(first_call){
          first_call = false;
          // make update on page 143
          double const scal = s_y / lp::vec_dot(y, n_ele);
          double *h = H;
          for(size_t i = 0; i < n_ele; ++i, h += i + 1)
            *h = scal;
        }

        std::fill(wrk, wrk + n_ele, 0.);
        lp::mat_vec_dot(H, y, wrk, n_ele);
        double const y_H_y = lp::vec_dot(y, wrk, n_ele);
        lp::bfgs_update(H, s, wrk, y_H_y, 1. / s_y, n_ele);

      } else
        // TODO: good idea?
        reset();

    } else
      // essentially no change in the input. Reset the Hessian approximation
      reset();

    record();
  };

  // declare lambda function to perform the line search
  auto line_search = [&](
      double const f0, double * __restrict__ x0, double * __restrict__ gr0,
      double * __restrict__ dir, double &fnew){
    double * const x_mem = wrk;

    // declare 1D functions
    auto psi = [&](double const alpha){
      for(size_t i = 0; i < n_ele; ++i)
        x_mem[i] = x0[i] + alpha * dir[i];
      ++n_eval;
      return prob.func(x_mem);
    };

    // returns the function value and the gradient
    auto dpsi = [&](double const alpha){
      for(size_t i = 0; i < n_ele; ++i)
        x_mem[i] = x0[i] + alpha * dir[i];
      ++n_grad;
      fnew = prob.grad(x_mem, gr0);
      return lp::vec_dot(gr0, dir, n_ele);
    };

    // the above at alpha = 0
    double dpsi_zero = lp::vec_dot(gr0, dir, n_ele);
    if(dpsi_zero > 0){
      // not a descent direction! Go the other way
      for(double * d = dir; d != dir + n_ele; ++d)
        *d *= -1;
      dpsi_zero *= -1;
    }

    constexpr double const NaNv = std::numeric_limits<double>::quiet_NaN();
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
        if(abs(dpsi_i) <= - c2 * dpsi_zero)
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
        lp::copy(x0, x_mem, n_ele);
        return out;
      }

      double const dpsi_i = dpsi(ai);
      Reporter::line_search_inner(trace, a_prev, ai, fi, false,
                                  dpsi_i, NaNv);

      if(abs(dpsi_i) <= - c2 * dpsi_zero){
        lp::copy(x0, x_mem, n_ele);
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
        lp::copy(x0, x_mem, n_ele);
        return out;
      }

      found_ok_prev = true;
      a_prev = ai;
      fold = fi;
    }

    return false;
  };

  // main loop
  for(size_t i = 0; i < max_it; ++i){
    double const fval_old = fval;
    std::fill(dir, dir + n_ele, 0.);
    lp::mat_vec_dot(H, gr, dir, n_ele);
    for(double * d = dir; d != dir + n_ele; ++d)
      *d *= -1;

    double const x1 = *val;
    constexpr size_t const n_print(100L);
    if(!line_search(fval_old, val, gr, dir, fval)){
      info = info_code::line_search_failed;
      Reporter::line_search
        (trace, i, n_eval, n_grad, fval_old, fval, false,
         std::numeric_limits<double>::quiet_NaN(),
         const_cast<double const *>(val),
         std::min(n_print, n_ele));
      break;
    }

    Reporter::line_search
      (trace, i, n_eval, n_grad, fval_old, fval, true,
       (*val - x1) / *dir, const_cast<double const *>(val),
       std::min(n_print, n_ele));

    bool const has_converged =
      abs(fval - fval_old) < rel_eps * (abs(fval_old) + rel_eps);
    if(has_converged){
      info = info_code::converged;
      break;
    }

    bfgs_update();
  }

  return { fval, info, n_eval, n_grad, 0 };
}
} // namespace PSQN

#endif

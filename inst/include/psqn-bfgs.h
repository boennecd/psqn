#ifndef PSQN_BFGS_H
#define PSQN_BFGS_H
#include "constant.h"
#include <cstddef>
#include "psqn-misc.h"
#include "lp.h"
#include "intrapolate.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

namespace PSQN {
using std::abs;
using std::sqrt;

namespace bfgs_defaults {
constexpr double rel_tol{.00000001},
                 c1{.0001},
                 c2{.9},
                 gr_tol{-1},
                 abs_tol{-1};
constexpr psqn_uint max_it{100};
constexpr int trace{0};
}

/** base problem class to pass to optimization method. */
class problem {
public:
  /** returns the number of parameters. */
  virtual psqn_uint size() const = 0;
  /** returns evaluates the function at val. */
  virtual double func(double const *val) = 0;
  /** evaluates the function and compute the gradient. */
  virtual double grad(double const * PSQN_RESTRICT val,
                      double       * PSQN_RESTRICT gr) = 0;
  virtual ~problem() = default;
};

/**
 returns the required working memory for bfgs.
 @param n_ele the number of parameters.
 */
constexpr psqn_uint bfgs_n_wmem(psqn_uint const n_ele){
  return 7 * n_ele + (n_ele * (n_ele + 1)) / 2;
}

/// overload that takes a problem
inline psqn_uint bfgs_n_wmem(problem const &prob){
  return bfgs_n_wmem(prob.size());
}

/**
 minimizes a function.
 @param prob problem with function to be minimized.
 @param val starting value. Result on return.
 @param mem working memory.
 @param rel_eps relative convergence threshold.
 @param max_it maximum number of iterations.
 @param c1,c2 thresholds for Wolfe condition.
 @param strong_wolfe true if the strong Wolfe condition should be used.
 @param trace controls the amount of tracing information.
 @param gr_tol convergence tolerance for the Euclidean norm of the gradient. A negative
 value yields no check.
 @param abs_tol absolute convergence threshold. A negative value yields not
 check.
 */
template<class Reporter = dummy_reporter,
         class interrupter = dummy_interrupter>
optim_info bfgs(
    problem &prob, double *val, double * const mem,
    double const rel_eps = bfgs_defaults::rel_tol,
    psqn_uint const max_it = bfgs_defaults::max_it,
    double const c1 = bfgs_defaults::c1, double const c2 = bfgs_defaults::c2,
    int const trace = bfgs_defaults::trace,
    double const gr_tol = bfgs_defaults::gr_tol,
    double const abs_tol = bfgs_defaults::abs_tol){
  // allocate the memory we need
  /* non-const due to
   *    https://www.mail-archive.com/gcc-bugs@gcc.gnu.org/msg531670.html */
  psqn_uint n_ele = prob.size();
  double * PSQN_RESTRICT const v_old  = mem,
         * PSQN_RESTRICT const gr     = v_old  + n_ele,
         * PSQN_RESTRICT const gr_old = gr     + n_ele,
         * PSQN_RESTRICT const s      = gr_old + n_ele,
         * PSQN_RESTRICT const y      = s      + n_ele,
         * PSQN_RESTRICT const wrk    = y      + n_ele,
         * PSQN_RESTRICT const dir    = wrk    + n_ele,
         * PSQN_RESTRICT const H      = dir    + n_ele;

  // initialize
  bool first_call(true);
  auto reset = [&]() -> void {
    std::fill(H, H + (n_ele * (n_ele + 1)) / 2, 0.);
    // set diagonals to 1
    double * h = H;
    for(psqn_uint i = 0; i < n_ele; ++i, h += i + 1)
      *h = 1.;
    first_call = true;
  };
  reset();
  psqn_uint n_eval(0),
         n_grad(0);
  double fval = prob.grad(const_cast<double const*>(val), gr);
  n_grad++;
  // declare lambda function to record parameter value and gradient
  auto record = [&]() -> void {
    lp::copy(v_old , val, n_ele);
    lp::copy(gr_old, gr , n_ele);
  };
  record();
  info_code info = info_code::max_it_reached;

  // declare lambda function to perform the BFGS update
  auto bfgs_update = [&]() -> void {
    lp::vec_diff(val, v_old , s, n_ele);

    // check if there is any changes in the input
    bool all_unchanged(true);
    for(psqn_uint i = 0; i < n_ele; ++i)
      if(abs(s[i]) > abs(val[i]) *
          std::numeric_limits<double>::epsilon() * 100){
        all_unchanged = false;
        break;
      }

    if(!all_unchanged){
      lp::vec_diff(gr , gr_old, y, n_ele);

      double const s_y = lp::vec_dot<false>(y, s, n_ele);
      if(s_y > 0){
        // TODO: implement damped BFGS?
        if(first_call){
          first_call = false;
          // make update on page 143
          double const scal = s_y / lp::vec_dot<false>(y, n_ele);
          double *h = H;
          for(psqn_uint i = 0; i < n_ele; ++i, h += i + 1)
            *h = scal;
        }

        std::fill(wrk, wrk + n_ele, 0.);
        lp::mat_vec_dot(H, y, wrk, n_ele);
        double const y_H_y = lp::vec_dot<false>(y, wrk, n_ele);
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
      double const f0, double * PSQN_RESTRICT x0, double * PSQN_RESTRICT gr0,
      double * PSQN_RESTRICT dir, double &fnew) -> bool {
    double * const x_mem = wrk;
    double const forg{fnew};

    // declare 1D functions
    auto psi = [&](double const alpha) -> double {
      for(psqn_uint i = 0; i < n_ele; ++i)
        x_mem[i] = x0[i] + alpha * dir[i];
      ++n_eval;
      return prob.func(const_cast<double const *>(x_mem));
    };

    // returns the function value and the gradient
    auto dpsi = [&](double const alpha) -> double {
      for(psqn_uint i = 0; i < n_ele; ++i)
        x_mem[i] = x0[i] + alpha * dir[i];
      ++n_grad;
      fnew = prob.grad(const_cast<double const *>(x_mem), gr0);
      return lp::vec_dot<false>(gr0, dir, n_ele);
    };

    // the above at alpha = 0
    double dpsi_zero = lp::vec_dot<false>(gr0, dir, n_ele);
    if(dpsi_zero > 0)
      // not a descent direction
      return false;

    constexpr double NaNv = std::numeric_limits<double>::quiet_NaN();
    auto zoom =
      [&](double a_low, double a_high, intrapolate &inter) -> bool {
        double f_low = psi(a_low);
        for(psqn_uint i = 0; i < 12L; ++i){
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
          if(abs(dpsi_i) <= - c2 * dpsi_zero)
            return true;

          if(dpsi_i * (a_high - a_low) >= 0.)
            a_high = a_low;

          a_low = ai;
          f_low = fi;
        }

        return false;
      };

    constexpr double mult_start{4};
    double fold(f0),
         a_prev(0),
             ai(1/mult_start);
    bool found_ok_prev = false,
           failed_once = false;
    double mult = mult_start;
    for(psqn_uint i = 0; i < 12L; ++i){
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
        if(out || (std::isfinite(fnew) && fnew < forg))
          lp::copy(x0, x_mem, n_ele);
        else
          fnew = forg; // we did not change x0
        return out;
      }

      double const dpsi_i = dpsi(ai);
      Reporter::line_search_inner(trace, a_prev, ai, fi, false,
                                  dpsi_i, NaNv);

      if(abs(dpsi_i) <= - c2 * dpsi_zero){
        lp::copy(x0, x_mem, n_ele);
        return true;
      }

      if(failed_once and fi < f0){
        lp::copy(x0, x_mem, n_ele);
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
        if(out || (std::isfinite(fnew) && fnew < forg))
          lp::copy(x0, x_mem, n_ele);
        else
          fnew = forg; // we did not change x0
        return out;
      }

      found_ok_prev = true;
      a_prev = ai;
      fold = fi;
    }

    return false;
  };

  // main loop
  int n_line_search_fail = 0;
  for(psqn_uint i = 0; i < max_it; ++i){
    if(i % 10 == 0)
      interrupter::check_interrupt();

    double const fval_old = fval;
    std::fill(dir, dir + n_ele, 0.);
    lp::mat_vec_dot(H, gr, dir, n_ele);
    for(double * d = dir; d != dir + n_ele; ++d)
      *d *= -1;

    double const x1 = *val;
    constexpr psqn_uint n_print(100L);
    if(!line_search(fval_old, val, gr, dir, fval)){
      info = info_code::line_search_failed;
      Reporter::line_search
        (trace, i, n_eval, n_grad, fval_old, fval, false,
         std::numeric_limits<double>::quiet_NaN(),
         const_cast<double const *>(val),
         std::min(n_print, n_ele));
      if(++n_line_search_fail > 2)
        break;
    } else {
      n_line_search_fail = 0;
      Reporter::line_search
        (trace, i, n_eval, n_grad, fval_old, fval, true,
         (*val - x1) / *dir, const_cast<double const *>(val),
         std::min(n_print, n_ele));
    }

    double const err{abs(fval - fval_old)};
    bool const passed_rel_tol{err < rel_eps * (abs(fval_old) + rel_eps)},
               passed_abs_tol{abs_tol < 0 || err < abs_tol},
               passed_gr_tol
      {gr_tol < 0 || lp::vec_dot<false>(gr, n_ele) < gr_tol * gr_tol};
    bool const has_converged
      {n_line_search_fail < 1 &&
        passed_rel_tol && passed_abs_tol && passed_gr_tol};
    if(has_converged){
      info = info_code::converged;
      break;
    }

    if(n_line_search_fail < 2)
      bfgs_update();
    else {
      reset();
      record();
    }

  }

  return { fval, info, n_eval, n_grad, 0 };
}

/// overload that allocates working memory
template<class Reporter = dummy_reporter,
         class interrupter = dummy_interrupter>
optim_info bfgs(
    problem &prob, double *val, double const rel_eps = bfgs_defaults::rel_tol,
    psqn_uint const max_it = bfgs_defaults::max_it,
    double const c1 = bfgs_defaults::c1, double const c2 = bfgs_defaults::c2,
    int const trace = bfgs_defaults::trace,
    double const gr_tol = bfgs_defaults::gr_tol,
    double const abs_tol = bfgs_defaults::abs_tol){
  std::vector<double> mem(bfgs_n_wmem(prob));
  return bfgs<Reporter, interrupter>
    (prob, val, mem.data(), rel_eps, max_it, c1, c2, trace, gr_tol, abs_tol);
}
} // namespace PSQN

#endif

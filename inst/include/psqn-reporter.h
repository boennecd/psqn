#ifndef PSQN_REPORTER_H
#define PSQN_REPORTER_H

#include <Rcpp.h>

namespace PSQN {
/** class used to print to the console during estimation */
struct R_reporter {
  static void cg(int const trace, size_t const iteration,
                 size_t const n_cg, bool const successful) {
    if(trace > 0)
      Rcpp::Rcout << "Conjugate gradient "
                  << (successful ? "succeeded" : "failed")
                  << " in itteration " << iteration + 1L
                  << '\n';

    if(trace > 2L)
      Rcpp::Rcout << "    " << n_cg
                  << " conjugate itterations have been used\n";
  }

  static void line_search
  (int const trace, size_t const iteration, size_t const n_eval,
   size_t const n_grad, double const fval_old,
   double const fval, bool const successful, double const step_size,
   double const *new_x, size_t const n_global) {
    if(trace > 0)
      Rcpp::Rcout << "Line search "
                  << (successful ? "succeeded" : "failed")
                  << '\n';

    if(trace > 1L)
      Rcpp::Rcout << "  New (old) function value is "
                  << fval << " (" << fval_old << ")\n";

    if(trace > 2L){
      Rcpp::Rcout << "    step size is " << step_size
                  << " and new global parameters are\n      ";
      for(size_t i = 0; i < n_global; ++i)
        Rcpp::Rcout << *(new_x + i) << " ";

      Rcpp::Rcout << "\n    " << n_eval
                  << " function evaluations and "
                  << n_grad << " gradient evaluations have been used\n";
    }

    if(trace > 0L)
      Rcpp::Rcout << '\n';
  }
};
} // namespace PSQN

#endif

// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "../inst/include/psqn.h"
#include <Rcpp.h>

using namespace Rcpp;

// psqn
List psqn(NumericVector par, Function fn, unsigned const n_ele_func, double const rel_eps, unsigned const max_it, unsigned const n_threads, double const cg_rel_eps, double const c1, double const c2, bool const use_bfgs, int const trace);
RcppExport SEXP _psqn_psqn(SEXP parSEXP, SEXP fnSEXP, SEXP n_ele_funcSEXP, SEXP rel_epsSEXP, SEXP max_itSEXP, SEXP n_threadsSEXP, SEXP cg_rel_epsSEXP, SEXP c1SEXP, SEXP c2SEXP, SEXP use_bfgsSEXP, SEXP traceSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type par(parSEXP);
    Rcpp::traits::input_parameter< Function >::type fn(fnSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type n_ele_func(n_ele_funcSEXP);
    Rcpp::traits::input_parameter< double const >::type rel_eps(rel_epsSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type max_it(max_itSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type n_threads(n_threadsSEXP);
    Rcpp::traits::input_parameter< double const >::type cg_rel_eps(cg_rel_epsSEXP);
    Rcpp::traits::input_parameter< double const >::type c1(c1SEXP);
    Rcpp::traits::input_parameter< double const >::type c2(c2SEXP);
    Rcpp::traits::input_parameter< bool const >::type use_bfgs(use_bfgsSEXP);
    Rcpp::traits::input_parameter< int const >::type trace(traceSEXP);
    rcpp_result_gen = Rcpp::wrap(psqn(par, fn, n_ele_func, rel_eps, max_it, n_threads, cg_rel_eps, c1, c2, use_bfgs, trace));
    return rcpp_result_gen;
END_RCPP
}

RcppExport SEXP run_testthat_tests();

static const R_CallMethodDef CallEntries[] = {
    {"_psqn_psqn", (DL_FUNC) &_psqn_psqn, 11},
    {"run_testthat_tests", (DL_FUNC) &run_testthat_tests, 0},
    {NULL, NULL, 0}
};

RcppExport void R_init_psqn(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}

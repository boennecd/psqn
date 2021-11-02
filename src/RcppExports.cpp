// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "../inst/include/psqn.h"
#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// psqn
List psqn(NumericVector par, SEXP fn, unsigned const n_ele_func, double const rel_eps, unsigned const max_it, unsigned const n_threads, double const c1, double const c2, bool const use_bfgs, int const trace, double const cg_tol, bool const strong_wolfe, SEXP env, int const max_cg, int const pre_method, IntegerVector const mask);
RcppExport SEXP _psqn_psqn(SEXP parSEXP, SEXP fnSEXP, SEXP n_ele_funcSEXP, SEXP rel_epsSEXP, SEXP max_itSEXP, SEXP n_threadsSEXP, SEXP c1SEXP, SEXP c2SEXP, SEXP use_bfgsSEXP, SEXP traceSEXP, SEXP cg_tolSEXP, SEXP strong_wolfeSEXP, SEXP envSEXP, SEXP max_cgSEXP, SEXP pre_methodSEXP, SEXP maskSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type par(parSEXP);
    Rcpp::traits::input_parameter< SEXP >::type fn(fnSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type n_ele_func(n_ele_funcSEXP);
    Rcpp::traits::input_parameter< double const >::type rel_eps(rel_epsSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type max_it(max_itSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type n_threads(n_threadsSEXP);
    Rcpp::traits::input_parameter< double const >::type c1(c1SEXP);
    Rcpp::traits::input_parameter< double const >::type c2(c2SEXP);
    Rcpp::traits::input_parameter< bool const >::type use_bfgs(use_bfgsSEXP);
    Rcpp::traits::input_parameter< int const >::type trace(traceSEXP);
    Rcpp::traits::input_parameter< double const >::type cg_tol(cg_tolSEXP);
    Rcpp::traits::input_parameter< bool const >::type strong_wolfe(strong_wolfeSEXP);
    Rcpp::traits::input_parameter< SEXP >::type env(envSEXP);
    Rcpp::traits::input_parameter< int const >::type max_cg(max_cgSEXP);
    Rcpp::traits::input_parameter< int const >::type pre_method(pre_methodSEXP);
    Rcpp::traits::input_parameter< IntegerVector const >::type mask(maskSEXP);
    rcpp_result_gen = Rcpp::wrap(psqn(par, fn, n_ele_func, rel_eps, max_it, n_threads, c1, c2, use_bfgs, trace, cg_tol, strong_wolfe, env, max_cg, pre_method, mask));
    return rcpp_result_gen;
END_RCPP
}
// psqn_bfgs
List psqn_bfgs(NumericVector par, SEXP fn, SEXP gr, double const rel_eps, unsigned int const max_it, double const c1, double const c2, int const trace, SEXP env);
RcppExport SEXP _psqn_psqn_bfgs(SEXP parSEXP, SEXP fnSEXP, SEXP grSEXP, SEXP rel_epsSEXP, SEXP max_itSEXP, SEXP c1SEXP, SEXP c2SEXP, SEXP traceSEXP, SEXP envSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type par(parSEXP);
    Rcpp::traits::input_parameter< SEXP >::type fn(fnSEXP);
    Rcpp::traits::input_parameter< SEXP >::type gr(grSEXP);
    Rcpp::traits::input_parameter< double const >::type rel_eps(rel_epsSEXP);
    Rcpp::traits::input_parameter< unsigned int const >::type max_it(max_itSEXP);
    Rcpp::traits::input_parameter< double const >::type c1(c1SEXP);
    Rcpp::traits::input_parameter< double const >::type c2(c2SEXP);
    Rcpp::traits::input_parameter< int const >::type trace(traceSEXP);
    Rcpp::traits::input_parameter< SEXP >::type env(envSEXP);
    rcpp_result_gen = Rcpp::wrap(psqn_bfgs(par, fn, gr, rel_eps, max_it, c1, c2, trace, env));
    return rcpp_result_gen;
END_RCPP
}
// psqn_generic
List psqn_generic(NumericVector par, SEXP fn, unsigned const n_ele_func, double const rel_eps, unsigned const max_it, unsigned const n_threads, double const c1, double const c2, bool const use_bfgs, int const trace, double const cg_tol, bool const strong_wolfe, SEXP env, int const max_cg, int const pre_method, IntegerVector const mask);
RcppExport SEXP _psqn_psqn_generic(SEXP parSEXP, SEXP fnSEXP, SEXP n_ele_funcSEXP, SEXP rel_epsSEXP, SEXP max_itSEXP, SEXP n_threadsSEXP, SEXP c1SEXP, SEXP c2SEXP, SEXP use_bfgsSEXP, SEXP traceSEXP, SEXP cg_tolSEXP, SEXP strong_wolfeSEXP, SEXP envSEXP, SEXP max_cgSEXP, SEXP pre_methodSEXP, SEXP maskSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type par(parSEXP);
    Rcpp::traits::input_parameter< SEXP >::type fn(fnSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type n_ele_func(n_ele_funcSEXP);
    Rcpp::traits::input_parameter< double const >::type rel_eps(rel_epsSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type max_it(max_itSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type n_threads(n_threadsSEXP);
    Rcpp::traits::input_parameter< double const >::type c1(c1SEXP);
    Rcpp::traits::input_parameter< double const >::type c2(c2SEXP);
    Rcpp::traits::input_parameter< bool const >::type use_bfgs(use_bfgsSEXP);
    Rcpp::traits::input_parameter< int const >::type trace(traceSEXP);
    Rcpp::traits::input_parameter< double const >::type cg_tol(cg_tolSEXP);
    Rcpp::traits::input_parameter< bool const >::type strong_wolfe(strong_wolfeSEXP);
    Rcpp::traits::input_parameter< SEXP >::type env(envSEXP);
    Rcpp::traits::input_parameter< int const >::type max_cg(max_cgSEXP);
    Rcpp::traits::input_parameter< int const >::type pre_method(pre_methodSEXP);
    Rcpp::traits::input_parameter< IntegerVector const >::type mask(maskSEXP);
    rcpp_result_gen = Rcpp::wrap(psqn_generic(par, fn, n_ele_func, rel_eps, max_it, n_threads, c1, c2, use_bfgs, trace, cg_tol, strong_wolfe, env, max_cg, pre_method, mask));
    return rcpp_result_gen;
END_RCPP
}

RcppExport SEXP run_testthat_tests(SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"_psqn_psqn", (DL_FUNC) &_psqn_psqn, 16},
    {"_psqn_psqn_bfgs", (DL_FUNC) &_psqn_psqn_bfgs, 9},
    {"_psqn_psqn_generic", (DL_FUNC) &_psqn_psqn_generic, 16},
    {"run_testthat_tests", (DL_FUNC) &run_testthat_tests, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_psqn(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}

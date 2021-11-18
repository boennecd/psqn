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
// psqn_hess
Eigen::SparseMatrix<double> psqn_hess(NumericVector val, SEXP fn, unsigned const n_ele_func, unsigned const n_threads, SEXP env, double const eps, double const scale, double const tol, unsigned const order);
RcppExport SEXP _psqn_psqn_hess(SEXP valSEXP, SEXP fnSEXP, SEXP n_ele_funcSEXP, SEXP n_threadsSEXP, SEXP envSEXP, SEXP epsSEXP, SEXP scaleSEXP, SEXP tolSEXP, SEXP orderSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type val(valSEXP);
    Rcpp::traits::input_parameter< SEXP >::type fn(fnSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type n_ele_func(n_ele_funcSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type n_threads(n_threadsSEXP);
    Rcpp::traits::input_parameter< SEXP >::type env(envSEXP);
    Rcpp::traits::input_parameter< double const >::type eps(epsSEXP);
    Rcpp::traits::input_parameter< double const >::type scale(scaleSEXP);
    Rcpp::traits::input_parameter< double const >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type order(orderSEXP);
    rcpp_result_gen = Rcpp::wrap(psqn_hess(val, fn, n_ele_func, n_threads, env, eps, scale, tol, order));
    return rcpp_result_gen;
END_RCPP
}
// psqn_aug_Lagrang
List psqn_aug_Lagrang(NumericVector par, SEXP fn, unsigned const n_ele_func, SEXP consts, unsigned const n_constraints, NumericVector multipliers, double const penalty_start, double const rel_eps, unsigned const max_it, unsigned const max_it_outer, double const violations_norm_thresh, unsigned const n_threads, double const c1, double const c2, double const tau, bool const use_bfgs, int const trace, double const cg_tol, bool const strong_wolfe, SEXP env, int const max_cg, int const pre_method, IntegerVector const mask);
RcppExport SEXP _psqn_psqn_aug_Lagrang(SEXP parSEXP, SEXP fnSEXP, SEXP n_ele_funcSEXP, SEXP constsSEXP, SEXP n_constraintsSEXP, SEXP multipliersSEXP, SEXP penalty_startSEXP, SEXP rel_epsSEXP, SEXP max_itSEXP, SEXP max_it_outerSEXP, SEXP violations_norm_threshSEXP, SEXP n_threadsSEXP, SEXP c1SEXP, SEXP c2SEXP, SEXP tauSEXP, SEXP use_bfgsSEXP, SEXP traceSEXP, SEXP cg_tolSEXP, SEXP strong_wolfeSEXP, SEXP envSEXP, SEXP max_cgSEXP, SEXP pre_methodSEXP, SEXP maskSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type par(parSEXP);
    Rcpp::traits::input_parameter< SEXP >::type fn(fnSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type n_ele_func(n_ele_funcSEXP);
    Rcpp::traits::input_parameter< SEXP >::type consts(constsSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type n_constraints(n_constraintsSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type multipliers(multipliersSEXP);
    Rcpp::traits::input_parameter< double const >::type penalty_start(penalty_startSEXP);
    Rcpp::traits::input_parameter< double const >::type rel_eps(rel_epsSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type max_it(max_itSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type max_it_outer(max_it_outerSEXP);
    Rcpp::traits::input_parameter< double const >::type violations_norm_thresh(violations_norm_threshSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type n_threads(n_threadsSEXP);
    Rcpp::traits::input_parameter< double const >::type c1(c1SEXP);
    Rcpp::traits::input_parameter< double const >::type c2(c2SEXP);
    Rcpp::traits::input_parameter< double const >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< bool const >::type use_bfgs(use_bfgsSEXP);
    Rcpp::traits::input_parameter< int const >::type trace(traceSEXP);
    Rcpp::traits::input_parameter< double const >::type cg_tol(cg_tolSEXP);
    Rcpp::traits::input_parameter< bool const >::type strong_wolfe(strong_wolfeSEXP);
    Rcpp::traits::input_parameter< SEXP >::type env(envSEXP);
    Rcpp::traits::input_parameter< int const >::type max_cg(max_cgSEXP);
    Rcpp::traits::input_parameter< int const >::type pre_method(pre_methodSEXP);
    Rcpp::traits::input_parameter< IntegerVector const >::type mask(maskSEXP);
    rcpp_result_gen = Rcpp::wrap(psqn_aug_Lagrang(par, fn, n_ele_func, consts, n_constraints, multipliers, penalty_start, rel_eps, max_it, max_it_outer, violations_norm_thresh, n_threads, c1, c2, tau, use_bfgs, trace, cg_tol, strong_wolfe, env, max_cg, pre_method, mask));
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
// psqn_aug_Lagrang_generic
List psqn_aug_Lagrang_generic(NumericVector par, SEXP fn, unsigned const n_ele_func, SEXP consts, unsigned const n_constraints, NumericVector multipliers, double const penalty_start, double const rel_eps, unsigned const max_it, unsigned const max_it_outer, double const violations_norm_thresh, unsigned const n_threads, double const c1, double const c2, double const tau, bool const use_bfgs, int const trace, double const cg_tol, bool const strong_wolfe, SEXP env, int const max_cg, int const pre_method, IntegerVector const mask);
RcppExport SEXP _psqn_psqn_aug_Lagrang_generic(SEXP parSEXP, SEXP fnSEXP, SEXP n_ele_funcSEXP, SEXP constsSEXP, SEXP n_constraintsSEXP, SEXP multipliersSEXP, SEXP penalty_startSEXP, SEXP rel_epsSEXP, SEXP max_itSEXP, SEXP max_it_outerSEXP, SEXP violations_norm_threshSEXP, SEXP n_threadsSEXP, SEXP c1SEXP, SEXP c2SEXP, SEXP tauSEXP, SEXP use_bfgsSEXP, SEXP traceSEXP, SEXP cg_tolSEXP, SEXP strong_wolfeSEXP, SEXP envSEXP, SEXP max_cgSEXP, SEXP pre_methodSEXP, SEXP maskSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type par(parSEXP);
    Rcpp::traits::input_parameter< SEXP >::type fn(fnSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type n_ele_func(n_ele_funcSEXP);
    Rcpp::traits::input_parameter< SEXP >::type consts(constsSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type n_constraints(n_constraintsSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type multipliers(multipliersSEXP);
    Rcpp::traits::input_parameter< double const >::type penalty_start(penalty_startSEXP);
    Rcpp::traits::input_parameter< double const >::type rel_eps(rel_epsSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type max_it(max_itSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type max_it_outer(max_it_outerSEXP);
    Rcpp::traits::input_parameter< double const >::type violations_norm_thresh(violations_norm_threshSEXP);
    Rcpp::traits::input_parameter< unsigned const >::type n_threads(n_threadsSEXP);
    Rcpp::traits::input_parameter< double const >::type c1(c1SEXP);
    Rcpp::traits::input_parameter< double const >::type c2(c2SEXP);
    Rcpp::traits::input_parameter< double const >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< bool const >::type use_bfgs(use_bfgsSEXP);
    Rcpp::traits::input_parameter< int const >::type trace(traceSEXP);
    Rcpp::traits::input_parameter< double const >::type cg_tol(cg_tolSEXP);
    Rcpp::traits::input_parameter< bool const >::type strong_wolfe(strong_wolfeSEXP);
    Rcpp::traits::input_parameter< SEXP >::type env(envSEXP);
    Rcpp::traits::input_parameter< int const >::type max_cg(max_cgSEXP);
    Rcpp::traits::input_parameter< int const >::type pre_method(pre_methodSEXP);
    Rcpp::traits::input_parameter< IntegerVector const >::type mask(maskSEXP);
    rcpp_result_gen = Rcpp::wrap(psqn_aug_Lagrang_generic(par, fn, n_ele_func, consts, n_constraints, multipliers, penalty_start, rel_eps, max_it, max_it_outer, violations_norm_thresh, n_threads, c1, c2, tau, use_bfgs, trace, cg_tol, strong_wolfe, env, max_cg, pre_method, mask));
    return rcpp_result_gen;
END_RCPP
}

RcppExport SEXP run_testthat_tests(SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"_psqn_psqn", (DL_FUNC) &_psqn_psqn, 16},
    {"_psqn_psqn_hess", (DL_FUNC) &_psqn_psqn_hess, 9},
    {"_psqn_psqn_aug_Lagrang", (DL_FUNC) &_psqn_psqn_aug_Lagrang, 23},
    {"_psqn_psqn_bfgs", (DL_FUNC) &_psqn_psqn_bfgs, 9},
    {"_psqn_psqn_generic", (DL_FUNC) &_psqn_psqn_generic, 16},
    {"_psqn_psqn_aug_Lagrang_generic", (DL_FUNC) &_psqn_psqn_aug_Lagrang_generic, 23},
    {"run_testthat_tests", (DL_FUNC) &run_testthat_tests, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_psqn(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}

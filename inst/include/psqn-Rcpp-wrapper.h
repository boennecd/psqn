#ifndef PSQN_RCPP_WRAPPER_H
#define PSQN_RCPP_WRAPPER_H

#ifdef PSQN_USE_EIGEN
#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppEigen.h>

#else
#include <Rcpp.h>
#endif

#endif

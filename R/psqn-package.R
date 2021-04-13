#' @importFrom Rcpp sourceCpp
#' @useDynLib psqn, .registration = TRUE
NULL

#' @title psqn: Partially Separable Quasi-Newton
#'
#' @description
#' The main methods in the psqn package are the \code{\link{psqn}} and
#' \code{\link{psqn_generic}} function.
#' Notice that it is also possible to use the package from C++. This may
#' yield a large reduction in the computation time. See the vignette for
#' details e.g. by calling \code{vignette("psqn", package = "psqn")}.
#' A brief introduction is provided in the "quick-intro" vignette
#' (see \code{vignette("quick-intro", package = "psqn")}).
#'
#' This package is fairly new. Thus, results may change and
#' contributions and feedback is much appreciated.
#' @aliases psqn-package
"_PACKAGE"

context("psqn_generic and the C++ interface works")

# parameters for the simulation
set.seed(1)
K <- 20L
n <- 5L * K

# # simulate the data
# truth_limit <- runif(K, -1, 1)
# dat <- replicate(
#   n, {
#     # sample the indices
#     n_samp <- sample.int(5L, 1L) + 1L
#     indices <- sort(sample.int(K, n_samp))
#
#     # sample the outcome, y, and return
#     list(y = rpois(1, exp(sum(truth_limit[indices]))),
#          indices = indices)
#   }, simplify = FALSE)
#
# # we need each variable to be present at least once
# stopifnot(length(unique(unlist(
#   lapply(dat, `[`, "indices")
# ))) == K) # otherwise we need to change the code
#
# saveRDS(dat, "GLM-generic-data.RDS")
dat <- readRDS("GLM-generic-data.RDS")

test_that("the R and C++ interface gives the same and correct result", {
  # test the R interface
  r_func <- function(i, par, comp_grad){
    z <- dat[[i]]
    if(length(par) == 0L)
      return(z$indices)

    eta <- sum(par)
    exp_eta <- exp(eta)
    out <- -z$y * eta + exp_eta
    if(comp_grad)
      attr(out, "grad") <- rep(-z$y + exp_eta, length(z$indices))
    out
  }

  R_res <- psqn_generic(
    par = numeric(K), fn = r_func, n_ele_func = length(dat), c1 = 1e-4, c2 = .1,
    trace = 0L, rel_eps = 1e-9, max_it = 1000L, env = environment())

  expect_known_value(R_res[c("par", "value")], "psqn_generic-glm-res.RDS")

  idx_mask <- c(2L, 5L, 19L)
  R_res_mask <- psqn_generic(
    par = numeric(K), fn = r_func, n_ele_func = length(dat), c1 = 1e-4, c2 = .1,
    trace = 0L, rel_eps = 1e-9, max_it = 1000L, env = environment(),
    mask = idx_mask)
  expect_known_value(R_res_mask[c("par", "value")],
                     "psqn_generic-glm-res-mask.RDS")

  # check for an error message
  expect_error(
    R_res <- psqn_generic(
      par = numeric(K), fn = r_func, n_ele_func = length(dat), c1 = 1e-4, c2 = .1,
      trace = 0L, rel_eps = 1e-9, max_it = 1000L, env = environment(),
      pre_method = 3L),
    "there is no custom preconditioner")

  # check that the C++ version gives the same
  skip_if_not_installed("Matrix")
  skip_on_macOS()
  skip_on_cran()
  library(Rcpp)
  library(Matrix)
  reset_info <- compile_cpp_file("generic_example.cpp")
  on.exit(reset_compile_cpp_file(reset_info), add = TRUE)
  setwd(reset_info$old_wd)

  cpp_arg <- lapply(dat, function(x){
    x$indices <- x$indices - 1L # C++ needs zero-based indices
    x
  })
  ptr <- get_generic_ex_obj(cpp_arg, max_threads = 2L)
  Cpp_res <- optim_generic_ex(
    val = numeric(K), ptr = ptr, rel_eps = 1e-9, max_it = 1000L,
    n_threads = 1L, c1 = 1e-4, c2 = .1, trace = 0L, cg_tol = .5)
  expect_equal(Cpp_res, R_res)

  # the Hessian yields the same result
  hess_ress_sparse <- get_sparse_Hess_approx_generic(ptr)
  expect_known_value(hess_ress_sparse, "psqn_generic-glm-hess-res.RDS")
  hess_ress <- get_Hess_approx_generic(ptr)
  expect_equal(as.matrix(hess_ress_sparse), hess_ress,
               check.attributes = FALSE)

  # we get the same with more threads
  Cpp_res <- optim_generic_ex(
    val = numeric(K), ptr = ptr, rel_eps = 1e-9, max_it = 1000L,
    n_threads = 2L, c1 = 1e-4, c2 = .1, trace = 0L, cg_tol = .5)
  expect_equal(Cpp_res, R_res)

  set_masked(ptr, idx_mask)
  Cpp_res_mask <- optim_generic_ex(
    val = numeric(K), ptr = ptr, rel_eps = 1e-9, max_it = 1000L,
    n_threads = 2L, c1 = 1e-4, c2 = .1, trace = 0L, cg_tol = .5)
  expect_equal(Cpp_res_mask, R_res_mask)
  clear_masked(ptr)

  # the gradient tolerance works
  gr_tol <- 1e-6
  Cpp_res <- optim_generic_ex(
    val = numeric(K), ptr = ptr, rel_eps = 1, max_it = 1000L,
    n_threads = 2L, c1 = 1e-4, c2 = .1, trace = 0L, cg_tol = .5,
    gr_tol = gr_tol)
  expect_lt(sqrt(sum(grad_generic_ex(Cpp_res$par, ptr, 1)^2)),
            gr_tol)

  # we the right result with other preconditioners
  for(i in 0:2){
    Cpp_res <- optim_generic_ex(
      val = numeric(K), ptr = ptr, rel_eps = 1e-9, max_it = 1000L,
      n_threads = 2L, c1 = 1e-4, c2 = .1, trace = 0L, cg_tol = .5,
      pre_method = i)
    expect_equal(Cpp_res$value, R_res$value, info = i)
    expect_equal(Cpp_res$par, R_res$par, info = i,
                 tolerance = 4 * sqrt(1e-9))

    R_res_new <- psqn_generic(
      par = numeric(K), fn = r_func, n_ele_func = length(dat), c1 = 1e-4,
      c2 = .1, trace = 0L, rel_eps = 1e-9, max_it = 1000L, env = environment(),
      pre_method = i)
    expect_equal(R_res_new$value, R_res$value, info = i)
    expect_equal(R_res_new$par, R_res$par, info = i,
                 tolerance = 4 * sqrt(1e-9))
  }

  # test that we get the same when we do not use Kahan summation algorithm
  (function(){
    reset_info <- compile_cpp_file("generic_example.cpp",
                                   "generic_example-Kahan.cpp",
                                   do_compile = FALSE)
    on.exit(reset_compile_cpp_file(reset_info), add = TRUE)

    old_lines <- readLines("generic_example-Kahan.cpp")
    tmp_file_con <- file("generic_example-Kahan.cpp")
    writeLines(
      c("#define PSQN_NO_USE_KAHAN", old_lines),
      tmp_file_con)
    close(tmp_file_con)
    sourceCpp("generic_example-Kahan.cpp")
    setwd(reset_info$old_wd)

    Cpp_res <- optim_generic_ex(
      val = numeric(K), ptr = ptr, rel_eps = 1e-9, max_it = 1000L,
      n_threads = 1L, c1 = 1e-4, c2 = .1, trace = 0L, cg_tol = .5)
    expect_equal(Cpp_res$value, R_res$value)

    # check for an error message
    expect_error(
      optim_generic_ex(
        val = numeric(K), ptr = ptr, rel_eps = 1e-9, max_it = 1000L,
        n_threads = 1L, c1 = 1e-4, c2 = .1, trace = 0L, cg_tol = .5,
        pre_method = 3L),
      "there is no custom preconditioner")
  })()
})

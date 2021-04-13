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
# saveRDS(dat, "GLMM-generic-data.RDS")
dat <- readRDS("GLMM-generic-data.RDS")

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

  expect_known_value(R_res[c("par", "value")], "psqn_generic-glmm-res.RDS")

  # check that the C++ version gives the same
  skip_if_not_installed("Rcpp")
  library(Rcpp)
  sourceCpp(system.file("generic_example.cpp", package = "psqn"))
  cpp_arg <- lapply(dat, function(x){
    x$indices <- x$indices - 1L # C++ needs zero-based indices
    x
  })
  ptr <- get_generic_ex_obj(cpp_arg, max_threads = 4L)
  Cpp_res <- optim_generic_ex(
    val = numeric(K), ptr = ptr, rel_eps = 1e-9, max_it = 1000L,
    n_threads = 1L, c1 = 1e-4, c2 = .1, trace = 0L, cg_tol = .5)
  expect_equal(Cpp_res, R_res)

  Cpp_res <- optim_generic_ex(
    val = numeric(K), ptr = ptr, rel_eps = 1e-9, max_it = 1000L,
    n_threads = 2L, c1 = 1e-4, c2 = .1, trace = 0L, cg_tol = .5)
  expect_equal(Cpp_res, R_res)

  # test that we get the same when we do not use Kahan summation algorithm
  skip_on_cran()
  tmp_file <- file.path(system.file(package = "psqn"),
                        "temp-file-to-be-compiled.cpp")
  (function(){
    on.exit({
      # clean up
      fs <- list.files(system.file(package = "psqn"), full.names = TRUE)
      to_delete <- grepl("temp-file-to-be-compiled.*", fs)
      sapply(fs[to_delete], unlink)
    })

    tmp_file_con <- file(tmp_file)
    writeLines(
      c("#define PSQN_NO_USE_KAHAN",
        readLines(system.file("generic_example.cpp", package = "psqn"))),
      tmp_file_con)
    close(tmp_file_con)
    sourceCpp(tmp_file)

    Cpp_res <- optim_generic_ex(
      val = numeric(K), ptr = ptr, rel_eps = 1e-9, max_it = 1000L,
      n_threads = 1L, c1 = 1e-4, c2 = .1, trace = 0L, cg_tol = .5)
    expect_equal(Cpp_res$value, R_res$value)
  })()
})

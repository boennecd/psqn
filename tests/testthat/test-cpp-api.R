context("Testing C++ headers from R")

# assign model parameters and number of random effects and fixed effects
q <- 4
p <- 5
beta <- sqrt((1:p) / sum(1:p))
Sigma <- diag(q)

# # simulate data set
# n_clusters <- 4L
# set.seed(1)
#
# sim_dat <- replicate(n_clusters, {
#   n_members <- sample.int(20L, 1L) + 2L
#   X <- matrix(runif(p * n_members, -sqrt(6 / 2), sqrt(6 / 2)),
#               p)
#   u <- drop(rnorm(q) %*% chol(Sigma))
#   Z <- matrix(runif(q * n_members, -sqrt(6 / 2 / q), sqrt(6 / 2 / q)),
#               q)
#   eta <- drop(beta %*% X + u %*% Z)
#   y <- as.numeric((1 + exp(-eta))^(-1) > runif(n_members))
#
#   list(X = X, Z = Z, y = y, u = u, Sigma_inv = solve(Sigma))
# }, simplify = FALSE)
# saveRDS(sim_dat, file.path("tests", "testthat", "mixed-logit.RDS"))
f <- file.path("tests", "testthat", "mixed-logit.RDS")
if(!file.exists(f))
  f <- "mixed-logit.RDS"
sim_dat <- readRDS(f)

test_that("mixed logit model gives the same", {
  skip_if_not_installed("Rcpp")
  skip_if_not_installed("RcppArmadillo")

  Rcpp::sourceCpp(system.file("mlogit-ex.cpp", package = "psqn"))
  optimizer <- get_mlogit_optimizer(sim_dat)

  val <- c(beta, sapply(sim_dat, function(x) x$u))
  fv <- eval_mlogit(val = val, ptr = optimizer)
  expect_equal(fv, 35.248256794375)

  gr <- grad_mlogit(val = val, ptr = optimizer)
  expect_equal(
    gr,
    structure(c(-3.09880891801443, -3.11919456739973, -1.26969269606313,
                5.50412085634227, -5.07396341960751, 0.407174373086653, 0.590879440022872,
                0.122784612825495, 0.222039129397368, -0.335740954704027, 0.967036607234169,
                -2.05362226641706, -0.517311694420387, -2.33859175587387, 0.911175257815629,
                -1.80538928384093, -0.81631781729293, 3.14749812772018, -0.894143195465396,
                0.647354085121652, 0.344172484580979), value = 35.248256794375))

  opt <- optim_mlogit(val = val, ptr = optimizer, rel_eps = 1e-8,
                      max_it = 100L)

  expect_equal(
    opt,
    list(par = c(0.64728530110947, 0.773194017021601, 0.657000534901989,
                 0.119680984770149, 0.855866285979022, -0.484305536586791, 0.412501613805294,
                 0.766995560437863, 0.422857367872878, 0.605158943583224, -0.783730421012958,
                 0.0961764408437856, -0.922959439713509, -0.483522109084062, 0.438228261355513,
                 -0.493282831674971, 0.0274308256734968, 0.426837683845752, 0.0282555513891497,
                 -0.180702023732985, -0.150661553719773), value = 25.121501499614,
         info = 0L, counts = c(`function` = 12, gradient = 11, n_cg = 132
         ), convergence = TRUE))
})

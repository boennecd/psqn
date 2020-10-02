context("Testing R interface for psqn")

# assign model parameters and number of random effects and fixed effects
q <- 4
p <- 5
beta <- sqrt((1:p) / sum(1:p))
Sigma <- diag(q)

f <- file.path("tests", "testthat", "mixed-logit.RDS")
if(!file.exists(f))
  f <- "mixed-logit.RDS"
sim_dat <- readRDS(f)

n_clusters <- length(sim_dat)

r_func <- function(i, par, comp_grad){
  dat <- sim_dat[[i]]
  X <- dat$X
  Z <- dat$Z
  y <- dat$y
  Sigma_inv <- dat$Sigma_inv

  if(length(par) < 1)
    # requested the dimension the parameter
    return(c(global_dim = NROW(dat$X), private_dim = NROW(dat$Z)))

  beta <- par[1:p]
  uhat <- par[1:q + p]
  eta <- drop(beta %*% X + uhat %*% Z)
  exp_eta <- exp(eta)

  out <- -drop(y %*% eta) + sum(log(1 + exp_eta)) +
    drop(uhat %*% Sigma_inv %*% uhat) / 2
  if(comp_grad){
    d_eta <- -y + exp_eta / (1 + exp_eta)
    grad <- c(rowSums(X * rep(d_eta, each = p)),
              rowSums(Z * rep(d_eta, each = q)) + Sigma_inv %*% uhat)
    attr(out, "grad") <- grad
  }

  out
}

test_that("mixed logit model gives the same", {
  rel_eps <- sqrt(.Machine$double.eps)
  val <- c(beta, sapply(sim_dat, function(x) x$u))
  opt <- psqn(
    par = val, fn = r_func, n_ele_func = n_clusters, rel_eps = rel_eps,
    max_it = 100L, cg_rel_eps = 1e-5, c1 = 1e-4, c2 = .9, n_threads = 1L)
  opt_res <- list(par = c(0.64728530110947, 0.773194017021601, 0.657000534901989,
                          0.119680984770149, 0.855866285979022, -0.484305536586791, 0.412501613805294,
                          0.766995560437863, 0.422857367872878, 0.605158943583224, -0.783730421012958,
                          0.0961764408437856, -0.922959439713509, -0.483522109084062, 0.438228261355513,
                          -0.493282831674971, 0.0274308256734968, 0.426837683845752, 0.0282555513891497,
                          -0.180702023732985, -0.150661553719773), value = 25.121501499614,
                  info = 0L, counts = c(`function` = 12, gradient = 11, n_cg = 132
                  ), convergence = TRUE)
  tol <- .Machine$double.eps^(1/3)
  do_check <- !names(opt) %in% "counts"
  expect_equal(opt[do_check], opt_res[do_check], tolerance = tol)

  opt <- psqn(
    par = val, fn = r_func, n_ele_func = n_clusters, rel_eps = rel_eps,
    max_it = 100L, cg_rel_eps = 1e-5, c1 = 1e-4, c2 = .9, n_threads = 2L)
  expect_equal(opt[do_check], opt_res[do_check], tolerance = tol)

  opt <- psqn(
    par = val, fn = r_func, n_ele_func = n_clusters, rel_eps = rel_eps,
    max_it = 100L, cg_rel_eps = 1e-5, c1 = 1e-4, c2 = .9, n_threads = 1L,
    use_bfgs = FALSE)
  opt_res <- list(par = c(0.647275645738644, 0.773166405224482, 0.65701868587754,
                          0.119620392079857, 0.855935194552437, -0.484320891227118, 0.412483498667473,
                          0.766965424427122, 0.422820956122594, 0.605160873384254, -0.783750631065262,
                          0.0961498616334638, -0.922989688317959, -0.483529642590115, 0.438265028750743,
                          -0.493282887150997, 0.0274560092292598, 0.42686782768471, 0.0283980041013476,
                          -0.180765748491386, -0.150514195778015), value = 25.1215014472174,
                  info = 0L, counts = c(`function` = 21, gradient = 14, n_cg = 121
                  ), convergence = TRUE)
  expect_equal(opt[do_check], opt_res[do_check], tolerance = tol)

  opt <- psqn(
    par = val, fn = r_func, n_ele_func = n_clusters, rel_eps = rel_eps,
    max_it = 100L, cg_rel_eps = 1e-5, c1 = 1e-4, c2 = .9, n_threads = 2L,
    use_bfgs = FALSE)
  expect_equal(opt[do_check], opt_res[do_check], tolerance = tol)
})

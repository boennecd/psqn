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
    max_it = 100L, c1 = 1e-4, c2 = .9, n_threads = 1L)
  opt_res <- list(par = c(0.647310437469711, 0.773205671911178, 0.656983834118283,
                          0.119681528415682, 0.855849653168437, -0.484311298187747, 0.412484752266496,
                          0.766994317977443, 0.422874453513229, 0.605169744625452, -0.783701817869368,
                          0.0961948999677121, -0.922934757392211, -0.483542761956257, 0.438193243670319,
                          -0.493309514536335, 0.0274193762771718, 0.426816737611666, 0.0281704108821128,
                          -0.180877909582702, -0.150512019701299), value = 25.1215015278071,
                  info = 0L, counts = c(`function` = 12, gradient = 11, n_cg = 40
                  ), convergence = TRUE)
  tol <- sqrt(rel_eps)
  do_check <- !names(opt) %in% "counts"
  expect_equal(opt[do_check], opt_res[do_check], tolerance = tol)

  opt <- psqn(
    par = val, fn = r_func, n_ele_func = n_clusters, rel_eps = rel_eps,
    max_it = 100L, c1 = 1e-4, c2 = .9, n_threads = 2L)
  expect_equal(opt[do_check], opt_res[do_check], tolerance = tol)

  opt <- psqn(
    par = val, fn = r_func, n_ele_func = n_clusters, rel_eps = rel_eps,
    max_it = 100L, c1 = 1e-4, c2 = .9, n_threads = 1L,
    use_bfgs = FALSE)
  opt_res <- list(par = c(0.647312399593272, 0.773180471730626, 0.6570617410787,
                          0.119630546874293, 0.85595502016993, -0.484316891358561, 0.412488738456401,
                          0.766975554121974, 0.422825073677682, 0.605153277060966, -0.783774377322635,
                          0.0961329299272066, -0.923001177401597, -0.483537409424477, 0.438262748069446,
                          -0.493258780253669, 0.0274569314927997, 0.426952405357697, 0.0284032062298845,
                          -0.18070846603226, -0.150657495438486), value = 25.1215014846691,
                  info = 0L, counts = c(`function` = 18, gradient = 13, n_cg = 50
                  ), convergence = TRUE)
  expect_equal(opt[do_check], opt_res[do_check], tolerance = tol)

  opt <- psqn(
    par = val, fn = r_func, n_ele_func = n_clusters, rel_eps = rel_eps,
    max_it = 100L, c1 = 1e-4, c2 = .9, n_threads = 2L,
    use_bfgs = FALSE)
  expect_equal(opt[do_check], opt_res[do_check], tolerance = tol)
})

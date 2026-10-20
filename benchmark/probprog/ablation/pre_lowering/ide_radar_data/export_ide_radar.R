# Export the fitted IDE radar model of STRbook Lab 5.2 (IDE_Radar_results.rda, spatially
# invariant kernel) into plain JSON, and evaluate the IDE package's own negative log
# likelihood (joint sparse precision, construct_Q + negloglik formula from IDEfunctions.R)
# at the fitted parameters and at extra points, with Matrix's sparse Cholesky in place of
# sparseinv::cholPermute.  Usage: Rscript export_ide_radar.R <rda> <IDEfunctions.R> <radar_df.rda> <out.json>
suppressMessages({library(Matrix); library(sp); library(jsonlite)})
a <- commandArgs(TRUE)
e <- new.env(); load(a[1], envir = e); obj <- get(ls(e)[1], envir = e)
f <- new.env(); load(a[3], envir = f); radar_df <- get("radar_df", envir = f)
IDEm <- obj$IDEmodel
env <- environment(IDEm$get)
# helper functions of the IDE package (pure Matrix code): construct_Q, Zeromat, logdet
src <- new.env(); sys.source(a[2], envir = src, keep.source = FALSE)
construct_Q <- get("construct_Q", envir = src); environment(construct_Q) <- globalenv()
Zeromat <- get("Zeromat", envir = src); assign("Zeromat", Zeromat, envir = globalenv())

Mf_env <- environment(get("Mfun", envir = env))
PHI <- as.matrix(get("PHI", envir = Mf_env))          # grid x r
GRAM <- as.matrix(get("GRAM", envir = Mf_env))
s <- IDEm$get("s")
r <- IDEm$get("r"); Tn <- IDEm$get("T"); m <- IDEm$get("m")
PHI_obs_list <- get("PHI_obs_list", envir = env)
nobs_t <- sapply(PHI_obs_list, nrow)
Z <- as.numeric(IDEm$get("Z"))
X_obs <- as.matrix(IDEm$get("X_obs"))
tp <- IDEm$get("time_points")

# the observation blocks of the data times must be identical (fixed radar pixel grid)
dat_idx <- which(nobs_t > 0)
P1 <- as.matrix(PHI_obs_list[[dat_idx[1]]])
same <- all(sapply(dat_idx, function(i) max(abs(as.matrix(PHI_obs_list[[i]]) - P1)) == 0))
cat("data time indices:", dat_idx, " nobs per time:", unique(nobs_t[dat_idx]), " identical PHI blocks:", same, "\n")

# kernel matrix and propagator exactly as construct_M / construct_kernel (spatially invariant)
Mfun_R <- function(k) {
  sg <- s$s_grid_mat
  sh <- sweep(sg, 2, c(k[3], k[4]), "+")
  D2 <- outer(sh[,1], sg[,1], "-")^2 + outer(sh[,2], sg[,2], "-")^2
  Kmat <- k[1] * exp(-D2 / k[2])
  solve(GRAM) %*% crossprod(t(Kmat) %*% PHI, PHI) * s$area^2
}
kfit <- unlist(IDEm$get("k"))
Mstored <- as.matrix(IDEm$get("M"))
cat("max |M_R - M_stored| =", max(abs(Mfun_R(kfit) - Mstored)), "\n")

# IDE package negloglik (IDEfunctions.R) with Matrix Cholesky in place of cholPermute.
nll_ide <- function(k, s2eps, s2eta, beta = NULL) {
  M <- Mfun_R(k)
  Q_eta <- 1/s2eta * Diagonal(r)
  Q_eps <- 1/s2eps * Diagonal(m)
  PHI_obs <- do.call("bdiag", PHI_obs_list)
  Q <- construct_Q(Q_eta, M, Tn)
  Qpost <- crossprod(chol(Q_eps) %*% PHI_obs) + Q
  Qpost <- forceSymmetric(Qpost)
  if (is.null(beta)) {   # GLS update_betahat
    Qps <- function(y) solve(Qpost, y)
    tPHIQepsZ <- t(PHI_obs) %*% Q_eps %*% Z
    tXQeps <- t(X_obs) %*% Q_eps
    tXQepsPHI <- tXQeps %*% PHI_obs
    Part1 <- solve(tXQeps %*% X_obs - tXQepsPHI %*% Qps(t(tXQepsPHI)))
    Part2 <- tXQeps %*% Z - tXQepsPHI %*% Qps(tPHIQepsZ)
    beta <- as.numeric(Part1 %*% Part2)
  }
  Ztilde <- Z - X_obs %*% beta
  alphahat <- solve(Qpost, t(PHI_obs) %*% Q_eps %*% Ztilde)
  ldQ <- as.numeric(determinant(forceSymmetric(Q), logarithm = TRUE)$modulus)
  ldQpost <- as.numeric(determinant(Qpost, logarithm = TRUE)$modulus)
  ldQeps <- m * log(1/s2eps)
  nll <- -(0.5*ldQ + 0.5*ldQeps - 0.5*ldQpost - 0.5*as.numeric(t(Ztilde) %*% (Q_eps %*% Ztilde)) +
           0.5*as.numeric(t(Ztilde) %*% Q_eps %*% PHI_obs %*% alphahat))
  list(nll = nll, beta = beta, loglik = -nll - 0.5*m*log(2*pi))
}
opt <- obj$optim_results$optim
fit <- nll_ide(kfit, IDEm$get("sigma2_eps"), IDEm$get("sigma2_eta"))
cat(sprintf("negloglik at fit (GLS beta %.10f): %.10f   DEoptim bestval %.10f\n", fit$beta, fit$nll, opt$bestval))
# extra reference points (fixed beta) for the joint-density check
pts <- list(
  list(k = kfit, s2eps = IDEm$get("sigma2_eps"), s2eta = IDEm$get("sigma2_eta"), beta = fit$beta),
  list(k = c(0.09, 4.0, -3.0, -1.0), s2eps = 35.0, s2eta = 5.0, beta = 1.5),
  list(k = c(0.2, 1.5, -7.0, -2.5), s2eps = 20.0, s2eta = 10.0, beta = 0.0))
refs <- lapply(pts, function(p) { v <- nll_ide(p$k, p$s2eps, p$s2eta, p$beta)
  cat(sprintf("  ref point k=(%s) s2eps=%g s2eta=%g beta=%g  loglik %.10f\n",
      paste(p$k, collapse=","), p$s2eps, p$s2eta, p$beta, v$loglik))
  c(p, list(loglik = v$loglik, ide_negloglik = v$nll)) })

out <- list(
  source = "STRbook::IDE_Radar_results (Lab 5.2, IDE package, spatially invariant kernel), radar_STIDF",
  r = r, T = Tn, m = m, grid_size = 41, n_grid = nrow(s$s_grid_mat), area = s$area,
  coordnames = IDEm$get("coordnames"),
  time_points = format(tp, "%Y-%m-%d %H:%M:%S"), data_time_index = dat_idx - 1,
  nobs_per_time = unique(nobs_t[dat_idx]),
  s_grid = s$s_grid_mat, PHI_grid = PHI, GRAM = GRAM, PHI_obs_block = P1,
  Z = matrix(Z, ncol = length(dat_idx)),   # nobs x n_data_times (column = data time)
  k_fit = kfit, sigma2_eps_fit = as.numeric(IDEm$get("sigma2_eps")),
  sigma2_eta_fit = as.numeric(IDEm$get("sigma2_eta")), beta_fit_gls = fit$beta,
  M_fit = Mstored, deoptim_bestmem = as.numeric(opt$bestmem), deoptim_bestval = opt$bestval,
  plausible_ranges = IDEm$get("plausible_ranges"),
  reference_points = refs,
  radar_df_head = head(radar_df))
writeLines(toJSON(out, digits = NA, matrix = "rowmajor", auto_unbox = TRUE), a[4])
cat("wrote", a[4], "\n")

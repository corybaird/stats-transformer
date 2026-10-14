source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "_common.R"))
suppressMessages(library(vars))
d <- read.csv(file.path(root, "tests/integration/var/canada.csv"))
y <- d[, c("e", "prod", "rw", "U")]
fit <- VAR(y, p = 2, type = "const")
amat <- diag(4)
diag(amat) <- NA
amat[2, 1] <- NA
amat[4, 1] <- NA
s <- SVAR(fit, estmethod = "direct", Amat = amat, hessian = TRUE, method = "BFGS", control = list(reltol = 1e-14, maxit = 10000))
out <- list(
  A = st_mat(s$A),
  impact = st_mat(solve(s$A)),
  Sigma_ml = st_mat(crossprod(resid(fit)) / nrow(resid(fit))),
  lr_test_stat = as.numeric(s$LR$statistic),
  lr_test_df = as.numeric(s$LR$parameter),
  iterations = as.numeric(s$iter)
)
st_write(out, c("vars"))

source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "_common.R"))
suppressMessages(library(vars))
d <- read.csv(file.path(root, "data/examples/timeseries/macrodata.csv"))
y <- d[, c("realgdp", "realcons", "realinv")]
fit <- VAR(y, p = 2, type = "const")
cf <- sapply(fit$varresult, function(m) coef(m))
out <- list(
  coef = st_mat(cf),
  Sigma_dof = st_mat(summary(fit)$covres),
  Sigma_ml = st_mat(crossprod(resid(fit)) / nrow(resid(fit))),
  nobs = fit$obs,
  loglik = as.numeric(logLik(fit)),
  roots = st_mat(Mod(roots(fit)))
)
st_write(out, c("vars"))

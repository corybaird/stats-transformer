source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "_common.R"))
d <- read.csv(file.path(root, "data/examples/timeseries/macrodata.csv"))
fit <- arima(d$realgdp, order = c(1, 0, 1), method = "ML", optim.control = list(reltol = 1e-14, maxit = 10000))
out <- list(
  coef = st_mat(coef(fit)),
  se = st_mat(sqrt(diag(fit$var.coef))),
  sigma2 = fit$sigma2,
  loglik = fit$loglik
)
st_write(out, c("stats"))

source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "_common.R"))
suppressMessages(library(urca))
d <- read.csv(file.path(root, "data/examples/timeseries/macrodata.csv"))
y <- d[, c("realgdp", "realcons", "realinv")]
jo <- ca.jo(y, type = "trace", ecdet = "none", K = 2, spec = "longrun")
r1 <- cajorls(jo, r = 1)
out <- list(
  lambda = st_mat(jo@lambda),
  trace_stats = st_mat(jo@teststat),
  trace_cval = st_mat(jo@cval),
  beta_normalized = st_mat(r1$beta),
  alpha = st_mat(coef(r1$rlm)["ect1", ]),
  rlm_coef = st_mat(coef(r1$rlm)),
  Sigma_ml = st_mat(crossprod(resid(r1$rlm)) / nrow(resid(r1$rlm))),
  nobs = nrow(resid(r1$rlm))
)
st_write(out, c("urca"))

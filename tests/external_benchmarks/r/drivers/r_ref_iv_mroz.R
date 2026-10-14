source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "_common.R"))
suppressMessages(library(AER))
d <- read.csv(file.path(root, "data/examples/regression/mroz.csv"))
d <- d[d$inlf == 1, ]
d$lwage <- as.numeric(d$lwage)
fit <- ivreg(lwage ~ exper + expersq + educ | exper + expersq + motheduc + fatheduc, data = d)
se <- function(v) as.numeric(sqrt(diag(v)))
out <- list(
  beta = st_mat(coef(fit)),
  se = st_mat(se(vcov(fit))),
  se_robust_hc0 = st_mat(se(sandwich::vcovHC(fit, type = "HC0"))),
  nobs = nobs(fit),
  df_residual = fit$df.residual
)
st_write(out, c("AER", "sandwich"))

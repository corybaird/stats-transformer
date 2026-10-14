source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "_common.R"))
suppressMessages(library(sandwich))
d <- read.csv(file.path(root, "data/examples/regression/longley.csv"))
fit <- lm(TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR, data = d)
se <- function(v) as.numeric(sqrt(diag(v)))
out <- list(
  beta = st_mat(coef(fit)),
  resid = st_mat(resid(fit)),
  se_classical = st_mat(se(vcov(fit))),
  se_hc0 = st_mat(se(vcovHC(fit, type = "HC0"))),
  se_hc1 = st_mat(se(vcovHC(fit, type = "HC1"))),
  se_hc2 = st_mat(se(vcovHC(fit, type = "HC2"))),
  se_hc3 = st_mat(se(vcovHC(fit, type = "HC3")))
)
st_write(out, c("sandwich"))

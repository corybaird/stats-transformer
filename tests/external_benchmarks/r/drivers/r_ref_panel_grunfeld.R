source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "_common.R"))
suppressMessages(library(plm))
d <- read.csv(file.path(root, "data/examples/regression/grunfeld.csv"))
pd <- pdata.frame(d, index = c("firm", "year"))
fit <- plm(invest ~ value + capital, data = pd, model = "within")
se <- function(v) as.numeric(sqrt(diag(v)))
out <- list(
  beta_fe = st_mat(coef(fit)),
  se_fe = st_mat(se(vcov(fit))),
  se_fe_cluster_hc1 = st_mat(se(vcovHC(fit, type = "HC1", cluster = "group"))),
  nobs = nobs(fit),
  r2_within = as.numeric(summary(fit)$r.squared["rsq"])
)
st_write(out, c("plm"))

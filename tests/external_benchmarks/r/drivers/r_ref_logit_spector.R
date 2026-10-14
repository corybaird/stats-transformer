source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "_common.R"))
d <- read.csv(file.path(root, "data/examples/discrete/spector.csv"))
fit <- glm(GRADE ~ GPA + TUCE + PSI, data = d, family = binomial(link = "logit"))
se <- function(v) as.numeric(sqrt(diag(v)))
out <- list(
  beta = st_mat(coef(fit)),
  se = st_mat(se(vcov(fit))),
  loglik = as.numeric(logLik(fit)),
  loglik_null = as.numeric(logLik(update(fit, . ~ 1))),
  nobs = nobs(fit)
)
st_write(out, c("stats"))

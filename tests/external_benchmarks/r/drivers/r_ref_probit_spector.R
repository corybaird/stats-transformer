source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "_common.R"))
d <- read.csv(file.path(root, "data/examples/discrete/spector.csv"))
fit <- glm(GRADE ~ GPA + TUCE + PSI, data = d, family = binomial(link = "probit"), control = glm.control(epsilon = 1e-14, maxit = 100))
out <- list(
  beta = st_mat(coef(fit)),
  se = st_mat(sqrt(diag(vcov(fit)))),
  loglik = as.numeric(logLik(fit)),
  nobs = nobs(fit)
)
st_write(out, c("stats"))

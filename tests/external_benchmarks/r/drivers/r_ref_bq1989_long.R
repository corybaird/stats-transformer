source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "_common.R"))
suppressMessages(library(vars))
y <- as.matrix(read.csv(input_csv, header = TRUE))
fit <- VAR(y, p = 8, type = "const")
bq <- BQ(fit)
irf_arr <- Phi(fit, nstep = 39)
out <- list(
  coef = st_mat(sapply(fit$varresult, function(m) coef(m))),
  Sigma_dof = st_mat(summary(fit)$covres),
  B_impact = st_mat(bq$B),
  LR = st_mat(bq$LRIM),
  Phi = irf_arr
)
st_write(out, c("vars"))

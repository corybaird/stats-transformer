source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "_common.R"))
suppressMessages(library(urca))
base <- file.path(root, "data/examples/timeseries/ghysels_ch7")
fit_one <- function(y, K, r) {
  jo <- ca.jo(y, type = "eigen", ecdet = "const", K = K, spec = "longrun")
  rl <- cajorls(jo, r = r)
  co <- coef(rl$rlm)
  list(jo = jo, beta = rl$beta, alpha = co[grep("^ect", rownames(co)), , drop = FALSE], Sigma = crossprod(resid(rl$rlm)) / nrow(resid(rl$rlm)))
}
sim <- read.csv(file.path(base, "simulated_cointegration.csv"))[1:400, c("x", "y")]
uk <- read.csv(file.path(base, "dataonly_uktermstructure.csv"))[1:216, c("r1", "r3", "r6", "r24")]
lei <- read.csv(file.path(base, "dataonly_uslei.csv"))[313:540, c("cci", "cli")]
s1 <- fit_one(sim, 2, 1)
u1 <- fit_one(uk, 3, 1)
u3 <- fit_one(uk, 3, 3)
l1 <- fit_one(lei, 2, 1)
out <- list(
  sim_lambda = st_mat(s1$jo@lambda), sim_eigen_stats = st_mat(s1$jo@teststat), sim_beta = st_mat(s1$beta), sim_alpha = st_mat(s1$alpha), sim_Sigma_ml = st_mat(s1$Sigma),
  uk_lambda = st_mat(u1$jo@lambda), uk_eigen_stats = st_mat(u1$jo@teststat), uk_r1_beta = st_mat(u1$beta), uk_r1_alpha = st_mat(u1$alpha), uk_r1_Sigma_ml = st_mat(u1$Sigma),
  uk_r3_beta = st_mat(u3$beta), uk_r3_alpha = st_mat(u3$alpha),
  lei_lambda = st_mat(l1$jo@lambda), lei_eigen_stats = st_mat(l1$jo@teststat), lei_beta = st_mat(l1$beta), lei_alpha = st_mat(l1$alpha), lei_Sigma_ml = st_mat(l1$Sigma)
)
st_write(out, c("urca"))

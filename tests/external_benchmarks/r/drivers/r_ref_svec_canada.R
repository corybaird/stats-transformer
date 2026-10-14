source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "_common.R"))
suppressMessages(library(vars))
d <- read.csv(file.path(root, "tests/integration/var/canada.csv"))
y <- d[, c("prod", "e", "U", "rw")]
jo <- ca.jo(y, type = "trace", ecdet = "trend", K = 3, spec = "transitory")
LR <- matrix(NA, 4, 4)
LR[1, 2:4] <- 0
LR[2:4, 4] <- 0
SR <- matrix(NA, 4, 4)
SR[4, 2] <- 0
s <- SVEC(jo, LR = LR, SR = SR, r = 1, lrtest = FALSE, boot = FALSE, max.iter = 1000, conv.crit = 1e-12)
out <- list(
  SR = st_mat(s$SR),
  LR = st_mat(s$LR),
  iterations = s$iter,
  ENDO = st_mat(as.matrix(y))
)
st_write(out, c("vars", "urca"))

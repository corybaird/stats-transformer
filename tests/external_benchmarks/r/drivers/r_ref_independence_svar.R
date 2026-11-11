source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "_common.R"))
suppressMessages(library(vars))
suppressMessages(library(svars))
d <- read.csv(file.path(root, "data/examples/timeseries/macrodata.csv"))
y <- d[, c("realgdp", "realcons", "realinv")]
fit <- VAR(y, p = 2, type = "const")
s <- id.dc(fit)
out <- list(B = st_mat(s$B))
st_write(out, c("vars", "svars"))

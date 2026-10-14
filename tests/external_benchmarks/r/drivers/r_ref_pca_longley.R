source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "_common.R"))
d <- read.csv(file.path(root, "data/examples/regression/longley.csv"))
p <- prcomp(d[, c("GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "YEAR")], center = TRUE, scale. = TRUE)
out <- list(
  eigenvalues = st_mat(p$sdev^2),
  loadings = st_mat(p$rotation)
)
st_write(out, c("stats"))

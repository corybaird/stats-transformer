suppressMessages(library(vars))
suppressMessages(library(tsDyn))
suppressMessages(library(jsonlite))

# 1. Standard VAR(2) on Canada dataset
data(Canada)
write.csv(Canada, "tests/integration/data/canada.csv", row.names = FALSE)

v <- VAR(Canada, p = 2, type = "const")
coef_var <- Bcoef(v)

# 2. Johansen VECM
vecm_fit <- VECM(Canada, lag = 2, r = 1, estim = "ML")
coef_vecm <- coef(vecm_fit)

res <- list(
  var_coef = coef_var,
  vecm_coef = coef_vecm
)

write_json(res, "tests/integration/var_results.json", auto_unbox = TRUE, digits = 8)

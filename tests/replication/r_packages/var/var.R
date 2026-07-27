suppressMessages(library(vars))
suppressMessages(library(jsonlite))

data(Canada)

# Save Canada dataset to CSV for Python to load
write.csv(Canada, "tests/replication/r_packages/var/canada.csv", row.names = FALSE)

# Fit VAR(2) model
fit_var <- VAR(Canada, p = 2, type = "const")

# Extract VAR coefficients
coef_e <- coef(fit_var)$e[, "Estimate"]
coef_prod <- coef(fit_var)$prod[, "Estimate"]
coef_rw <- coef(fit_var)$rw[, "Estimate"]
coef_U <- coef(fit_var)$U[, "Estimate"]

res <- list(
  coef_e = as.list(coef_e),
  coef_prod = as.list(coef_prod),
  coef_rw = as.list(coef_rw),
  coef_U = as.list(coef_U)
)

write_json(res, "tests/replication/r_packages/var/var_results.json", auto_unbox = TRUE, digits = 8)

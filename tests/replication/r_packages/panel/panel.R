suppressMessages(library(jsonlite))
suppressMessages(library(plm))

data("Grunfeld", package = "plm")

# Write out Grunfeld dataset to CSV for Python to load
write.csv(Grunfeld, "tests/replication/r_packages/panel/grunfeld.csv", row.names = FALSE)
# Run pooled OLS model
fit_pooled <- plm(inv ~ value + capital, data = Grunfeld, model = "pooling")
sum_pooled <- summary(fit_pooled)

# Run fixed effects model (within)
fit_fe <- plm(inv ~ value + capital, data = Grunfeld, model = "within")
sum_fe <- summary(fit_fe)

# Run random effects model (random)
fit_re <- plm(inv ~ value + capital, data = Grunfeld, model = "random")
sum_re <- summary(fit_re)

# Export coefficients and SEs to JSON
res <- list(
  pooled = list(coefficients = as.list(coef(fit_pooled)), se = as.list(sqrt(diag(vcov(fit_pooled))))),
  fe = list(coefficients = as.list(coef(fit_fe)), se = as.list(sqrt(diag(vcov(fit_fe))))),
  re = list(coefficients = as.list(coef(fit_re)), se = as.list(sqrt(diag(vcov(fit_re)))))
)

write_json(res, "tests/replication/r_packages/panel/panel_results.json", auto_unbox = TRUE, digits = 8)

suppressMessages(library(jsonlite))
suppressMessages(library(plm))

data("Grunfeld", package = "plm")

# Write out Grunfeld dataset to CSV for Python to load
write.csv(Grunfeld, "tests/integration/panel/grunfeld.csv", row.names = FALSE)

# 1. Fixed Effects (Within)
fe_model <- plm(inv ~ value + capital, data = Grunfeld, model = "within")
fe_coef <- coef(fe_model)

# 2. Random Effects
re_model <- plm(inv ~ value + capital, data = Grunfeld, model = "random")
re_coef <- coef(re_model)

res <- list(
  fe_coef = as.list(fe_coef),
  re_coef = as.list(re_coef)
)

write_json(res, "tests/integration/panel/panel_results.json", auto_unbox = TRUE, digits = 8)

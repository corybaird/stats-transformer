suppressMessages(library(jsonlite))
suppressMessages(library(sandwich))
suppressMessages(library(lmtest))
suppressMessages(library(AER))

# 1. OLS & Robust OLS on Ghysels US GDP data
us_gdp <- read.csv("tests/integration/data/ghysels/ex2_regress_gdp_us.csv")

fit_ols <- lm(y ~ ipr + su + pr + sr, data = us_gdp)
coef_ols <- coef(fit_ols)

# Robust SEs
se_hc0 <- sqrt(diag(vcovHC(fit_ols, type = "HC0")))
se_hc1 <- sqrt(diag(vcovHC(fit_ols, type = "HC1")))
se_hc2 <- sqrt(diag(vcovHC(fit_ols, type = "HC2")))
se_hc3 <- sqrt(diag(vcovHC(fit_ols, type = "HC3")))

# 2. Logit Model using simulated binary outcome
us_gdp$y_bin <- ifelse(us_gdp$y > 0, 1, 0)
fit_logit <- glm(y_bin ~ ipr + su, data = us_gdp, family = binomial(link = "logit"))
coef_logit <- coef(fit_logit)

# 3. IV Regression using AER::ivreg
# Formula: y ~ ipr + su | sr + pr (ipr endog, sr/pr instruments)
fit_iv <- ivreg(y ~ ipr + su | sr + pr + su, data = us_gdp)
coef_iv <- coef(fit_iv)

res <- list(
  ols_coef = as.list(coef_ols),
  se_hc0 = as.list(se_hc0),
  se_hc1 = as.list(se_hc1),
  se_hc2 = as.list(se_hc2),
  se_hc3 = as.list(se_hc3),
  logit_coef = as.list(coef_logit),
  iv_coef = as.list(coef_iv)
)

write_json(res, "tests/integration/regression/regression_results.json", auto_unbox = TRUE, digits = 8)

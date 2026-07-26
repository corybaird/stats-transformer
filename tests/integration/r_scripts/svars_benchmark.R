suppressMessages(library(vars))
suppressMessages(library(svars))
suppressMessages(library(jsonlite))

# 1. Load USA dataset
data(USA)

# Write to CSV for Python to load
write.csv(USA, "tests/integration/data_usa.csv", row.names=FALSE)

# 2. Fit VAR(6)
v <- VAR(USA, p = 6, type = "const")

# 3. Fit changes in volatility SVAR
# Break in Q3 1979 (observation 79)
x <- id.cv(v, SB = 79)

# Extract structural matrix B and Lambda (diagonal)
B <- x$B
Lambda <- x$Lambda

# 4. Fit distance covariance SVAR
# x_dc <- id.dc(v)
# B_dc <- x_dc$B

# Output as JSON
res <- list(
  B_cv = B,
  Lambda_cv = diag(Lambda)
)

write_json(res, "tests/integration/svars_results.json")

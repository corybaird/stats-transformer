suppressMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
root <- args[1]
out_path <- args[2]
input_csv <- if (length(args) >= 3) args[3] else NA
st_write <- function(outputs, packages) {
  labels <- list()
  for (nm in names(outputs)) {
    v <- outputs[[nm]]
    if (is.matrix(v)) {
      labels[[nm]] <- list(rows = if (is.null(rownames(v))) list() else as.list(rownames(v)), cols = if (is.null(colnames(v))) list() else as.list(colnames(v)))
      dimnames(v) <- NULL
      outputs[[nm]] <- v
    }
  }
  pk <- sapply(packages, function(p) as.character(packageVersion(p)), USE.NAMES = TRUE)
  res <- list(
    outputs = outputs,
    labels = labels,
    r_version = R.version.string,
    platform = R.version$platform,
    packages = as.list(pk)
  )
  write_json(res, out_path, auto_unbox = TRUE, digits = I(17), na = "string", matrix = "rowmajor")
}
st_mat <- function(x) {
  as.matrix(x)
}

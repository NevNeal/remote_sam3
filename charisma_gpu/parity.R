# Reference results from R charisma, for tests/parity against charisma_gpu.
#
#   Rscript charisma_gpu/parity.R <file listing PNG paths, one per line> <out.csv> [clut.csv]
#
# One row per image and colour class: path, k, colour, prop, seconds.
suppressPackageStartupMessages(library(charisma))
args <- commandArgs(trailingOnly = TRUE)
paths <- readLines(args[1])
clut <- if (length(args) >= 3) read.csv(args[3], stringsAsFactors = FALSE) else charisma::clut
rows <- list()
for (p in paths) {
  t0 <- Sys.time()
  x <- tryCatch(suppressMessages(suppressWarnings(
    charisma(p, threshold = 0, pavo = FALSE, plot = FALSE, clut = clut))),
    error = function(e) { message("error on ", p, ": ", conditionMessage(e)); NULL })
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  if (is.null(x)) next
  tab <- x$charisma_calls_table
  rows[[length(rows) + 1]] <- data.frame(path = p, k = x$k, colour = tab$classification,
                                         prop = tab$prop, seconds = secs)
}
write.csv(do.call(rbind, rows), args[2], row.names = FALSE)

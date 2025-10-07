library(mvcluster)
library(RcppCNPy)
library(jsonlite)
library(data.table)

run_mvbc <- function(G_path, C_path, lambda_W = 1, lambda_H_G = 1, lambda_H_C = 1) {
  lvs <- c(lambda_H_G, lambda_H_C)
  lz  <- lambda_W

  # --- Read G ---
  hdr <- fread(G_path, nrows = 0)
  G   <- fread(G_path, select = 7:ncol(hdr))
  G   <- as.matrix(G)
  storage.mode(G) <- "integer"

  # --- Read C (.npy) ---
  C <- npyLoad(C_path)
  storage.mode(C) <- "integer"

  # --- Run MVBC ---
  paste("running MVBC")
  datasets <- list(G, C)
  result   <- mvsvdl1(datasets, lvs, lz)

  # --- Return as JSON ---
  cat(toJSON(result, auto_unbox = TRUE))
}

# ---- main ----
args <- commandArgs(trailingOnly = TRUE)

# Set defaults if arguments missing
G_path      <- args[1]
C_path      <- args[2]
lambda_W    <- if (length(args) >= 3) as.numeric(args[3]) else 1
lambda_H_G  <- if (length(args) >= 4) as.numeric(args[4]) else 1
lambda_H_C  <- if (length(args) >= 5) as.numeric(args[5]) else 1
maxOuter  <- if (length(args) >= 5) as.numeric(args[5]) else 100000 # 100000 is default

run_mvbc(G_path, C_path, lambda_W, lambda_H_G, lambda_H_C)

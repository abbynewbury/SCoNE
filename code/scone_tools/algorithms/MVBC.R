library(mvcluster)
library(jsonlite)
library(data.table)

l0 <- function(x) sum(abs(x) > 0)

read_matrix <- function(path) {
  # if csv/tsv: assumes that the first column is index names
  ext <- tools::file_ext(path)
  if (ext == "csv") { x <- as.matrix(read.csv(path, row.names = 1, check.names = FALSE)) }
  else if (ext == "tsv") { x <- as.matrix(read.delim(path, row.names = 1, check.names = FALSE)) }
  else { stop("Unsupported file type: ", ext) }
  storage.mode(x) <- "double"
  x
}



run_mvbc <- function(G_path, C_path, rank, lambda_W, lambda_H_G, lambda_H_C, maxOuter) {
  lvs <- c(lambda_H_G, lambda_H_C)
  lz  <- lambda_W

  # --- Read G ---
  G <- read_matrix(G_path)

  # --- Read C ---
  C <- read_matrix(C_path)

  stopifnot(!any(is.nan(G)))
  stopifnot(!any(is.nan(C)))

  # --- Run MVBC ---
  remaining <- seq_len(nrow(G)) # indices of individuals still under consideration
  W <- matrix(0, nrow = dim(G)[1], ncol = rank)
  for (i in seq_len(rank)){
    G_i <- G[remaining, , drop = FALSE]
    C_i <- C[remaining, , drop = FALSE]

    datasets <- list(G_i, C_i)
    result <- mvsvdl1(datasets, lvs, lz,maxOuter=maxOuter)
    # store W
    cl <- as.vector(result$Cluster)
    cl_full <- rep(0, nrow(G))  # fill with 0s by default
    cl_full[remaining] <- cl  
    W[, i] <- cl_full
    
    # drop individuals in cluster
    drop_mask <- cl == 1 # individuals to drop before next rank 1 run 
    if (all(drop_mask, na.rm = TRUE)) {
    message("No more people to remove; stopping at iteration ", i)
    break
    } 
    remaining <- remaining[!drop_mask]  

  }
  factor_matrices <- list(W = W)
  loss_history <- list(0) # just a placeholder

  output <- list(factor_matrices = factor_matrices,loss_history = loss_history)
  cat(toJSON(output, auto_unbox = TRUE))
}

# ---- main ----
args <- commandArgs(trailingOnly = TRUE)

# Set defaults if arguments missing
G_path      <- args[1]
C_path      <- args[2]
rank        <- as.numeric(args[3])
lambda_W    <- if (length(args) >= 4) as.numeric(args[4]) else 1
lambda_H_G  <- if (length(args) >= 5) as.numeric(args[5]) else 1
lambda_H_C  <- if (length(args) >= 6) as.numeric(args[6]) else 1
maxOuter  <- 1000000 # 100000 is default, we increase it to 1000000 & if convergence still fails, consider it failed run
run_mvbc(G_path, C_path, rank, lambda_W, lambda_H_G, lambda_H_C, maxOuter)

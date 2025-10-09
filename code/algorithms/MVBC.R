library(mvcluster)
library(RcppCNPy)
library(jsonlite)
library(data.table)

l0 <- function(x) sum(abs(x) > 0)
tr <- function(M) sum(diag(M))


run_mvbc <- function(G_path, C_path, rank, lambda_W, lambda_H_G, lambda_H_C, maxOuter) {
  lvs <- c(lambda_H_G, lambda_H_C)
  lz  <- lambda_W

  # --- Read G ---
  hdr <- fread(G_path, nrows = 0)
  G   <- fread(G_path, select = 7:ncol(hdr))
  G   <- as.matrix(G)
  storage.mode(G) <- "integer"

  # --- Read C (.npy) ---
  C <- npyLoad(C_path,"integer")
  storage.mode(C) <- "integer"

  # --- Run MVBC ---
  remaining <- seq_len(nrow(G)) # indices of individuals still under consideration
  W <- matrix(0, nrow = dim(G)[1], ncol = rank)
  total_loss <- list()
  G_plus_C_loss <- list()
  for (i in seq_len(rank)){
    G_i <- G[remaining, , drop = FALSE]
    C_i <- C[remaining, , drop = FALSE]
    datasets <- list(G_i, C_i)
    result <- mvsvdl1(datasets, lvs, lz)
    # store W
    cl <- as.vector(result$Cluster)
    cl_full <- numeric(dim(G)[1]) # fill with 0s by default
    cl_full[remaining] <- cl  
    W[, i] <- cl_full

    # compute loss
    G_plus_C_loss_i <- 0
    for (j in seq_len(length(datasets))) {
      # calculate sigma_i
      zu <- result$z * result$U[,j]  
      B <- tcrossprod(zu, result$V[[j]]) 
      sigma <- sum(datasets[[j]] * B)/(norm(B,"F")^2)

      # rank-1 reconstruction: (z ⊙ u_j) v_j^T 
      R <- datasets[[j]] -  sigma * B
      G_plus_C_loss_i <- G_plus_C_loss_i + norm(R,"F")^2 
    }
    total_loss_i <- G_plus_C_loss_i + lambda_W*l0(result$z) + lambda_H_G*l0(result$V[[1]]) + lambda_H_C*l0(result$V[[2]]) 
    total_loss[[i]] <- total_loss_i
    G_plus_C_loss[[i]] <- G_plus_C_loss_i

    # drop individuals in cluster
    drop_mask <- cl == 1 # individuals to drop before next rank 1 run 
    if (all(drop_mask, na.rm = TRUE)) {
    message("No more people to remove; stopping at iteration ", i)
    break
    } 
    remaining <- remaining[!drop_mask]  

  }
  # compute loss as sum over len(rank)
  factor_matrices <- list(W = W)
  loss_history <- list(G_plus_C_loss=mean(unlist(G_plus_C_loss)), total_loss=mean(unlist(total_loss)))
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

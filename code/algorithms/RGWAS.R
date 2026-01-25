library(rgwas)
library(data.table)
library(jsonlite)
library(RcppCNPy)

args <- commandArgs(trailingOnly=TRUE)
G_path <- as.character(args[1])
C_path <- as.character(args[2])
Z_path <- as.character(args[3]) 
rank <- as.integer(args[4])
num_init <- as.integer(args[5])

# read in matrices
read_matrix <- function(path, mode = "double") {
  # if csv/tsv: assumes that the first column is index names
  ext <- tools::file_ext(path)

  if (ext %in% c("csv", "tsv")) {
    df <- if (ext == "csv") {
      read.csv(path, row.names = 1)
    } else {
      read.delim(path, row.names = 1)
    }

    vals <- as.vector(as.matrix(df)) # store as vector first since R matrices are column order and npy are row-order
    storage.mode(vals) <- "double"
    x <- matrix(vals, nrow = nrow(df), ncol = ncol(df), byrow = TRUE)

  } else if (ext == "npy") {
    x <- RcppCNPy::npyLoad(path, "integer")
  } else {
    stop("Unsupported file type: ", ext)
  }

  storage.mode(x) <- mode
  x
}

G <- read_matrix(G_path)
C <- read_matrix(C_path)

# read in Z
Z <- read_matrix(Z_path)
Z <- Z[,-ncol(Z)] # last column of Z is perfectly multicollinear with rest

covars <- cbind(1,G,Z)
C_binary <- (C != 0) + 0L
result <- mfmr(Yb=C_binary,  Yq=NULL, G=covars, K=rank, nrun=num_init)
cat(toJSON(result, auto_unbox = TRUE))

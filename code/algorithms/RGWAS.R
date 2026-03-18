library(rgwas)
library(data.table)
library(jsonlite)

args <- commandArgs(trailingOnly=TRUE)
G_path <- as.character(args[1])
C_path <- as.character(args[2])
Z_path <- as.character(args[3]) 
rank <- as.integer(args[4])
num_init <- as.integer(args[5])

# read in matrices
read_matrix <- function(path) {
  # if csv/tsv: assumes that the first column is index names
  ext <- tools::file_ext(path)
  if (ext == "csv") { x <- as.matrix(read.csv(path, row.names = 1, check.names = FALSE)) }
  else if (ext == "tsv") { x <- as.matrix(read.delim(path, row.names = 1, check.names = FALSE)) }
  else { stop("Unsupported file type: ", ext) }
  storage.mode(x) <- "double"
  x
}

G <- read_matrix(G_path)
C <- read_matrix(C_path)

# read in Z
Z <- read_matrix(Z_path)

covars <- cbind(1,G,Z)
C_binary <- (C != 0) + 0L
result <- mfmr(Yb=C_binary,  Yq=NULL, G=covars, K=rank, nrun=num_init)
cat(toJSON(result, auto_unbox = TRUE))

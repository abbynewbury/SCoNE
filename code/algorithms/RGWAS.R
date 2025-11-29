library(rgwas)
library(data.table)
library(jsonlite)
library(RcppCNPy)

args <- commandArgs(trailingOnly=TRUE)
G_path <- as.character(args[1])
C_path <- as.character(args[2])
Z_path <- as.character(args[3]) # /gpfs/commons/datasets/1000genomes/release-20130502-supporting/admixture_files/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05.5.Q
rank <- as.integer(args[4])
num_init <- as.integer(args[5])

# read in matrices, assumes G is in .raw format
G <- npyLoad(G_path,"integer")
storage.mode(G) <- "double"
C <- npyLoad(C_path,"integer")
storage.mode(C) <- "double"

# read in Z
Z <- npyLoad(Z_path,"integer")
storage.mode(Z) <- "double"
Z <- Z[,-ncol(Z)] # last column of Z is perfectly multicollinear with rest

covars <- cbind(1,G,Z)
C_binary <- (C != 0) + 0L
result <- mfmr(Yb=C_binary,  Yq=NULL, G=covars, K=rank, nrun=num_init)
cat(toJSON(result, auto_unbox = TRUE))

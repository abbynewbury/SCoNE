library(rgwas)
library(data.table)
library(jsonlite)
library(RcppCNPy)

args <- commandArgs(trailingOnly=TRUE)
G_path <- as.character(args[1])
C_path <- as.character(args[2])
iid_index_path <- as.character(args[3]) # defines the iids used in this simulation
Z_path <- as.character(args[4]) # /gpfs/commons/datasets/1000genomes/release-20130502-supporting/admixture_files/ALL.wgs.phase3_shapeit2_filtered.20141217.maf0.05.5.Q
rank <- as.integer(args[5])
num_init <- as.integer(args[6])

# read in matrices, assumes G is in .raw format
hdr <- fread(G_path, nrows = 0)
G <- fread(G_path, select = 7:ncol(hdr)) # skip first 6 columns
G <- as.matrix(G)
storage.mode(G) <- "integer" 
C <- npyLoad(C_path,"integer")

# read in Z
Z <- read.table(Z_path, header = FALSE, sep = "", stringsAsFactors = FALSE, check.names = FALSE)
iid_index <- npyLoad(iid_index_path,"integer") 
iid_index <- iid_index + 1
Z <- Z[iid_index,,]
Z <- data.matrix(Z)

covars <- cbind(1,G,Z)
C_binary <- (C != 0) + 0L
result <- mfmr(Yb=C_binary,  Yq=NULL, G=covars, K=rank, nrun=num_init)
cat(toJSON(result, auto_unbox = TRUE))

## Packages
library("NetworkRiskMeasures")
library(ggplot2)
library(ggnetwork)
library(igraph)
library("systemicrisk")

## Datasets
data_EBA = read.csv('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/Data EBA/dataset_EBA_cleaned.csv', row.names=1)

## Generate pertubed a and l 
set.seed(2)
a = data_EBA$Total.Interbank.Assets
n = length(a)
epsilon_vec = rnorm(n = n, mean = 0, sd = 100)
l_pert = (a + epsilon_vec) * sum(a) / sum(a + epsilon_vec)
l_pert[n] = sum(a) - sum(l_pert[1:(n-1)])
print(l_pert)

# Create new function sample_ERE because default parameter epsilon is set to 1e-9, needs to be set higher. 
sample_ERE_new = function (l, a, p, lambda, nsamples = 10000, thin = 1, burnin = 10000)
{
  n <- length(l)
  if (!is.matrix(p)) {
    p <- matrix(p, nrow = n, ncol = n)
    diag(p) <- 0
  }
  if (!is.matrix(lambda)) {
    lambda <- matrix(lambda, nrow = n, ncol = n)
    diag(lambda) <- 0
  }
  L <- findFeasibleMatrix(l, a, p, eps=1e-7)                    #here, we adjust epsilon to 1e-7 from 1e-9, due to numerical precision errors
  steps_ERE(L = L, p = p, lambda = lambda, nsamples = nsamples,
            thin = thin, burnin = burnin)
}

# Generate liability matrix using method of Gandy and Veraart (2016)
# Method 1: Equal linking probabilities
p = 0.5
lambda = p * n * (n-1) / sum(a)
# W_GV = sample_ERE_new(a, l_pert, p, lambda, nsamples = 10, thin = 10^4, burnin=10^9)    #use same param. settings as in Feinsteind and Hurd (2022)

# Method 2: Core-periphery structure
n_l = 8
n_s = n - n_l
p_ER = 0.3
p_l = 0.5
p_s = (n*(n-1)*p_ER - n_l*(n-1)*p_l - n_l*n_s*p_l)/(n_s*(n_s-1))
indices_largest_banks = head(order(data_EBA$Total.Interbank.Assets, decreasing = TRUE), 8) 
p_cp = matrix(0, n, n)
for (i in 1:n){
  for (j in 1:n){
    if (i %in% indices_largest_banks || j %in% indices_largest_banks) {
      p_cp[i,j] = p_l } 
    else {
      p_cp[i,j] = p_s
    }
  }
}
diag(p_cp) = 0 
lambda_cp = sum(p_cp) / sum(a)
W_GV_cp = sample_ERE_new(a, l_pert, p_cp, lambda_cp, nsamples = 10, thin = 10^5, burnin=10^9)

# Export matrices
for (i in 1:10){
  #W_df_GV = as.data.frame(W_GV[[i]])
  W_df_GV_cp = as.data.frame(W_GV_cp[[i]])
  #write.csv(W_df_GV, sprintf("/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/W_matrix_GV_%d.csv", i))
  write.csv(W_df_GV_cp, sprintf("/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/W_matrix_GV_cp_%d.csv", i))
}

# Plot example matrix
png("/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/Test2.png", width = 15, height = 15, units = "in", res = 300)
ig <- graph.adjacency(W_GV_cp[[1]], mode="undirected", weighted=TRUE)
ig_filter <- delete_vertices(ig, which(degree(ig) < 5))
plot.igraph(ig_filter,vertex.size=3, 
            vertex.label.cex=.5, 
            layout=layout.fruchterman.reingold(ig_filter, niter=10000)) 



# Save matrices as CSV for export to Python

# write.csv(W_df_ME, "/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/W_matrix_ME.csv")

# Generate liability matrix using maximum entropy (Upper, 2004) and minimum density estimation (Anand et al, 2015).
# We use the package 'NetworkRiskMeasures' of Carlos Scinelli (https://github.com/carloscinelli/NetworkRiskMeasures)
# W_MD = matrix_estimation(rowsums = a, colsums = l_pert, method = "md")
# W_df_MD = as.data.frame(W_MD)

W_ME = matrix_estimation(rowsums = a[1:12], colsums = a[1:12], method = "me")
W_df_ME = as.data.frame(W_ME)

W_ME_normalized = t(transpose(W_ME) /rowSums(W_ME))
write.csv(W_ME_normalized, "/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/W_matrix_ME.csv")

# Liability matrix ME for Chapter 5
a = c(rep(7, 35), rep(7*20, 5))
l = c(rep(7, 35), rep(7*20, 5))
W_ME_CH5 = matrix_estimation(rowsums = l, colsums = a, method = "me")
W_ME_CH5_normalized = t(t(W_ME_CH5) /rowSums(W_ME_CH5))
check = colSums(W_ME_CH5_normalized)

write.csv(W_ME_CH5_normalized, "/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/W_matrix_CH5_ME.csv")

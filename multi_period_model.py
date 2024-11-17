import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from helper_functions import compute_equilibrium

## Model parameters 
n = 6
c = np.ones(n) * 10 
m = np.ones(n) 
l = c / m   
W = np.ones((n,n), dtype=int) * 1/(n-1)
np.fill_diagonal(W, 0)
T = 2

# Settings GBM
np.random.seed(1)
r = 0.0
sigma = 0.10
a1_tilde = np.array([45, 20, 50, 13, 33, 48]) 
d = 20
a_1 = a1_tilde - d
_, _, _, s_1 = compute_equilibrium(l, c, m, W, a1_tilde * np.exp(r) - d)
print(np.round(s_1, 4))

# Simulate time 2 values
Z_mvn = np.random.normal(size=n)
a_2 = a1_tilde * np.exp((r - 0.5*sigma**2) + sigma * Z_mvn) - d

# Compute time 1 prices 
n_sim_MC = 10_000
s_array = np.zeros((n_sim_MC, n))
for k in range(n_sim_MC):
    Z_mvn = np.random.normal(size=n)
    a_2_k = a1_tilde * np.exp((r - 0.5*sigma**2) + sigma * Z_mvn) - d
    _, _, _, s = compute_equilibrium(l, c, m, W, a_2_k)
    s_array[k, :] = s
s_1_MC = s_array.mean(axis = 0)
s_1_MC_se = s_array.std(axis = 0) / np.sqrt(n_sim_MC)

# Compute time 2 prices
B_1 = np.array([])
C_1 = np.array([i for i in range(n) if s_1_MC[i] <= l[i]])
H_1 = np.array([i for i in range(n) if s_1_MC[i] > l[i]])
B_2, C_2, H_2, s_2 = compute_equilibrium(l, c, m, W, a_2, pre_converted=C_1)
print(a_1, np.round(s_1_MC, 4), np.round(s_1_MC_se, 6))
print(np.round(a_2, 4), np.round(s_2, 4))

import numpy as np
import matplotlib.pyplot as plt
import time
from helper_functions import generate_network, import_weight_matrices_R, generate_network, compute_equilibrium_multiple_thresholds, plot_results_sensitivity
import pickle 
import pandas as pd 


# alpha = 1.0
# c = np.array([[12 * alpha, 12 * (1-alpha)], [12 * alpha, 12 * (1-alpha)],  [12 * alpha, 12 * (1-alpha)]])
# m = np.array([[1, 100000], [1, 100000], [1, 100000]])
# l = c/m
# W = np.array([[0, 0.25, 0.25], [0.25, 0, 0.25], [0.25, 0.25, 0]])
# a = np.array([15, 15, -4])
# B, G2, G1, H, s_list = compute_equilibrium_multiple_thresholds(l, c, m, np.tile(W, (2, 1, 1)), a)

# print(0.5 * s_list[2] * 100000)
# print(0.5 * s_list[2] * 1 + 0.5 * c[2, 0])

# print(B, G2, G1, H, s_list)


# Import other data and define model parameters 
# np.set_printoptions(suppress=True)
# dataset_EBA = pd.read_csv('Data EBA/dataset_EBA_cleaned.csv', index_col=0) / 1000       #scale by 1000 to prevent numerical issues
# Wmatrices = import_weight_matrices_R('Data EBA/W_matrix_GV_cp_1.csv', num_matrices=1) 

# c = dataset_EBA["Total Interbank Liabilities"].to_numpy() + dataset_EBA["Total External Liabilities"].to_numpy()
# a = dataset_EBA["Total External Assets"].to_numpy() * 0.70

# c_small = c#[:12]
# a_small = a#[:12]
# alphas = np.linspace(0.0, 0.8, 9)
# value_int_arr = np.zeros((alphas.shape[0]))
# value_ext_arr = np.zeros((alphas.shape[0]))
# value_shares_arr = np.zeros((alphas.shape[0]))
# value_default_arr = np.zeros((alphas.shape[0]))
# for i, alpha in enumerate(alphas):
#     c_CoCo = dataset_EBA["Total Interbank Liabilities"].to_numpy()  * alpha 
#     c_regular = dataset_EBA["Total External Liabilities"].to_numpy() + dataset_EBA["Total Interbank Liabilities"].to_numpy() * (1 - alpha)
#     c_multi = np.vstack((c_CoCo, c_regular)).T
#     m = np.vstack((np.ones(c.shape[0]) * 1.0, np.ones(c.shape[0])*100000)).T
#     l = c_multi/m 
#     W = Wmatrices[0] * dataset_EBA["Total Interbank Liabilities"].to_numpy() / c_small
#     B, G2, G1, H, s_list = compute_equilibrium_multiple_thresholds(l, c_multi, m, np.tile(W, (2, 1, 1)), a_small, verbose=True)
    
#     value_ext = ((2/3) * G2 * s_list * (1 + m[:, 1]) * dataset_EBA["Total External Liabilities"].to_numpy() / c_regular  \
#                 + (1 - G2) * (dataset_EBA["Total External Liabilities"].to_numpy())) / dataset_EBA["Total External Liabilities"].to_numpy()
    
#     value_int = (G2 * s_list * m[:, 1] * dataset_EBA["Total Interbank Liabilities"].to_numpy() * (1 - alpha) / c_regular + \
#                 + G1 * (dataset_EBA["Total Interbank Liabilities"].to_numpy() * (1-alpha) + alpha * s_list * m[:, 0]) + \
#                 + H * dataset_EBA["Total Interbank Liabilities"].to_numpy()) / dataset_EBA["Total Interbank Liabilities"].to_numpy()

#     print(B)
#     print(G2)
#     print(G1)
#     print(H)
#     print(s_list)
#     print(f"Value of alpha: {alpha}")
#     value_int_arr[i] = value_int.mean()
#     value_ext_arr[i] = value_ext.mean()
#     value_shares_arr[i] = s_list.sum()
#     value_default_arr[i] = G2.sum()
    
#     print(value_ext.sum() + value_int.sum() + s_list.sum())
#     print(value_ext.sum())
#     print(B.sum())
#     print(G2.sum())
#     print(G1.sum())
#     print(H.sum())

# plt.plot(alphas, value_default_arr)
# plt.show()
# plt.plot(alphas, value_int_arr)
# plt.show()
# plt.plot(alphas, value_ext_arr)
# plt.show()
# plt.plot(alphas, value_shares_arr)
# plt.show()
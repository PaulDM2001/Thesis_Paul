import numpy as np
from scipy.stats import norm
from helper_functions import compute_equilibrium, generate_problem_instance, generate_2d_figure, generate_3d_figure

# # Problem definition
# c = np.array([8, 8])
# m = np.array([1, 1])
# l = c / m   
# W = np.array([[0, 0.75], [0.75, 0]])

# # Simulate asset values given initial values
# n = 2
# a0 = [12, 12]
# n_sim = 1_000
# GBM = False
# mu_GBM = 0.10
# sigma_GBM = 0.25
# sigma_N = 5
# if GBM:
#     a1 = a0 * np.exp((mu_GBM - sigma_GBM**2) + sigma_GBM*np.random.normal(size=(n_sim, n)))             
# else:
#     a1 = a0 + np.random.normal(loc=[0]*n, scale=[sigma_N]*n, size=(n_sim, n))    

# # Compute default probabilities
# bankruptcies = np.zeros((n,1))
# conversions = np.zeros((n,1))
# healthy = np.zeros((n,1))
# all_healthy = 0
# stock_prices = np.zeros((n_sim,2))

# for i in range(n_sim): 
#    B, C, H, s_list = compute_equilibrium(l, c, m, W, a1[i, :])
#    stock_prices[i, ] = s_list
   
#    if len(H) == n:
#         all_healthy +=1

#    for j in range(c.size):
#         bankruptcies[j] += 1 if j in B else 0
#         conversions[j] += 1 if j in C else 0
#         healthy[j] += 1 if j in H else 0
    
# avg_bankrupt = bankruptcies / n_sim
# avg_conversion = conversions / n_sim
# avg_healthy = healthy / n_sim
# avgbh = all_healthy / n_sim

# print(f"Average P(bank i bankrupt): {avg_bankrupt.squeeze()}")
# print(f"Average P(bank i converts): {avg_conversion.squeeze()}")
# print(f"Average P(bank i healthy): {avg_healthy.squeeze()}")
# print(f"Average P(all healthy): {round(avgbh, 4)}")
# print(f"Avg. stock prices: {stock_prices.mean(axis=0)}")

## Plot example 2D in thesis text 
# Fair case
c = [6, 6]
m = [3/4, 3/4]
l = np.array(c) / np.array(m) 
w = [0.5, 0.5]
limit_up = 20
limit_down = -20

# generate_2d_figure(c, l, m, w, limit_up, limit_down, save=False)

# Sub-fair case
l_sub = l - 3 
# generate_2d_figure(c, l_sub, m, w, limit_up, limit_down, save=False)

# Super-fair case
l_sup = l + 3 
# generate_2d_figure(c, l_sup, m, w, limit_up, limit_down, save=False)

## Plot example 3D in thesis text
# Fair case
c = np.array([6, 6, 6])
m = np.array([3/4, 3/4, 3/4])
l = np.array(c) / np.array(m) 
W = np.array([[0, 1/2, 1/2], [1/2, 0, 1/2], [1/2, 1/2, 0]])
n_samples = 10_000

limit_up = 30
limit_down = -30

# All sets
# generate_3d_figure(c, l, m, W, limit_up, limit_down, n_samples, save=False, save_name='3dplot_total', seed=True)

# Emphasise CCC
# generate_3d_figure(c, l, m, W, limit_up, limit_down, n_samples, specific_set='CCC', save=True, save_name='3dplot_CCC', seed=True)

## Sensitivty to the model parameters
c = [10, 10]
m = [3/4, 3/4]
l = np.array(c) / np.array(m) 
w = [0.5, 0.5]
limit_up = 50
limit_down = -50

generate_2d_figure(c, l, m, w, limit_up, limit_down, save=False)
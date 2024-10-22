import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from helper_functions import compute_equilibrium

## Two period model
c = np.array([8, 8])
m = np.array([1, 1])
l = c / m   
W = np.array([[0, 0.75], [0.75, 0]])
e_a_2 = np.array([1, 1])
a_2 = np.array([16, 16])

def two_period_model(l, c, m, W, e_a_2, a_2):
    n = len(c)

    # First period: compute prices based on expected value
    B, C, H_1, s_1 = compute_equilibrium(l, c, m, W, e_a_2)
    C_1 = B + C   # banks cannot go bankrupt in period 1 

    # Second period: fix banks in C_1 and compute equilibrium
    B_2, C_2, H_2, s_2 = compute_equilibrium(l, c, m, W, a_2, pre_converted=C_1, verbose=False)

    return C_1, H_1, B_2, C_2, H_2, s_1, s_2

C_1, H_1, B_2, C_2, H_2, s_1, s_2 = two_period_model(l, c, m, W, e_a_2, a_2)
print([], C_1, H_1, s_1)
print(B_2, C_2, H_2, s_2)
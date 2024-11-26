import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from helper_functions import generate_2d_figure, generate_3d_figure
import pickle 
import time

## Plot example 2D
# Fair case
c = [6, 6]
m = [3/4, 3/4]
l = np.array(c) / np.array(m) 
w = [0.5, 0.5]
limit_up = 20
limit_down = -20

generate_2d_figure(c, l, m, w, limit_up, limit_down, save=False)

# Sub-fair case
l_sub = l - 3 
generate_2d_figure(c, l_sub, m, w, limit_up, limit_down, save=False)

# Super-fair case
l_sup = l + 3 
generate_2d_figure(c, l_sup, m, w, limit_up, limit_down, save=False)

## Plot example 3D
# Fair case
c = np.array([6, 6, 6])
m = np.array([3/4, 3/4, 3/4])
l = np.array(c) / np.array(m) 
W = np.array([[0, 1/2, 1/2], [1/2, 0, 1/2], [1/2, 1/2, 0]])
n_samples = 10_000

limit_up = 30
limit_down = -30

# All sets
generate_3d_figure(c, l, m, W, limit_up, limit_down, n_samples, save=False, save_name='3dplot_total', seed=True)

# Emphasise CCC
generate_3d_figure(c, l, m, W, limit_up, limit_down, n_samples, specific_set='CCC', save=True, save_name='3dplot_CCC', seed=True)





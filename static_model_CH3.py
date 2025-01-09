import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from helper_functions import generate_2d_figure, generate_3d_figure, generate_2d_figure_multi
import pickle 
import time
import seaborn as sns

## Plot example 2D
# Fair case
c = [6, 6]
m = [3/4, 3/4]
l = np.array(c) / np.array(m) 
w = [0.5, 0.5]
limit_up = 20
limit_down = -20

generate_2d_figure(c, l, m, w, limit_up, limit_down, save=True, save_name='fair2d')

# Sub-fair case
l_sub = l - 3 
generate_2d_figure(c, l_sub, m, w, limit_up, limit_down, save=True, save_name='subfair2d')

# Super-fair case
l_sup = l + 3 
generate_2d_figure(c, l_sup, m, w, limit_up, limit_down, save=True, save_name='superfair2d')

## Plot example 3D
np.random.seed(0)
# Fair case
c = np.array([6, 6, 6])
m = np.array([3/4, 3/4, 3/4])
l = np.array(c) / np.array(m) 
W = np.array([[0, 1/2, 1/2], [1/2, 0, 1/2], [1/2, 1/2, 0]])
n_samples = 100_000

limit_up = 30
limit_down = -30

# All sets
generate_3d_figure(c, l, m, W, limit_up, limit_down, n_samples, save=True, save_name='3dplot_total', seed=True)

# Emphasise CCC
generate_3d_figure(c, l, m, W, limit_up, limit_down, n_samples, specific_set='CCC', save=True, save_name='3dplot_CCC', seed=True)

## Multi-tranch model 
c_single = np.array([6, 6])
m_single = np.ones(2) * 0.75
c = np.vstack((c_single * 0.4, c_single * 0.6)).T
m = np.vstack((m_single*0.3, m_single*0.7)).T
l = c/m
W = np.tile(np.array([[0, 1/2], [1/2, 0]]), (2, 1, 1))

generate_2d_figure_multi(c, l, m, W, n_samples=100_000, save=True, save_name='multi_tranches_example')

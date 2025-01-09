import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sns
from helper_functions import compute_equilibrium
from scipy import stats

## Model parameters 
np.random.seed(0)
n = 3
c = np.ones(n) * 20
m = np.ones(n) * 1
l = c / m   
W = np.ones((n,n), dtype=int) * 1/(n-1)
np.fill_diagonal(W, 0)
T = 20

# Settings GBM
r = 0.0
sigma = np.array([0.04] * n)

a0_tilde = 40 + np.random.normal(1, 2, n)
d = 20
a_0 = a0_tilde - d

a_array = np.zeros((n, T+1))
a_array[:, 0] = a_0
for t in range(1, T+1):
    Z_mvn = np.random.normal(size=n)
    a_array[:, t] = (a_array[:, t-1] + d) * np.exp((r - 0.5*sigma**2) + sigma * Z_mvn) - d

# Compute time t prices 
n_sim_MC = 2000
s_array = np.zeros((n, T))
s_sd_array = np.zeros((n, T))
C_t = []
st = time.time()
for t in range(T):
    print(f"Iteration {t+1}/{T}", end='\r')
    if t < T-1: 
        s_sim = np.zeros((n, n_sim_MC))
        for k in range(n_sim_MC):
            Z_mvn_new = np.random.normal(size=(n, T-t))
            Z_mvn_new_cumulative = np.cumsum(Z_mvn_new, axis=1)
            aTk_atk = (a_array[:, t+1] + d) * np.exp((r - 0.5*sigma**2) * (T-t) + sigma * Z_mvn_new_cumulative[:, -1]) - d
            _, _, _, s = compute_equilibrium(l, c, m, W, aTk_atk, pre_converted=C_t)
            s_sim[:, k] = s
        s_array[:, t] = s_sim.mean(axis = 1)
        s_sd_array[:, t] = s_sim.std(axis = 1) / np.sqrt(n_sim_MC)
        C_t.extend([i for i in range(n) if s_array[i, t] <= l[i]])
    else:
        _, _, _, s = compute_equilibrium(l, c, m, W,  a_array[:, -1], pre_converted=C_t)
        s_array[:, t] = s
    it = time.time()
    print(f"{np.round(it - st, 2)} seconds", end='\r')
B, C, H, s = compute_equilibrium(l, c, m, W,  a=[28, 28, 21], pre_converted=[])
print(B,C,H,s)
cmap = plt.get_cmap('tab10')
alpha_CI = 0.975
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(n):
    ax.plot(np.arange(1, T+1), s_array[i, :], color = cmap(i), label=f"Stock price bank {i+1}")
    ax.plot(np.arange(1, T+1), a_array[i, 1:], color = cmap(i), label=f"Asset value bank {i+1}", linestyle='dashed')
    ci = stats.norm.ppf(alpha_CI) * s_sd_array[i, :]
    ax.fill_between(np.arange(1, T+1), (s_array[i, :]-ci), (s_array[i, :]+ci), color = cmap(i), alpha=.25)
    ax.plot(np.arange(1, T+1), np.ones(T)*l[0], color ='black', linestyle='dashed')
plt.legend()
plt.xticks(ticks=np.arange(1, T+1))
plt.title("Evolution of stock price and asset value")
plt.xlabel("Period ($t$)")
plt.ylabel("Stock price / Asset value")
save = True
plt.savefig('stockpricestotal2.png') if save else None
plt.show()

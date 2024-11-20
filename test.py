import numpy as np
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# from charset_normalizer import detect

# with open('dataset_EBA.csv', 'r', encoding='utf-8', errors='replace') as file:
#     df = pd.read_csv(file)

# C_COUNTRY_CODE_list = ["AT", "BE", "CY", "DE", "DK", "ES", "FI", "FR", "GB", "GR", "HU", "IE", "IT", "LU", "MT", "NL", "NO", "PL", "PT", "SE", "SI"]
# filter = df[(df['INFORMATION_CODE'] == 33010) & (df['C_COUNTRY_CODE'].isin(C_COUNTRY_CODE_list))]
# print(filter[["C_COUNTRY_CODE", "DE017"]])

mvnorm = stats.multivariate_normal(mean=[0, 0], cov=[[1., 0.5], 
                                                     [0.5, 1.]])
x = mvnorm.rvs(1_000)
# g = sns.jointplot(x=x[:, 0], y=x[:, 1], kind='kde', fill=True, cmap="Blues")
# g.set_axis_labels('X1', 'X2', fontsize=16);
# plt.show()

# u = stats.norm.cdf(x)
# x1_transformed = stats.beta(a=1, b=3).ppf(u[:, 0])
# x2_transformed = stats.beta(a=1, b=0.8).ppf(u[:, 1])
# h = sns.jointplot(x=x1_transformed, y=x2_transformed, kind='kde', fill=True, cmap="Blues", xlim=(-0.2, 1.2), ylim=(-0.2, 1.2))
# plt.show()

# x1 = stats.beta(a=1, b=2).rvs(1_000)
# x2 = stats.beta(a=1, b=2).rvs(1_000)
# h = sns.jointplot(x=x1, y=x2, kind='kde', fill=True, cmap="Blues", xlim=(-0.2, 1.2), ylim=(-0.2, 1.2))
# plt.show()

import networkx as nx
import matplotlib.pyplot as plt
 
# Generate a fully connected small-world network
while True:
    # Create a Watts-Strogatz small-world graph
    small_world = nx.watts_strogatz_graph(4, 2, 0.5)
    
    # Check if the graph is connected
    if nx.is_connected(small_world):
        break

# Draw the graph
plt.figure(figsize=(6, 6))
nx.draw(
    small_world, 
    with_labels=True, 
    node_color='lightgreen', 
    node_size=800, 
    font_weight='bold', 
    edge_color='gray'
)
plt.title("Small-World Network (Watts-Strogatz Model)", fontsize=14)
plt.show()
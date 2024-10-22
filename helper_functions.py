import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Polygon
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

def compute_equilibrium(l, c, m, W, a, pre_converted=None, verbose=False): 
    '''
    Returns the equilibrium stock prices of an interbank system with CoCo's that have market triggers.

            Parameters:
                    l (np.array): 1-D array of conversion thresholds
                    c (np.array): 1-D array of convertible debt values
                    m (np.array): 1-D array of received shares in case of conversion
                    W (np.array): 2-D array of interbank holding coefficients
                    a (np.array): 1-D array of asset values
                    verbose (bool): print information or not 
            
            Returns:
                    B, C, H, s_list (tuple): Tuple of the sets B, C, H, and the equilibrium stock prices
    '''     
    if any(l > c/m) and not any(l < c/m):
        case = "Super-fair case"
    elif any(l < c/m):
        case = "Sub-fair case"
    else:
        case = "Fair case"

    # Parameters and help variables
    M = 10**3
    n = len(l)
    I_min_W = np.identity(n) - W

    # Set up model
    model = gp.Model("CoCo-Model")
    model.setParam(gp.GRB.Param.OutputFlag, 0)  
    if case != "Fair case":  
        model.setParam(gp.GRB.Param.PoolSearchMode, 2)          # In the case l > c/m, model admits multiple solutions
        model.setParam(gp.GRB.Param.PoolSolutions, 3**n)        # Note: 3**n is an upperbound on the number of solutions        

    # Model variables
    s = model.addVars(n, vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY)
    delta_B = model.addVars(n, vtype=gp.GRB.BINARY, lb=0, ub=1)
    delta_C = model.addVars(n, vtype=gp.GRB.BINARY, lb=0, ub=1)
    delta_H = model.addVars(n, vtype=gp.GRB.BINARY, lb=0, ub=1)
    mu_B = model.addVars(n, vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY)
    mu_C = model.addVars(n, vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY)

    # Objective function 
    model.setObjective(gp.quicksum(delta_H[i] for i in range(n)), gp.GRB.MAXIMIZE)      # Note: objective is to optimize healthy banks, can also be set to a constant

    # Constraints
    for i in range(n):
        # Converted in previous period
        if (pre_converted is not None) and (i in pre_converted):
            model.addConstr(delta_H[i]  == 0)
        # Constraint on deltas
        model.addConstr(delta_B[i] + delta_C[i] + delta_H[i] == 1)      
        # Constraint on partition consistency
        model.addConstr(s[i] <= M * (1 - delta_B[i]))
        model.addConstr(-M * (1 - delta_C[i]) <= s[i])
        if (pre_converted is None) or (i not in pre_converted):
            model.addConstr(s[i] <= l[i] * delta_C[i] + M * (1 - delta_C[i]))
        model.addConstr(l[i] * delta_H[i] - M * (1 - delta_H[i]) <= s[i])
        # Auxiliary variables 
        model.addConstr(-M * (1 - delta_B[i]) + m[i]*s[i] <= mu_B[i])
        model.addConstr(mu_B[i] <= M * (1 - delta_B[i]) + m[i]*s[i])
        model.addConstr(-M * delta_B[i] <= mu_B[i])
        model.addConstr(mu_B[i] <= M * delta_B[i])
        model.addConstr(-M * (1 - delta_C[i]) + m[i]*s[i] <= mu_C[i])
        model.addConstr(mu_C[i] <= M * (1 - delta_C[i]) + m[i]*s[i])
        model.addConstr(-M * delta_C[i] <= mu_C[i])
        model.addConstr(mu_C[i] <= M * delta_C[i])
        # (B,C,H)-equilibrium constraint
        model.addConstr(a[i] == s[i] + mu_B[i] + gp.quicksum(I_min_W[i,j] * (mu_C[j] + c[j] * delta_H[j]) for j in range(n)))
    
    # Optimize
    model.optimize() 

    # Solutions
    if model.status == gp.GRB.OPTIMAL:
        n_sol = model.SolCount
        print(f"{case} - number of solutions found: {n_sol}") if verbose else None
        for i in range(n_sol):
            model.setParam(gp.GRB.Param.SolutionNumber, i)
            # Return partition and stock prices
            B = []
            C = []
            H = []
            s_list = []
            for i in range(n):
                s_list.append(s[i].Xn)
                if delta_B[i].Xn > 0.999:
                    B.append(i)
                elif delta_C[i].Xn > 0.999:
                    C.append(i)
                elif delta_H[i].Xn > 0.999:
                    H.append(i)
            print(f"Bankrupt: {[i+1 for i in B]}, Converting: {[i+1 for i in C]}, Healthy: {[i+1 for i in H]}") if verbose else None
            print(f"Stock price vector (theoretic): {[round(i, 5) for i in s_list]}") if verbose else None
            print(f"Stock price vector (economic): {[max(round(i, 5), 0) for i in s_list]}") if verbose else None
        
        return B, C, H, s_list
        
    if model.status == gp.GRB.INFEASIBLE:
        print(f"{case} - model is infeasible, no equilibrium found") 
        
        return ""

# TODO: consider edge cases 


def compute_equilibrium_multiple_thresholds(l, c, m, W, a):
    # Parameters and help variables
    M = 10**3
    n = l.shape[0]
    k = l.shape[1]
    I_min_W = np.broadcast_to(np.identity(n)[None, ...], (n,n,k)) - W

    # Check fairness:
    for i in range(n):
        if all(l[i, :] == c[i, :] / m[i, :]):
            case = "Fair case"
        else:
            case = "Not fair case"
            break
    print(case)

    # Set up model
    model = gp.Model("CoCo-Model-MultiThreshold")
    model.setParam(gp.GRB.Param.OutputFlag, 0)  
    if case != "fair":
        model.setParam(gp.GRB.Param.PoolSearchMode, 2)          # In the case l > c/m, model admits multiple solutions
        model.setParam(gp.GRB.Param.PoolSolutions, 3**n)        # Note: 3**n is an upperbound on the number of solutions   

    # Model variables
    s = model.addVars(n, vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY)
    delta_B = model.addVars(n, vtype=gp.GRB.BINARY, lb=0, ub=1)
    delta_G = model.addVars(n, k ,vtype=gp.GRB.BINARY, lb=0, ub=1)
    delta_H = model.addVars(n, vtype=gp.GRB.BINARY, lb=0, ub=1)

    delta_Ct = model.addVars(n, k, vtype=gp.GRB.BINARY, lb=0, ub=1)
    delta_Ht = model.addVars(n, k, vtype=gp.GRB.BINARY, lb=0, ub=1)

    mu_Bt = model.addVars(n, k, vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY)
    mu_Ct = model.addVars(n, k, vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY)

    # Objective function 
    model.setObjective(0, gp.GRB.MAXIMIZE) #gp.quicksum(delta_H[i] for i in range(n)), gp.GRB.MAXIMIZE)      # Note: objective is to optimize healthy banks, can also be set to a constant

    # Constraints
    for i in range(n):
        # Constraint on deltas
        model.addConstr(delta_B[i] + gp.quicksum(delta_G[i,t] for t in range(k)) + delta_H[i] == 1)  
        for t in range(k): 
            model.addConstr(delta_Ct[i, t] == gp.quicksum(delta_G[i,u] for u in range(t,k)))
            model.addConstr(delta_Ht[i, t] == delta_H[i] + gp.quicksum(delta_G[i,u] for u in range(t)))

        # Constraint on partition consistency
        model.addConstr(s[i] <= M * (1 - delta_B[i]))
        model.addConstr(s[i] >= -M * (1 - delta_G[i,k-1]))
        model.addConstr(s[i] <= l[i,(k-1)] * delta_G[i,(k-1)] + M * (1 - delta_G[i,(k-1)]))
        for t in range(k-1):
            model.addConstr(s[i] >= l[i, t+1]*delta_G[i,t] - M * (1 - delta_G[i,t]) )
            model.addConstr(s[i] <= l[i, t]*delta_G[i,t] + M * (1 - delta_G[i,t])) 
        model.addConstr(s[i] >= l[i,0] * delta_H[i] - M * (1 - delta_H[i]))

        # Auxiliary variables 
        for t in range(k):
            model.addConstr(mu_Bt[i,t] >= -M * (1 - delta_B[i]) + m[i,t]*s[i])
            model.addConstr(mu_Bt[i,t] <= M * (1 - delta_B[i]) + m[i,t]*s[i])
            model.addConstr(mu_Bt[i,t] >= -M * delta_B[i])
            model.addConstr(mu_Bt[i,t] <= M * delta_B[i])
            
            model.addConstr(mu_Ct[i,t] >= -M * (1 - delta_Ct[i,t]) + m[i,t]*s[i] )
            model.addConstr(mu_Ct[i,t] <= M * (1 - delta_Ct[i,t]) + m[i,t]*s[i])
            model.addConstr(mu_Ct[i,t] >= -M * delta_Ct[i,t])
            model.addConstr(mu_Ct[i,t] <= M * delta_Ct[i,t])
        # (B,C,H)-equilibrium constraint
        model.addConstr(a[i] == s[i] + gp.quicksum(mu_Bt[i,t] for t in range(k)) + gp.quicksum(gp.quicksum(I_min_W[t,i,j] * (mu_Ct[j,t] + c[j,t] * delta_Ht[j,t]) for j in range(n)) for t in range(k)))
                        
    # Optimizes
    model.optimize() 

    # Solutions
    if model.status == gp.GRB.OPTIMAL:
        n_sol = model.SolCount
        for i in range(n_sol):
            model.setParam(gp.GRB.Param.SolutionNumber, i)
            # Return partition and stock prices
            B = []
            G = 0
            G_1 = []
            G_2 = [] 
            H = []
            s_list = []
            for i in range(n):
                s_list.append(s[i].Xn)
                if delta_B[i].Xn > 0.999:
                    B.append(i)
                elif delta_H[i].Xn > 0.999:
                    H.append(i)
                elif delta_G[i,0].Xn > 0.999:
                    G_1.append(i)
                elif delta_G[i,1].Xn > 0.999:
                    G_2.append(i)
            print(f"B: {B}, G2: {G_2}, G1: {G_1}, H: {H}, stock prices: {np.round(np.array(s_list),4)}")
        
        return B, G, H, s_list
        
    if model.status == gp.GRB.INFEASIBLE:
        return "Model is infeasible"

# l = np.array([[10, 6], [10, 6]])
# c = np.array([[10, 6], [10, 6]])
# m = np.divide(c, l)
# a = [12, 20]
# W = np.array([[[0, 0.5], [0.5, 0]], [[0, 0.5], [0.5, 0]]])

# B, G, H, s_list = compute_equilibrium_multiple_thresholds(l, c, m, W, a)

def generate_problem_instance(n, case="fair"):

    '''
    Returns an instance of the problem

            Parameters:
                    n (integer): Number of banks
            
            Returns:
                    c, m, l, W
    '''     

    # Generate face value of CoCo, set m to 1
    c = np.random.uniform(2, 3, (n,1))
    m = np.ones((n,1))

    # Generate conversion threshold based on setting
    if case == "fair":
        l = c / m 
    elif case == "super-fair":
        l = np.random.uniform(3, 4, (n,1))
    elif case == "sub-fair":
        l = np.random.uniform(1, 2, (n,1))

    # Generate weight matrix with 0 diagonal and row sums < 1
    W = np.random.rand(n, n)
    np.fill_diagonal(W, 0)
    W /= W.sum(axis=1, keepdims=True) * np.random.uniform(1, 2, (n,1))

    return c.squeeze(), m.squeeze(), l.squeeze(), W

# Function to generate nice figures as in Balter et al. (2023)
def generate_2d_figure(c, l, m, w, limit_up, limit_down, show=True, save=False):
    # HH region
    corner_HH1 = (l[0] + c[0] - w[0]*c[1], l[1] + c[1] - w[1]*c[0])
    corner_HH2 = (l[0] + c[0] - w[0]*c[1], limit_up)
    corner_HH3 = (limit_up, l[1] + c[1] - w[1]*c[0])
    corner_HH4 = (limit_up, limit_up)
    ## HC region
    corner_HC1 = (l[0] + c[0], -w[1]*c[0])
    corner_HC2 = (limit_up, -w[1]*c[0])
    corner_HC3 = (l[0] + c[0] - w[0]*m[1]/(1+m[1])*((1+m[1])*l[1]), (1+m[1])*l[1] - w[1]*c[0]) 
    corner_HC4 = (limit_up, (1+m[1])*l[1] - w[1]*c[0]) 
    ## HB region
    corner_HB1 = (l[0] + c[0], -w[1]*c[0])
    corner_HB2 = (l[0] + c[0], limit_down)
    corner_HB3 = (limit_up, -w[1]*c[0])
    corner_HB4 = (limit_up, limit_down)

    ## BH region
    corner_BH1 = (limit_down, limit_up)
    corner_BH2 = (limit_down, l[1] + c[1])
    corner_BH3 = (-w[0]*c[1], l[1] + c[1])
    corner_BH4 = (-w[0]*c[1], limit_up)
    ## CH region
    corner_CH1 = (-w[0]*c[1], l[1]+c[1])
    corner_CH2 = (-w[0]*c[1], limit_up)
    corner_CH3 = ((1+m[0])*l[0] - w[0]*c[1], l[1]+c[1]-w[1]*m[0]/(1+m[0])*((1+m[0])*l[0]))
    corner_CH4 = ((1+m[0])*l[0] - w[0]*c[1], limit_up)
    ## CC region    
    X_1 = 1 + m[0] - w[1]*m[0]/(1+m[0]) * m[1] * w[0]
    X_2 = 1 + m[1] - w[0]*m[1]/(1+m[1]) * m[0] * w[1]
    
    c2_1 = X_2 * l[1] / (w[1]*m[0]/(1+m[1]) - (1+m[1])/(w[0]*m[1]))
    c2_2 = -c2_1 * (1+m[0]) / (w[0]*m[1])

    c3_1 = X_1 * l[0] * (1+m[0])/(w[0]*m[1]) / ((1+m[0])/(w[0]*m[1]) - (w[1]*m[0])/(1+m[1]))
    c3_2 = -c3_1 * (w[1]*m[0]) / (1+m[1])

    c4_1 = (X_2 * l[1] - X_1 * l[0] * (1+m[0])/(w[0]*m[1])) / ((w[1]*m[0])/(1+m[1]) - (1+m[0])/(w[0]*m[1]))
    c4_2 = X_2 * l[1] - (w[1]*m[0]) / (1+m[1]) * c4_1

    corner_CC1 = (0, 0)
    corner_CC2 = (c2_1, c2_2)
    corner_CC3 = (c3_1, c3_2)
    corner_CC4 = (c4_1, c4_2)

    ## CB region
    corner_CB1 = (0, 0)
    corner_CB2 = (0, limit_down)
    corner_CB3 = (l[0]*(1+m[0]), -w[1]*m[0]/(1+m[0])*l[0]*(1+m[0]))
    corner_CB4 = (l[0]*(1+m[0]), limit_down)
    ## BC region
    corner_BC1 = (0,0)
    corner_BC2 = (limit_down, 0)
    corner_BC3 = (-w[0]*m[1]/(1+m[1])*l[1]*(1+m[1]), l[1]*(1+m[1]))
    corner_BC4 = (limit_down, l[1]*(1+m[1]))
    ## BB region
    corner_BB1 = (0, limit_down)
    corner_BB2 = (0, 0)
    corner_BB3 = (limit_down, limit_down)
    corner_BB4 = (limit_down, 0)

    # Polygons
    polygonHH = [corner_HH1, corner_HH2, corner_HH4, corner_HH3] 
    polygonHB = [corner_HB1, corner_HB2, corner_HB4, corner_HB3] 
    polygonBH = [corner_BH1, corner_BH2, corner_BH3, corner_BH4]
    polygonBB = [corner_BB1, corner_BB2, corner_BB4, corner_BB3] 
    polygonHC = [corner_HC1, corner_HC2, corner_HC4, corner_HC3] 
    polygonCH = [corner_CH1, corner_CH2, corner_CH4, corner_CH3] 
    polygonCB = [corner_CB1, corner_CB2, corner_CB4, corner_CB3] 
    polygonBC = [corner_BC1, corner_BC2, corner_BC4, corner_BC3] 
    polygonCC = [corner_CC1, corner_CC2, corner_CC4, corner_CC3] 

    polygon_list = [polygonHH, polygonHB, polygonBH, polygonBB, polygonHC, polygonHC, polygonCB, polygonBC, polygonCC]
    color_list = [(0.6, 0.8, 1, 0.5), (0.6, 1, 0.6, 0.5), (1, 0.6, 0.6, 0.5), (1, 1, 0.6, 0.5), (1, 0.8, 0.8, 0.5), (0.8, 0.6, 1, 0.5), (1, 0.8, 0.6, 0.5), (0.6, 1, 0.8, 0.5), (0.8, 0.8, 0.8, 0.5)]

    _, ax = plt.subplots()
    for polygon, color in zip(polygon_list, color_list):
        polygon = Polygon(polygon, closed=True, edgecolor='black', facecolor = color, linewidth = 1)
        ax.add_patch(polygon)

    ax.set_aspect('equal')
    ax.set_xlim(limit_down, limit_up)
    ax.set_ylim(limit_down, limit_up)
    plt.xticks(range(limit_down, limit_up+1, 5))  
    plt.yticks(range(limit_down, limit_up+1, 5))  
    plt.xlabel("Asset value bank 1 ($a_1$)")
    plt.ylabel("Asset value bank 2 ($a_2$)")

    # Compute centrom of polygon 
    def calculate_centroid(points):
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        return (sum(x) / len(points), sum(y) / len(points))

    # Add labels in the middle of each polygon
    labels = ['HH', 'BB', 'CC', 'CB', 'BC', 'BH', 'HB', 'CH', 'HC']
    polygons = [polygonHH, polygonBB, polygonCC, polygonCB, polygonBC, polygonBH, polygonHB, polygonCH, polygonHC]

    for label, points in zip(labels, polygons):
        centroid = calculate_centroid(points)
        ax.text(centroid[0], centroid[1], label, ha='center', va='center', fontsize=10, color='black', fontweight='bold')

    for spine in ax.spines.values():
        spine.set_linewidth(1)  
        spine.set_color('black')  

    # Save/show
    plt.grid(False)
    if save:
        plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/w_high', dpi=400)
    if show:
        plt.show()

def generate_3d_figure(c, l, m, W, limit_up, limit_down, n_samples, specific_set=None, save=False, save_name='', seed=True):
    if seed:
        np.random.seed(0)

    # Generate random samples in 3D space
    random_points = np.random.uniform(low=limit_down, high=limit_up, size=(n_samples, 3))
    random_points[:, 2] = 2

    outcomes = []
    for i in range(n_samples):
        a = random_points[i, :]
        B, C, H, s_list = compute_equilibrium(l, c, m, W, a, verbose=False)
        str = ''
        for i in range(3):
            if s_list[i] < 0:
                str += 'B'
            elif s_list[i] >= 0 and s_list[i] <= l[i]:
                str += 'C'
            else:
                str += 'H'
        outcomes.append(str)

    # Assign colors to the set labels
    unique_labels = np.unique(outcomes)
    label_to_color = {label: color for label, color in zip(unique_labels, sns.color_palette("tab20", len(unique_labels)))}
    colors = [label_to_color[label] for label in outcomes]
    if specific_set:
        colors = [(0, 1, 0, 1.0) if label==specific_set else (0.827, 0.827, 0.827, 0.01) for label in outcomes]


    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sct_plot = ax.scatter(random_points[:, 0], random_points[:, 1], random_points[:, 2], c=colors)

    # Set labels
    ax.set_xlabel('Asset value bank 1 ($a_1$)')
    ax.set_ylabel('Asset value bank 2 ($a_2$)')
    ax.set_zlabel('Asset value bank 3 ($a_3$)')

    for label in unique_labels:
        ax.scatter([], [], [], color=label_to_color[label], label=label)
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(-0.35, 1), columnspacing=0.4, fontsize="small")

    if save:
        plt.savefig(f'/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/{save_name}', dpi=400)

    plt.show()

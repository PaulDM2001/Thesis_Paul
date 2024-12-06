import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
import pandas as pd

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
    M = 10**6
    n = len(l)
    I_min_W = np.identity(n) - W

    # Set up model
    model = gp.Model("CoCo-Model")
    model.setParam(gp.GRB.Param.OutputFlag, 0)  
    model.setParam('FeasibilityTol', 1e-9)

    if case != "Fair case":  
        model.setParam(gp.GRB.Param.PoolSearchMode, 2)          # In the case l > c/m, model admits multiple solutions
        model.setParam(gp.GRB.Param.PoolSolutions, 3**n)        # Note: 3**n is an upperbound on the number of solutions        

    # Model variables
    s = model.addVars(n, vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
    delta_B = model.addVars(n, vtype=gp.GRB.BINARY, lb=0, ub=1)
    delta_C = model.addVars(n, vtype=gp.GRB.BINARY, lb=0, ub=1)
    delta_H = model.addVars(n, vtype=gp.GRB.BINARY, lb=0, ub=1)
    mu_B = model.addVars(n, vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
    mu_C = model.addVars(n, vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)

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
                s_rounded = np.round(s_list[i], 8)
                if delta_B[i].Xn > 0.999 and s_rounded < 0:
                    B.append(i)
                elif delta_C[i].Xn > 0.999 or s_rounded == l[i] or s_rounded == 0:      #edge cases when s equals the boundary; the bank is then in C but num. precision errors may put it in H/B
                    C.append(i)
                elif delta_H[i].Xn > 0.999 and s_rounded > l[i]:
                    H.append(i)
                else:
                    print("Unassigned algo needed")
                    if s_rounded >= l[i] - 1e-6 and s_rounded < l[i]:
                        H.append(i) 
                    elif s_rounded >= 0 - 1e-6 and s_rounded < 0:
                        C.append(i)
                    else:
                        B.append(i)

            if len(B + C + H) != n:
                raise Exception("Unassigned banks!")

            print(f"Bankrupt: {[i+1 for i in B]}, Converting: {[i+1 for i in C]}, Healthy: {[i+1 for i in H]}") if verbose else None
            print(f"Stock price vector (theoretic): {[round(i, 5) for i in s_list]}") if verbose else None
            print(f"Stock price vector (economic): {[max(round(i, 5), 0) for i in s_list]}") if verbose else None
        

        return B, C, H, np.round(s_list, 8)
        
    if model.status == gp.GRB.INFEASIBLE:
        print(f"{case} - model is infeasible, no equilibrium found") if verbose else None
        
        return 0, 0, 0, 0


def compute_equilibrium_multiple_thresholds(l, c, m, W, a, s0, verbose=False):
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
    # Parameters and help variables
    M = s0[0] * (1 + m[0]) * 5        # Note, M cannot be chosen too large as to prevent the 'Trickle Flow' issue, see https://support.gurobi.com/hc/en-us/articles/16566259882129-How-do-I-diagnose-a-wrong-solution. 
    n = l.shape[0]
    k = l.shape[1]
    I_min_W =  np.tile(np.identity(n), (k, 1, 1)) - W

    # Check fairness:
    for i in range(n):
        if all(l[i, :] == np.round(c[i, :] / m[i, :], 6)):
            case = "Fair case"
        elif any(l[i, :] < np.round(c[i, :] / m[i, :],6)): 
            case = "Sub-fair case"
            break
        else:
            case = "Super-fair case"
    print(case) if verbose else None

    # Set up model
    model = gp.Model("CoCo-Model-MultiThreshold")
    model.setParam(gp.GRB.Param.OutputFlag, 0)  
    model.setParam('FeasibilityTol', 1e-9)
    if case != "Fair case":
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
            model.addConstr(s[i] >= l[i, t+1]*delta_G[i,t] - M * (1 - delta_G[i,t]))
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
        B_array = np.zeros((n, n_sol))
        G1_array = np.zeros((n, n_sol))
        G2_array = np.zeros((n, n_sol))
        H_array = np.zeros((n, n_sol))
        s_array = np.zeros((n, n_sol))

        for i in range(n_sol):
            model.setParam(gp.GRB.Param.SolutionNumber, i)
            # Return partition and stock prices
            for j in range(n):
                s_array[j, i] = s[j].Xn
                if delta_B[j].Xn > 0.999:
                    B_array[j, i] = 1
                elif delta_H[j].Xn > 0.999:
                    H_array[j, i] = 1
                elif delta_G[j, 0].Xn > 0.999:
                    G1_array[j, i] = 1
                elif delta_G[j, 1].Xn > 0.999:
                    G2_array[j, i] = 1
        
        if n_sol > 1 and case == "Fair case":   #handeling of numerical issues due to feasibility tolerance
            for k in range(n_sol-1):
                summed_diff_s = np.sum(np.abs(s_array[:, k+1] - s_array[:, k]))
                summed_diff_B = np.sum(np.abs(B_array[:, k+1] - B_array[:, k]))
                summed_diff_G1 = np.sum(np.abs(G1_array[:, k+1] - G1_array[:, k]))
                summed_diff_G2 = np.sum(np.abs(G2_array[:, k+1] - G2_array[:, k]))
                summed_diff_H = np.sum(np.abs(H_array[:, k+1] - H_array[:, k]))
                if summed_diff_s + summed_diff_B + summed_diff_G1 + summed_diff_G2 + summed_diff_H > 0.01:
                    raise Exception("Two different solutions in fair case. Check inputs.")
            B_array = B_array[:,:-1]
            G1_array = G1_array[:,:-1]
            G2_array = G2_array[:,:-1]
            H_array = H_array[:,:-1]
            s_array = s_array[:,:-1]
    
        return B_array, G2_array, G1_array, H_array, s_array
        
    if model.status == gp.GRB.INFEASIBLE:
        print(f"{case} - model is infeasible, no equilibrium found")
        
        return 0, 0, 0, 0, 0


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

def compute_systemic_risk_metrics(l, c, m, W, s_initial, shocked_banks, a_simulations):
    B_array = np.zeros((a_simulations.shape[0], a_simulations.shape[1]))
    C_array = np.zeros((a_simulations.shape[0], a_simulations.shape[1]))
    H_array = np.zeros((a_simulations.shape[0], a_simulations.shape[1]))
    Prb_array = np.zeros((3, a_simulations.shape[1]))
    L_creditors_array = np.zeros((a_simulations.shape[0], 1))
    L_creditors_contagion_array = np.zeros((a_simulations.shape[0], 1))
    L_equityholders_array = np.zeros((a_simulations.shape[0], 1))
    L_equityholders_contagion_array = np.zeros((a_simulations.shape[0], 1))

    nbInfeasible = 0
    for i in range(a_simulations.shape[0]):  
        # Solve problem
        B, C, H, s = compute_equilibrium(l, c, m, W, a_simulations[i, :], s_initial)
        if (B, C, H, s) == (0, 0, 0, 0):
            nbInfeasible += 1
            continue
        
        L_creditor = 0
        L_creditor_contagion = 0
        L_equityholder = 0
        L_equityholder_contagion = 0
        for j in range(a_simulations.shape[1]):

            B_array[i, j] = 1 if j in B else 0
            C_array[i, j] = 1 if j in C else 0
            H_array[i, j] = 1 if j in H else 0

            L_creditor += c[j] if j in B else 0
            L_creditor += (c[j] - m[j]*s[j]) if j in C else 0

            L_creditor_contagion += (c[j] - m[j]*s[j]) if j in C and j not in shocked_banks else 0

            L_equityholder += s[j] if j in H else 0
            L_equityholder += s[j] if j in C else 0
            L_equityholder_contagion += s[j] if j in H and j not in shocked_banks else 0
            L_equityholder_contagion += s[j] if j in C and j not in shocked_banks else 0

        # Metrics
        L_creditors_array[i] = L_creditor / np.sum(c)
        L_creditors_contagion_array[i] =  L_creditor_contagion / np.sum(c)
        L_equityholders_contagion_array[i] = ((np.sum(s_initial) - np.sum(s_initial[shocked_banks])) - L_equityholder_contagion) / np.sum(s_initial)
        L_equityholders_array[i] = (np.sum(s_initial) - L_equityholder) / np.sum(s_initial)
        

    Prb_array[0, :] = B_array.mean(axis=0)
    Prb_array[1, :] = C_array.mean(axis=0)
    Prb_array[2, :] = H_array.mean(axis=0)
    
    if nbInfeasible > 0:
        print(f"Warning: {nbInfeasible} simulations lead to an infeasible outcome of the MILP-solver.")
    
    return Prb_array, L_creditors_array, L_creditors_contagion_array, L_equityholders_array, L_equityholders_contagion_array


def compute_systemic_risk_metrics_multitranch(l, c, m, W, s_initial, shocked_banks, a_simulations):
    B_array = np.zeros((a_simulations.shape[0], a_simulations.shape[1]))
    G2_array = np.zeros((a_simulations.shape[0], a_simulations.shape[1]))
    G1_array = np.zeros((a_simulations.shape[0], a_simulations.shape[1]))
    H_array = np.zeros((a_simulations.shape[0], a_simulations.shape[1]))
    Prb_array = np.zeros((4, a_simulations.shape[1]))
    L_creditors_array = np.zeros((a_simulations.shape[0], 1))
    L_creditors_contagion_array = np.zeros((a_simulations.shape[0], 1))
    L_equityholders_array = np.zeros((a_simulations.shape[0], 1))
    L_equityholders_contagion_array = np.zeros((a_simulations.shape[0], 1))

    nbInfeasible = 0
    for i in range(a_simulations.shape[0]):  
        # Solve problem
        B, G2, G1, H, s = compute_equilibrium_multiple_thresholds(l, c, m, W, a_simulations[i, :], s_initial)
        if isinstance(s, int):
            nbInfeasible +=1
            continue
        
        B_array[i, :] = B.squeeze()
        G2_array[i, :] = G2.squeeze()
        G1_array[i, :] = G1.squeeze()
        H_array[i, :] = H.squeeze()

        L_creditor = 0
        L_equityholder = 0
        L_creditor_contagion = 0
        L_equityholder_contagion = 0
        for j in range(a_simulations.shape[1]):

            L_creditor += (c[j,0] + c[j,1]) if B[j] == 1 else 0
            L_creditor += (c[j,0] + c[j,1] - (m[j,0] + m[j,1])*s[j]) if G2[j] == 1 else 0
            L_creditor += (c[j,0] - m[j,0]*s[j]) if G1[j] == 1 else 0

            L_creditor_contagion += (c[j,0] + c[j,1]) if B[j] == 1 and j not in shocked_banks else 0
            L_creditor_contagion += (c[j,0] + c[j,1]  - (m[j,0] + m[j,1])*s[j]) if G2[j] == 1 and j not in shocked_banks else 0
            L_creditor_contagion += (c[j,0] - m[j,0]*s[j]) if G1[j] == 1 and j not in shocked_banks else 0

            L_equityholder += s[j] if ((H[j] == 1 or G1[j] ==1 or G2[j] ==1)) else 0
            L_equityholder_contagion += s[j] if ((H[j] == 1 or G1[j] ==1 or G2[j] ==1)) and j not in shocked_banks else 0

        # Metrics
        L_creditors_array[i] = L_creditor / np.sum(c)
        L_creditors_contagion_array[i] =  L_creditor_contagion / np.sum(c)
        L_equityholders_array[i] = (np.sum(s_initial) - L_equityholder) / np.sum(s_initial)
        L_equityholders_contagion_array[i] = ((np.sum(s_initial) - s_initial[shocked_banks]) - L_equityholder_contagion) / np.sum(s_initial)

    Prb_array[0, :] = B_array.mean(axis=0)
    Prb_array[1, :] = G2_array.mean(axis=0)
    Prb_array[2, :] = G1_array.mean(axis=0)
    Prb_array[3, :] = H_array.mean(axis=0)
    
    if nbInfeasible > 0:
        print(f"Warning: {nbInfeasible} simulations lead to an infeasible outcome of the MILP-solver.")
    
    return Prb_array, L_creditors_array, L_creditors_contagion_array, L_equityholders_array, L_equityholders_contagion_array


def compute_sensitivity_contract_parameters(n, W_variants, core_banks, shocked_banks, shocked_banks_periphery, b_rc, b_cp, X_shock, m_range, c_range, effect):
    P_total_ring = np.zeros((m_range.shape[0], 3, n))
    P_total_complete = np.zeros((m_range.shape[0], 3, n))
    P_total_cp = np.zeros((m_range.shape[0], 3, n))
    P_total_cp_p = np.zeros((m_range.shape[0], 3, n))

    L_creditors_ring = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_creditors_complete = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_creditors_cp = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_creditors_cp_p = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))

    L_creditors_contagion_ring = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_creditors_contagion_complete = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_creditors_contagion_cp = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_creditors_contagion_cp_p = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))

    L_equityholders_ring = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_equityholders_complete = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_equityholders_cp = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_equityholders_cp_p = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))

    L_equityholders_contagion_ring = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_equityholders_contagion_complete = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_equityholders_contagion_cp = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_equityholders_contagion_cp_p = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))

    st = time.time()
    et = time.time()
    for k, x in enumerate(m_range if effect == "m" else c_range):
        print(f"Iteration {k}/{m_range.shape[0]}, elapsed time: {np.round(et-st,2)} seconds", end='\r')
       
        # Determine parameters based on chosen effect
        if effect == "m": 
            m = np.ones(n) * x
            c_small = 12
            c_rc = np.ones(n) * c_small
            c_cp = np.ones(n) * c_small
            c_cp[core_banks] = c_small * (n - len(core_banks))

        elif effect == "c": 
            m = np.ones(n)
            c_rc = np.ones(n) * x
            c_cp = np.ones(n) * x
            c_cp[core_banks] = x * (n - len(core_banks))

        else:
            raise Exception("No valid effect chosen. Calculation stops")

        l_rc = np.round(c_rc / m, 6)
        l_cp = np.round(c_cp / m, 6)

        # Compute initial assets and shocked assets
        a_rc = np.tile(l_rc + c_rc - np.matmul(W_variants[0], c_rc) + b_rc, (X_shock.shape[0], 1))
        a_cp = np.tile(l_cp + c_cp - np.matmul(W_variants[2], c_cp) + b_cp, (X_shock.shape[0], 1))
        a_cp_p = np.tile(l_cp + c_cp - np.matmul(W_variants[2], c_cp) + b_cp, (X_shock.shape[0], 1))

        a_rc[:, shocked_banks] = np.minimum(X_shock, a_rc[:, shocked_banks])
        a_cp[:, shocked_banks] = np.minimum(X_shock * (n-len(core_banks)), a_cp[:, shocked_banks])
        a_cp_p[:, shocked_banks_periphery] = np.minimum(X_shock, a_cp_p[:, shocked_banks_periphery])

        # Compute initial prices
        s0_rc = l_rc + b_rc
        s0_cp = l_cp + b_cp

        # Compute systemic risk metrics
        Prb_array_ring, L_creditors_array_ring, L_creditors_contagion_array_ring, L_equityholders_array_ring, L_equityholders_contagion_array_ring = compute_systemic_risk_metrics(l_rc, c_rc, m, W_variants[0], s0_rc, shocked_banks, a_rc)
        Prb_array_cplt, L_creditors_array_cplt, L_creditors_contagion_array_cplt, L_equityholders_array_cplt, L_equityholders_contagion_array_cplt  = compute_systemic_risk_metrics(l_rc, c_rc, m, W_variants[1], s0_rc, shocked_banks, a_rc)
        Prb_array_cp, L_creditors_array_cp, L_creditors_contagion_array_cp, L_equityholders_array_cp, L_equityholders_contagion_array_cp  = compute_systemic_risk_metrics(l_cp, c_cp, m, W_variants[2], s0_cp, shocked_banks, a_cp)
        Prb_array_cp_p, L_creditors_array_cp_p, L_creditors_contagion_array_cp_p, L_equityholders_array_cp_p, L_equityholders_contagion_array_cp_p  = compute_systemic_risk_metrics(l_cp, c_cp, m, W_variants[2], s0_cp, shocked_banks_periphery, a_cp_p)

        # Store systemic risk metrics
        P_total_ring[k, :, :] = Prb_array_ring
        P_total_complete[k, :, :] = Prb_array_cplt
        P_total_cp[k, :, :] = Prb_array_cp
        P_total_cp_p[k, :, :] = Prb_array_cp_p

        L_creditors_ring[k] = L_creditors_array_ring.mean()
        L_creditors_complete[k] = L_creditors_array_cplt.mean()
        L_creditors_cp[k] = L_creditors_array_cp.mean()
        L_creditors_cp_p[k] = L_creditors_array_cp_p.mean()

        L_creditors_contagion_ring[k] = L_creditors_contagion_array_ring.mean()
        L_creditors_contagion_complete[k] = L_creditors_contagion_array_cplt.mean()
        L_creditors_contagion_cp[k] = L_creditors_contagion_array_cp.mean()
        L_creditors_contagion_cp_p[k] = L_creditors_contagion_array_cp_p.mean()

        L_equityholders_ring[k] = L_equityholders_array_ring.mean()
        L_equityholders_complete[k] = L_equityholders_array_cplt.mean()
        L_equityholders_cp[k] = L_equityholders_array_cp.mean()
        L_equityholders_cp_p[k] = L_equityholders_array_cp_p.mean()

        L_equityholders_contagion_ring[k] = L_equityholders_contagion_array_ring.mean()
        L_equityholders_contagion_complete[k] = L_equityholders_contagion_array_cplt.mean()
        L_equityholders_contagion_cp[k] = L_equityholders_contagion_array_cp.mean()
        L_equityholders_contagion_cp_p[k] = L_equityholders_contagion_array_cp_p.mean()
        
        et = time.time()

    return P_total_ring, P_total_complete, P_total_cp, P_total_cp_p, L_creditors_ring, L_creditors_complete, L_creditors_cp, L_creditors_cp_p, \
        L_creditors_contagion_ring, L_creditors_contagion_complete, L_creditors_contagion_cp, L_creditors_contagion_cp_p, L_equityholders_ring, \
            L_equityholders_complete, L_equityholders_cp, L_equityholders_cp_p, L_equityholders_contagion_ring, L_equityholders_contagion_complete, \
                L_equityholders_contagion_cp, L_equityholders_contagion_cp_p

def plot_results_sensitivity(m_range, c_range, effect, single, P_total_ring, P_total_complete, P_total_cp, P_total_cp_p, L_creditors_ring, L_creditors_complete, \
                             L_creditors_cp, L_creditors_cp_p, L_creditors_contagion_ring, L_creditors_contagion_complete, L_creditors_contagion_cp, \
                                L_creditors_contagion_cp_p, L_equityholders_ring, L_equityholders_complete, L_equityholders_cp, L_equityholders_cp_p, \
                                    L_equityholders_contagion_ring, L_equityholders_contagion_complete, L_equityholders_contagion_cp, L_equityholders_contagion_cp_p, save=False):
    
    # Plot settings
    plt.rc('axes', labelsize=12)       
    plt.rc('xtick', labelsize=10)       
    plt.rc('ytick', labelsize=10)      
    plt.rc('legend', fontsize='small')

    # Plot system level metrics
    plt.plot(m_range if effect == "m" else c_range, L_creditors_ring, linestyle=':')
    plt.plot(m_range if effect == "m" else c_range, L_creditors_complete, linestyle='-.')
    plt.plot(m_range if effect == "m" else c_range, L_creditors_cp, linestyle='--')
    plt.plot(m_range if effect == "m" else c_range, L_creditors_cp_p, linestyle='-')
    plt.legend(["Ring network", "Complete network", "Star network (core node)", "Star network (periphery node)"], fontsize='small')
    plt.title("Systemic losses for creditors (total)")
    plt.xlabel("Conversion rate $m$" if effect == "m" else "Issued convertible debt $c$")
    plt.ylabel(r'$L_{\mathrm{creditors, total}}$')
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/losscreditors', bbox_inches="tight", dpi=300) if save else None
    plt.show()

    plt.plot(m_range if effect == "m" else c_range, L_creditors_contagion_ring, linestyle=':')
    plt.plot(m_range if effect == "m" else c_range, L_creditors_contagion_complete, linestyle='-.')
    plt.plot(m_range if effect == "m" else c_range, L_creditors_contagion_cp, linestyle='--')
    plt.plot(m_range if effect == "m" else c_range, L_creditors_contagion_cp_p, linestyle='-')
    plt.legend(["Ring network", "Complete network", "Star network (core node)", "Star network (periphery node)"], fontsize='small')
    plt.title("Systemic losses for creditors (due to contagion)")
    plt.xlabel("Conversion rate $m$" if effect == "m" else "Issued convertible debt $c$")
    plt.ylabel(r'$L_{\mathrm{creditors, contagion}}$')
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/losscreditorscontagion', bbox_inches="tight", dpi=300) if save else None
    plt.show()

    plt.plot(m_range if effect == "m" else c_range, L_equityholders_ring, linestyle=':')
    plt.plot(m_range if effect == "m" else c_range, L_equityholders_complete, linestyle='-.')
    plt.plot(m_range if effect == "m" else c_range, L_equityholders_cp, linestyle='--')
    plt.plot(m_range if effect == "m" else c_range, L_equityholders_cp_p, linestyle='-')
    plt.legend(["Ring network", "Complete network", "Star network (core node)", "Star network (periphery node)"], fontsize='small')
    plt.title("Systemic losses for equityholders")
    plt.xlabel("Conversion rate $m$" if effect == "m" else "Issued convertible debt $c$")
    plt.ylabel(r'$L_{\mathrm{equityholders}}$')
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/lossequityholder', bbox_inches="tight", dpi=300) if save else None
    plt.show()

    plt.plot(m_range if effect == "m" else c_range, L_equityholders_contagion_ring, linestyle=':')
    plt.plot(m_range if effect == "m" else c_range, L_equityholders_contagion_complete, linestyle='-.')
    plt.plot(m_range if effect == "m" else c_range, L_equityholders_contagion_cp, linestyle='--')
    plt.plot(m_range if effect == "m" else c_range, L_equityholders_contagion_cp_p, linestyle='-')
    plt.legend(["Ring network", "Complete network", "Star network (core node)", "Star network (periphery node)"], fontsize='small')
    plt.title("Systemic losses for equityholders due to contagion")
    plt.xlabel("Conversion rate $m$" if effect == "m" else "Issued convertible debt $c$")
    plt.ylabel(r'$L_{\mathrm{equityholders, contagiion}}$')
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/lossequityholdercontagion', bbox_inches="tight", dpi=300) if save else None
    plt.show()
    
    # Plot probability metrics
    colors=['#6A1C1C', '#DAA520', '#006400', '#000080']
    for i in range(P_total_cp.shape[2]):
        plt.stackplot(m_range if effect == "m" else c_range, P_total_ring[:, :, i].T, colors=colors, alpha=0.75)
        plt.legend(["Bankrupt", "Conversion", "Healthy"] if single else ["Bankrupt", "Conversion Low", "Conversion High", "Healthy"]) 
        plt.title(f"Probability of bank {i+1} being in B, C and H (ring network)" if single else f"Probability of bank {i+1} being in B, CL, CH and H (ring network)")
        plt.xlabel("Conversion rate $m$" if effect == "m" else "Issued convertible debt $c$")
        plt.ylabel("Probability (cumulative)")
        plt.savefig(rf'/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/prbbank{i+1}ring', bbox_inches="tight", dpi=300) if save else None
        plt.show()

        plt.stackplot(m_range if effect == "m" else c_range, P_total_complete[:, :, i].T, colors=colors, alpha=0.75)
        plt.legend(["Bankrupt", "Conversion", "Healthy"] if single else ["Bankrupt", "Conversion Low", "Conversion High", "Healthy"]) 
        plt.title(f"Probability of bank {i+1} being in B, C and H (complete network)" if single else f"Probability of bank {i+1} being in B, CL, CH and H (complete network)")
        plt.xlabel("Conversion rate $m$" if effect == "m" else "Issued convertible debt $c$")
        plt.ylabel("Probability (cumulative)")
        plt.savefig(rf'/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/prbbank{i+1}complete', bbox_inches="tight", dpi=300) if save else None
        plt.show()

        plt.stackplot(m_range if effect == "m" else c_range, P_total_cp[:, :, i].T, colors=colors, alpha=0.75)
        plt.legend(["Bankrupt", "Conversion", "Healthy"] if single else ["Bankrupt", "Conversion Low", "Conversion High", "Healthy"]) 
        plt.title(f"Probability of bank {i+1} being in B, C and H (star network, core node)" if single else f"Probability of bank {i+1} being in B, CL, CH and H (star network, core node)")
        plt.xlabel("Conversion rate $m$" if effect == "m" else "Issued convertible debt $c$")
        plt.ylabel("Probability (cumulative)")
        plt.savefig(rf'/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/prbbank{i+1}coreperiphery', bbox_inches="tight", dpi=300) if save else None
        plt.show()

        plt.stackplot(m_range if effect == "m" else c_range, P_total_cp_p[:, :, i].T, colors=colors, alpha=0.75)
        plt.legend(["Bankrupt", "Conversion", "Healthy"] if single else ["Bankrupt", "Conversion Low", "Conversion High", "Healthy"]) 
        plt.title(f"Probability of bank {i+1} being in B, C and H (star network, periphery node)" if single else f"Probability of bank {i+1} being in B, CL, CH and H (star network, periphery node)")
        plt.xlabel("Conversion rate $m$" if effect == "m" else "Issued convertible debt $c$")
        plt.ylabel("Probability (cumulative)")
        plt.savefig(rf'/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/prbbank{i+1}coreperiphery_p', bbox_inches="tight", dpi=300) if save else None
        plt.show()
    
    return 


def compute_sensitivity_contract_parameters_multitranch(n, W_variants, zeta, xi, core_banks, shocked_banks, shocked_banks_periphery, b_rc, b_cp, X_shock, m_range, c_range, effect):
    P_total_ring = np.zeros((m_range.shape[0], 4, n))
    P_total_complete = np.zeros((m_range.shape[0], 4, n))
    P_total_cp = np.zeros((m_range.shape[0], 4, n))
    P_total_cp_p = np.zeros((m_range.shape[0], 4, n))

    L_creditors_ring = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_creditors_complete = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_creditors_cp = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_creditors_cp_p = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))

    L_creditors_contagion_ring = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_creditors_contagion_complete = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_creditors_contagion_cp = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_creditors_contagion_cp_p = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))

    L_equityholders_ring = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_equityholders_complete = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_equityholders_cp = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_equityholders_cp_p = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))

    L_equityholders_contagion_ring = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_equityholders_contagion_complete = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_equityholders_contagion_cp = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_equityholders_contagion_cp_p = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))

    st = time.time()
    et = time.time()
    for k, x in enumerate(m_range if effect == "m" else c_range):
        print(f"Iteration {k}/{m_range.shape[0]}, elapsed time: {np.round(et-st,2)} seconds", end='\r')
       
        # Determine parameters based on chosen effect
        if effect == "m": 
            m_total = np.ones(n) * x
            c_total = np.ones(n) * 12

        elif effect == "c": 
            m_total = np.ones(n) 
            c_total = np.ones(n) * x

        else:
            raise Exception("No valid effect chosen. Calculation stops")

        m_low = np.round(m_total * (1 - xi), 6)
        m_high = np.round(m_total * xi, 6)
        c_low = np.round(c_total * (1 - zeta), 6)
        c_high = np.round(c_total * zeta, 6)

        c_low_cp = c_total * (1 - zeta)
        c_high_cp = c_total * zeta
        c_low_cp[core_banks] *= (n - len(core_banks))
        c_high_cp[core_banks] *= (n - len(core_banks))
        c_low_cp = np.round(c_low_cp, 6)
        c_high_cp = np.round(c_high_cp, 6)

        l_low = np.round(c_low / m_low, 6)
        l_high = np.round(c_high / m_high, 6)
        l_low_cp = np.round(c_low_cp / m_low, 6)
        l_high_cp = np.round(c_high_cp / m_high, 6)

        l_rc = np.vstack((l_high, l_low)).T
        l_cp = np.vstack((l_high_cp, l_low_cp)).T
        m = np.vstack((m_high, m_low)).T
        c_rc = np.vstack((c_high, c_low)).T
        c_cp = np.vstack((c_high_cp, c_low_cp)).T

        # Compute initial assets and shocked assets
        a_rc = np.tile(l_high + c_high + c_low - np.matmul(W_variants[0], c_high) - np.matmul(W_variants[0], c_low) + b_rc, (X_shock.shape[0], 1))
        a_cp = np.tile(l_high_cp + c_high_cp + c_low_cp - np.matmul(W_variants[2], c_high_cp) - np.matmul(W_variants[2], c_low_cp) + b_cp, (X_shock.shape[0], 1)) 
        a_cp_p = np.tile(l_high_cp + c_high_cp + c_low_cp - np.matmul(W_variants[2], c_high_cp) - np.matmul(W_variants[2], c_low_cp) + b_cp, (X_shock.shape[0], 1)) 

        a_rc[:, shocked_banks] = np.minimum(X_shock, a_rc[:, shocked_banks])
        a_cp[:, shocked_banks] = np.minimum(X_shock * (n-1), a_cp[:, shocked_banks])
        a_cp_p[:, shocked_banks_periphery] = np.minimum(X_shock, a_cp_p[:, shocked_banks_periphery])
        s0_rc = l_high + b_rc
        s0_cp = l_high_cp + b_cp

        # Compute systemic risk metrics        
        Prb_array_ring, L_creditors_array_ring, L_creditors_contagion_array_ring, L_equityholders_array_ring, L_equityholders_contagion_array_ring = compute_systemic_risk_metrics_multitranch(l_rc, c_rc, m,  np.tile(W_variants[0], (2, 1, 1)), s0_rc, shocked_banks, a_rc)
        Prb_array_cplt, L_creditors_array_cplt, L_creditors_contagion_array_cplt, L_equityholders_array_cplt, L_equityholders_contagion_array_cplt  = compute_systemic_risk_metrics_multitranch(l_rc, c_rc, m, np.tile(W_variants[1], (2, 1, 1)), s0_rc, shocked_banks, a_rc)
        Prb_array_cp, L_creditors_array_cp, L_creditors_contagion_array_cp, L_equityholders_array_cp, L_equityholders_contagion_array_cp = compute_systemic_risk_metrics_multitranch(l_cp, c_cp, m, np.tile(W_variants[2], (2, 1, 1)), s0_cp, shocked_banks, a_cp)
        Prb_array_cp_p, L_creditors_array_cp_p, L_creditors_contagion_array_cp_p, L_equityholders_array_cp_p, L_equityholders_contagion_array_cp_p = compute_systemic_risk_metrics_multitranch(l_cp, c_cp, m, np.tile(W_variants[2], (2, 1, 1)), s0_cp, shocked_banks_periphery, a_cp_p)

        # Store systemic risk metrics
        P_total_ring[k, :, :] = Prb_array_ring
        P_total_complete[k, :, :] = Prb_array_cplt
        P_total_cp[k, :, :] = Prb_array_cp
        P_total_cp_p[k, :, :] = Prb_array_cp_p

        L_creditors_ring[k] = L_creditors_array_ring.mean()
        L_creditors_complete[k] = L_creditors_array_cplt.mean()
        L_creditors_cp[k] = L_creditors_array_cp.mean()
        L_creditors_cp_p[k] = L_creditors_array_cp_p.mean()

        L_creditors_contagion_ring[k] = L_creditors_contagion_array_ring.mean()
        L_creditors_contagion_complete[k] = L_creditors_contagion_array_cplt.mean()
        L_creditors_contagion_cp[k] = L_creditors_contagion_array_cp.mean()
        L_creditors_contagion_cp_p[k] = L_creditors_contagion_array_cp_p.mean()

        L_equityholders_ring[k] = L_equityholders_array_ring.mean()
        L_equityholders_complete[k] = L_equityholders_array_cplt.mean()
        L_equityholders_cp[k] = L_equityholders_array_cp.mean()
        L_equityholders_cp_p[k] = L_equityholders_array_cp_p.mean()

        L_equityholders_contagion_ring[k] = L_equityholders_contagion_array_ring.mean()
        L_equityholders_contagion_complete[k] = L_equityholders_contagion_array_cplt.mean()
        L_equityholders_contagion_cp[k] = L_equityholders_contagion_array_cp.mean()
        L_equityholders_contagion_cp_p[k] = L_equityholders_contagion_array_cp_p.mean()
        
        et = time.time()

    return P_total_ring, P_total_complete, P_total_cp, P_total_cp_p, L_creditors_ring, L_creditors_complete, L_creditors_cp, L_creditors_cp_p, \
        L_creditors_contagion_ring, L_creditors_contagion_complete, L_creditors_contagion_cp, L_creditors_contagion_cp_p, L_equityholders_ring, \
            L_equityholders_complete, L_equityholders_cp, L_equityholders_cp_p, L_equityholders_contagion_ring, L_equityholders_contagion_complete, \
                L_equityholders_contagion_cp, L_equityholders_contagion_cp_p


def simulate_shocks_correlated(n_shocks, beta, rho):
    '''
    Returns shocks to gross asset values according to Gaussian Copula 

            Parameters:
                    n_shocks (int): number of scenarios 
                    beta (np.array): 1-D array of asset quality indices
                    rho (float): correlation parameter in [0, 1] 
    
            Returns:
                    x_shock (np.array): N-D array of N shocks for D banks
    '''     
    n = beta.shape[0]
    mu_mvn = [0]* n 
    Sigma_mvn =  np.ones((n, n)) * rho
    np.fill_diagonal(Sigma_mvn, 1)
    Y = stats.multivariate_normal(mean=mu_mvn, cov=Sigma_mvn).rvs(n_shocks)
    U = stats.norm.cdf(Y)
    X_shock = np.zeros((n_shocks, n))
    for i in range(n):
        X_shock[:, i] =  stats.beta(a=1, b=beta[i]).ppf(U[:, i])

    return X_shock

def import_weight_matrices_R(filename):
    W_collection = []
    for i in range(10):
        path = filename + rf"_{i+1}.csv"
        W_matrix_i = pd.read_csv(path, index_col=0).to_numpy()
        W_matrix_i_normalized = W_matrix_i / np.sum(W_matrix_i, axis=0)
        W_collection.append(W_matrix_i_normalized)
    
    return W_collection
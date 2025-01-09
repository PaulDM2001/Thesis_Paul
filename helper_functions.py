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
    if np.any(np.round(l,6) > np.round(c/m,6)) and not np.any(np.round(l,6) < np.round(c/m,6)):
        case = "Super-fair case"
    elif np.any(np.round(l,6) < np.round(c/m,6)):
        case = "Sub-fair case"
    else:
        case = "Fair case"

    # Parameters and help variables
    M = 10 ** 6
    n = len(l)
    I_min_W = np.identity(n) - W

    # Set up model
    model = gp.Model("CoCo-Model")
    model.setParam(gp.GRB.Param.OutputFlag, 0)  
    model.setParam('FeasibilityTol', 1e-9)

    # Note, we do search for multiple solutions here even in the fair case as we check for numerical issues manually at the
    # end of this function 
    # if case != "Fair case":  
    model.setParam(gp.GRB.Param.PoolSearchMode, 2)          # In the case l > c/m, model admits multiple solutions
    model.setParam(gp.GRB.Param.PoolSolutions, 3**4)        # Note: 3**n is an upperbound on the number of solutions        

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

        i_not_assigned = [] 
        for i in range(n_sol):
            model.setParam(gp.GRB.Param.SolutionNumber, i)
            # Return partition and stock prices
            B_array = np.zeros((n, n_sol))
            C_array = np.zeros((n, n_sol))
            H_array = np.zeros((n, n_sol))
            s_array = np.zeros((n, n_sol))
            for j in range(n):
                s_rounded = np.round(s[j].Xn, 6)
                s_array[j, i] = s_rounded
                if delta_B[j].Xn > 0.999 and s_rounded < 0:
                    B_array[j, i] = 1
                elif delta_C[j].Xn > 0.999 or s_rounded == l[j] or s_rounded == 0:      # Edge cases when s equals the boundary; the bank is then in C but num. precision errors may put it in H/B
                    C_array[j, i] = 1
                elif delta_H[j].Xn > 0.999 and s_rounded > l[j]:    
                    H_array[j, i] = 1
                
            # Check if all banks are assigned to a set (numerical issues may prevent this)
            all_assigned = np.sum(B_array[:, i] + C_array[:, i] + H_array[:, i])
            if np.sum(all_assigned) != n:
                i_not_assigned.append(i)
                print(f"\n Not all banks assigned in solution {i}\{n_sol}.")

        i_assigned = [i for i in range(n) if i not in i_not_assigned]
        if n_sol > 1 and case == "Fair case":   #handeling of numerical issues due to feasibility tolerance
            print("Multiple solutions")
            avg_s = np.mean(s_array, axis=1)
            dev_from_avg = s_array - avg_s[:, np.newaxis]
            print(dev_from_avg)
            print(s_array)
            if np.any(np.abs(dev_from_avg) > 1e-4):
                print("Deviation more than 1e-4")
                
            # Taking the first solution where all banks are assigned as correct solution and printing to check if this is correct 
            i_star = i_assigned[0]
            B_array = B_array[:,i_star]
            C_array = C_array[:,i_star]
            H_array = H_array[:,i_star]
            s_array = s_array[:,i_star]
            print(B_array)
            print(C_array)
            print(H_array)

        return B_array.squeeze(), C_array.squeeze(), H_array.squeeze(), s_array.squeeze()
        
    if model.status == gp.GRB.INFEASIBLE:
        print(f"{case} - model is infeasible, no equilibrium found") if verbose else None
        
        return 0, 0, 0, 0


def compute_equilibrium_multiple_thresholds(l, c, m, W, a, s0=None, verbose=False, combination_multi_single=False):
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
    n = l.shape[0]
    k = l.shape[1]
    if s0 is not None:
        M = s0 * (1 + np.sum(m, axis = 1)) * 5                           # Note, M cannot be chosen too large as to prevent the 'Trickle Flow' issue, see https://support.gurobi.com/hc/en-us/articles/16566259882129-How-do-I-diagnose-a-wrong-solution. 
    else:
        M = np.ones(n) * 10 ** 4
    I_min_W =  np.tile(np.identity(n), (k, 1, 1)) - W

    # Check fairness:
    if combination_multi_single:
        case = "Fair case"
    elif np.all(np.round(l, 6) == np.round(c / m, 6)):
        case = "Fair case"
    elif np.any(np.round(l, 6) < np.round(c / m,6 )):
        case = "Sub-fair case"
    else:
        case = "Super-fair case"
    print(case) if verbose else None

    # Set up model
    model = gp.Model("CoCo-Model-MultiThreshold")
    model.setParam(gp.GRB.Param.OutputFlag, 0)  
    model.setParam('FeasibilityTol', 1e-9)
    
    # Note, we do search for multiple solutions here even in the fair case as we check for numerical issues manually at the
    # end of this function 
    #if case != "Fair case": 
    model.setParam(gp.GRB.Param.PoolSearchMode, 2)          # In the case l > c/m, model admits multiple solutions
    model.setParam(gp.GRB.Param.PoolSolutions, 3**4)        # Note: 3**n is an upperbound on the number of solutions   

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
        if combination_multi_single:
            # Extra constraint to prevent numerical issues when some banks have only a single-tranch CoCo 
            model.addConstr(delta_G[i, 1] == 0) if m[i, 1] == 0 else None        

        # Constraint on deltas
        model.addConstr(delta_B[i] + gp.quicksum(delta_G[i,t] for t in range(k)) + delta_H[i] == 1)  
        for t in range(k): 
            model.addConstr(delta_Ct[i, t] == gp.quicksum(delta_G[i,u] for u in range(t,k)))
            model.addConstr(delta_Ht[i, t] == delta_H[i] + gp.quicksum(delta_G[i,u] for u in range(t)))

        # Constraint on partition consistency
        model.addConstr(s[i] <= M[i] * (1 - delta_B[i]))
        model.addConstr(s[i] >= -M[i] * (1 - delta_G[i,k-1]))
        model.addConstr(s[i] <= l[i,(k-1)] * delta_G[i,(k-1)] + M[i] * (1 - delta_G[i,(k-1)]))
        for t in range(k-1):
            model.addConstr(s[i] >= l[i, t+1]*delta_G[i,t] - M[i] * (1 - delta_G[i,t]))
            model.addConstr(s[i] <= l[i, t]*delta_G[i,t] + M[i] * (1 - delta_G[i,t])) 
        model.addConstr(s[i] >= l[i,0] * delta_H[i] - M[i] * (1 - delta_H[i]))          

        # Auxiliary variables 
        for t in range(k):
            model.addConstr(mu_Bt[i,t] >= -M[i] * (1 - delta_B[i]) + m[i,t]*s[i])
            model.addConstr(mu_Bt[i,t] <= M[i] * (1 - delta_B[i]) + m[i,t]*s[i])
            model.addConstr(mu_Bt[i,t] >= -M[i] * delta_B[i])
            model.addConstr(mu_Bt[i,t] <= M[i] * delta_B[i])
            
            model.addConstr(mu_Ct[i,t] >= -M[i] * (1 - delta_Ct[i,t]) + m[i,t]*s[i] )
            model.addConstr(mu_Ct[i,t] <= M[i] * (1 - delta_Ct[i,t]) + m[i,t]*s[i])
            model.addConstr(mu_Ct[i,t] >= -M[i] * delta_Ct[i,t])
            model.addConstr(mu_Ct[i,t] <= M[i] * delta_Ct[i,t])
        # (B,C,H)-equilibrium constraint
        model.addConstr(a[i] == s[i] + gp.quicksum(mu_Bt[i,t] for t in range(k)) + gp.quicksum(gp.quicksum(I_min_W[t,i,j] * (mu_Ct[j,t] + c[j,t] * delta_Ht[j,t]) for j in range(n)) for t in range(k)))
                        
    # Optimizes
    model.optimize() 

    # Solutions
    if model.status == gp.GRB.OPTIMAL:
        n_sol = model.SolCount 
        B_array = np.zeros((n, n_sol))
        G2_array = np.zeros((n, n_sol))
        G1_array = np.zeros((n, n_sol))
        H_array = np.zeros((n, n_sol))
        s_array = np.zeros((n, n_sol))

        i_not_assigned = [] 
        for i in range(n_sol):
            model.setParam(gp.GRB.Param.SolutionNumber, i)
            # Return partition and stock prices
            for j in range(n):
                s_rounded = np.round(s[j].Xn, 6)
                s_array[j, i] = s_rounded
                if delta_B[j].Xn > 0.999 and s_rounded < 0:
                    B_array[j, i] = 1
                elif delta_H[j].Xn > 0.999 and s_rounded > l[j, 0]:
                    H_array[j, i] = 1
                elif delta_G[j, 0].Xn > 0.999 and s_rounded <= l[j, 0] and s_rounded > l[j, 1]:
                    G1_array[j, i] = 1
                elif delta_G[j, 1].Xn > 0.999 and s_rounded <= l[j, 1] and s_rounded > 0:
                    G2_array[j, i] = 1
            
            # Check if all banks are assigned to a set (numerical issues may prevent this)
            all_assigned = np.sum(B_array[:, i] + H_array[:, i] + G1_array[:, i] + G2_array[:, i])
            if np.sum(all_assigned) != n:
                i_not_assigned.append(i)
                print(f"\n Not all banks assigned in solution {i}.")

        i_assigned = [i for i in range(n) if i not in i_not_assigned]
        if n_sol > 1 and case == "Fair case":   #handeling of numerical issues due to feasibility tolerance
            print("Multiple solutions")
            avg_s = np.mean(s_array, axis=1)
            dev_from_avg = s_array - avg_s[:, np.newaxis]
            print(dev_from_avg)
            if np.any(np.abs(dev_from_avg) > 1e-4):
                print("Deviation more than 1e-4")
            # Taking the first solution where all banks are assigned as correct solution and printing to check if this is correct 
            i_star = i_assigned[0]
            B_array = B_array[:,i_star]
            G1_array = G1_array[:,i_star]
            G2_array = G2_array[:,i_star]
            H_array = H_array[:,i_star]
            s_array = s_array[:,i_star]
            print(s_array)
            print(B_array)
            print(G2_array)
            print(G1_array)
            print(H_array)

        return B_array.squeeze(), G2_array.squeeze(), G1_array.squeeze(), H_array.squeeze(), s_array.squeeze()
        
    if model.status == gp.GRB.INFEASIBLE:
        print(f"{case} - model is infeasible, no equilibrium found")
        
        return 0, 0, 0, 0, 0


# Function to generate nice figures as in Balter et al. (2023)
def generate_2d_figure(c, l, m, w, limit_up, limit_down, show=True, save=False, save_name=None):
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

    polygon_list = [polygonHH, polygonHB, polygonBH, polygonBB, polygonHC, polygonCH, polygonCB, polygonBC, polygonCC]
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
    plt.rc('axes', labelsize=12)       
    plt.rc('xtick', labelsize=10)       
    plt.rc('ytick', labelsize=10)      
    if save:
        plt.tight_layout()
        plt.savefig(fr'Images/Images CH3/{save_name}', dpi=400)
    if show:
        plt.show()



def generate_3d_figure(c, l, m, W, limit_up, limit_down, n_samples, specific_set=None, save=False, save_name='', seed=True):
    if seed:
        np.random.seed(0)

    # Generate random samples in 3D space
    random_points = np.random.uniform(low=limit_down, high=limit_up, size=(n_samples, 3))
    outcomes = []
    for i in range(n_samples):
        if i % 10000 == 0:
            print(i)
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
        plt.savefig(f'Images/Images CH3/{save_name}', dpi=400)

    plt.show()



def generate_2d_figure_multi(c, l, m, W, limit_up=30, limit_down=-30, n_samples=100_000, specific_set=None, save=False, save_name='', seed=True):
    if seed:
        np.random.seed(0)

    # Generate random samples in 2D space
    random_points = np.random.uniform(low=limit_down, high=limit_up, size=(n_samples, 2))
    outcomes = []
    for i in range(n_samples):
        if i % 10000 == 0:
            print(i)
        a = random_points[i, :]
        B, CL, CH, H, s_list = compute_equilibrium_multiple_thresholds(l, c, m, W, a, verbose=False)
        str = ''
        for i in range(2):
            if s_list[i] < 0:
                str += 'B'
            elif s_list[i] >= 0 and s_list[i] <= l[i, 1]:
                str += 'C1'
            elif s_list[i] > l[i, 1] and s_list[i] <= l[i, 0]:
                str += 'C2'
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
    ax = fig.add_subplot()
    ax.set_aspect('equal')
    ax.set_xlim(limit_down, limit_up)
    ax.set_ylim(limit_down, limit_up)
    plt.xticks(range(limit_down, limit_up+1, 5))  
    plt.yticks(range(limit_down, limit_up+1, 5))  
    plt.xlabel("Asset value bank 1 ($a_1$)")
    plt.ylabel("Asset value bank 2 ($a_2$)")
    sct_plot = ax.scatter(random_points[:, 0], random_points[:, 1], c=colors, s=1.5)

    for label in unique_labels:
        ax.scatter([], [], [], color=label_to_color[label], label=label)
    legend = ax.legend(loc="upper right", ncol=1, bbox_to_anchor=(1.25, 1), columnspacing=0.4, fontsize="small")
    for handle in legend.legend_handles:
        handle.set_sizes([9.0])

    if save:
        plt.savefig(f'Images/Images CH3/{save_name}', dpi=400)

    plt.show()

    return 0



def generate_network(n, type, fixed=True, weights_fixed = 1.0, core_nodes=[1], gamma_mixed=0.5):
    # Ring network
    adj_matrix_ring = np.roll(np.eye(n), 1, axis=1)

    # Core-pheriphery network 
    if type == "core periphery" and core_nodes is None:
        raise Exception("Core periphery network is selected but list of core nodes is empty")
    else: 
        adj_matrix_core_periphery = np.zeros((n, n), dtype=int)
        for i in core_nodes:
            adj_matrix_core_periphery[i, :] = 1
            adj_matrix_core_periphery[:, i] = 1
        np.fill_diagonal(adj_matrix_core_periphery, 0)

    # Complete Network
    adj_matrix_complete = np.ones((n,n), dtype=int)
    np.fill_diagonal(adj_matrix_complete, 0)

    if fixed:
        W_ring = adj_matrix_ring * weights_fixed
        W_core_periphery = np.zeros((n, n))
        for i in range(n):
            if i in core_nodes:
                W_core_periphery[:, i] = adj_matrix_core_periphery[:, i] * weights_fixed / (n-1)
                W_core_periphery[i, :] = adj_matrix_core_periphery[i, :] * weights_fixed 
        W_complete = adj_matrix_complete * weights_fixed/(n-1) 
    else:
        random_weights = np.random.rand(n, n)
        W_ring = np.round(adj_matrix_ring * random_weights, 4)
        W_complete = adj_matrix_complete * np.round(random_weights / np.sum(random_weights, axis=0), 2)

    # Convex combination of Ring and Complete
    W_mixed = gamma_mixed * W_ring + (1 - gamma_mixed) * W_complete

    W_types = {"ring": W_ring, "core periphery": W_core_periphery, "complete": W_complete, "mixed": W_mixed}

    return W_types[type]



def plot_results_sensitivity(m_range, c_range, effect, single, P_total_ring, P_total_complete, P_total_cp, P_total_cp_p, L_creditors_ring, L_creditors_complete, L_creditors_cp, L_creditors_cp_p, \
        L_creditors_contagion_ring, L_creditors_contagion_complete, L_creditors_contagion_cp, L_creditors_contagion_cp_p, L_creditors_direct_ring, \
        L_creditors_direct_complete, L_creditors_direct_cp, L_creditors_direct_cp_p, L_equityholders_ring, L_equityholders_complete, L_equityholders_cp, \
        L_equityholders_cp_p, L_equityholders_contagion_ring, L_equityholders_contagion_complete, L_equityholders_contagion_cp, L_equityholders_contagion_cp_p, \
        L_equityholders_direct_ring, L_equityholders_direct_complete, L_equityholders_direct_cp, L_equityholders_direct_cp_p, save=False):
    
    # Plot settings
    plt.rc('axes', labelsize=12)       
    plt.rc('xtick', labelsize=10)       
    plt.rc('ytick', labelsize=10)      
    plt.rc('legend', fontsize='small')

    # Plot system level metrics
    plt.plot(m_range if effect == "m" else c_range, L_creditors_ring.mean(axis=1), linestyle=':')
    plt.plot(m_range if effect == "m" else c_range, L_creditors_complete.mean(axis=1), linestyle='-.')
    plt.plot(m_range if effect == "m" else c_range, L_creditors_cp.mean(axis=1), linestyle='--')
    plt.plot(m_range if effect == "m" else c_range, L_creditors_cp_p.mean(axis=1), linestyle='-')
    plt.legend(["Ring network", "Complete network", "Star network (shock c-node)", "Star network (shock p-node)"], fontsize='small')
    plt.xlabel("Conversion rate $m_{1}$" if effect == "m" else "Issued convertible debt $c_{1}$")
    plt.ylabel(r'$L_{\mathrm{Creditors}}$')
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/losscreditors', bbox_inches="tight", dpi=300) if save else None
    plt.show()

    plt.plot(m_range if effect == "m" else c_range, L_creditors_contagion_ring.mean(axis=1), linestyle=':')
    plt.plot(m_range if effect == "m" else c_range, L_creditors_contagion_complete.mean(axis=1), linestyle='-.')
    plt.plot(m_range if effect == "m" else c_range, L_creditors_contagion_cp.mean(axis=1), linestyle='--')
    plt.plot(m_range if effect == "m" else c_range, L_creditors_contagion_cp_p.mean(axis=1), linestyle='-')
    plt.legend(["Ring network", "Complete network", "Star network (shock c-node)", "Star network (shock p-node)"], fontsize='small')
    plt.xlabel("Conversion rate $m_{1}$" if effect == "m" else "Issued convertible debt $c_{1}$")
    plt.ylabel(r'$L_{\mathrm{CreditorsC}}$')
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/losscreditorscontagion', bbox_inches="tight", dpi=300) if save else None
    plt.show()

    plt.plot(m_range if effect == "m" else c_range, L_creditors_direct_ring.mean(axis=1), linestyle=':')
    plt.plot(m_range if effect == "m" else c_range, L_creditors_direct_complete.mean(axis=1), linestyle='-.')
    plt.plot(m_range if effect == "m" else c_range, L_creditors_direct_cp.mean(axis=1), linestyle='--')
    plt.plot(m_range if effect == "m" else c_range, L_creditors_direct_cp_p.mean(axis=1), linestyle='-')
    plt.legend(["Ring network", "Complete network", "Star network (shock c-node)", "Star network (shock p-node)"], fontsize='small')
    plt.xlabel("Conversion rate $m_{1}$" if effect == "m" else "Issued convertible debt $c_{1}$")
    plt.ylabel(r'$L_{\mathrm{CreditorsD}}$')
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/losscreditorsdirect', bbox_inches="tight", dpi=300) if save else None
    plt.show()

    plt.plot(m_range if effect == "m" else c_range, L_equityholders_ring.mean(axis=1), linestyle=':')
    plt.plot(m_range if effect == "m" else c_range, L_equityholders_complete.mean(axis=1), linestyle='-.')
    plt.plot(m_range if effect == "m" else c_range, L_equityholders_cp.mean(axis=1), linestyle='--')
    plt.plot(m_range if effect == "m" else c_range, L_equityholders_cp_p.mean(axis=1), linestyle='-')
    plt.legend(["Ring network", "Complete network", "Star network (shock c-node)", "Star network (shock p-node)"], fontsize='small')
    plt.xlabel("Conversion rate $m_{1}$" if effect == "m" else "Issued convertible debt $c_{1}$")
    plt.ylabel(r'$L_{\mathrm{EQH}}$')
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/lossequityholder', bbox_inches="tight", dpi=300) if save else None
    plt.show()

    plt.plot(m_range if effect == "m" else c_range, L_equityholders_contagion_ring.mean(axis=1), linestyle=':')
    plt.plot(m_range if effect == "m" else c_range, L_equityholders_contagion_complete.mean(axis=1), linestyle='-.')
    plt.plot(m_range if effect == "m" else c_range, L_equityholders_contagion_cp.mean(axis=1), linestyle='--')
    plt.plot(m_range if effect == "m" else c_range, L_equityholders_contagion_cp_p.mean(axis=1), linestyle='-')
    plt.legend(["Ring network", "Complete network", "Star network (shock c-node)", "Star network (shock p-node)"], fontsize='small')
    plt.xlabel("Conversion rate $m_{1}$" if effect == "m" else "Issued convertible debt $c_{1}$")
    plt.ylabel(r'$L_{\mathrm{EQHC}}$')
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/lossequityholdercontagion', bbox_inches="tight", dpi=300) if save else None
    plt.show()
    
    plt.plot(m_range if effect == "m" else c_range, L_equityholders_direct_ring.mean(axis=1), linestyle=':')
    plt.plot(m_range if effect == "m" else c_range, L_equityholders_direct_complete.mean(axis=1), linestyle='-.')
    plt.plot(m_range if effect == "m" else c_range, L_equityholders_direct_cp.mean(axis=1), linestyle='--')
    plt.plot(m_range if effect == "m" else c_range, L_equityholders_direct_cp_p.mean(axis=1), linestyle='-')
    plt.legend(["Ring network", "Complete network", "Star network (shock c-node)", "Star network (shock p-node)"], fontsize='small')
    plt.xlabel("Conversion rate $m_{1}$" if effect == "m" else "Issued convertible debt $c_{1}$")
    plt.ylabel(r'$L_{\mathrm{EQHD}}$')
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/lossequityholderdirect', bbox_inches="tight", dpi=300) if save else None
    plt.show()

    # Plot probability metrics
    colors=['#6A1C1C', '#DAA520', '#006400', '#000080']
    for i in range(P_total_cp.shape[2]):
        plt.stackplot(m_range if effect == "m" else c_range, P_total_ring[:, :, i].T, colors=colors, alpha=0.75)
        plt.legend(["Bankrupt", "Conversion", "Healthy"] if single else ["Bankrupt", "Conversion Low", "Conversion High", "Healthy"]) 
        plt.title(f"Probability of bank {i+1} being in B, C and H (ring)" if single else f"Probability of bank {i+1} being in B, CL, CH and H (ring)")
        plt.xlabel("Conversion rate $m_{1}$" if effect == "m" else "Issued convertible debt $c_{1}$")
        plt.ylabel("Probability (cumulative)")
        plt.savefig(rf'/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/prbbank{i+1}ring', bbox_inches="tight", dpi=300) if save else None
        plt.show()

        plt.stackplot(m_range if effect == "m" else c_range, P_total_complete[:, :, i].T, colors=colors, alpha=0.75)
        plt.legend(["Bankrupt", "Conversion", "Healthy"] if single else ["Bankrupt", "Conversion Low", "Conversion High", "Healthy"]) 
        plt.title(f"Probability of bank {i+1} being in B, C and H (complete)" if single else f"Probability of bank {i+1} being in B, CL, CH and H (complete)")
        plt.xlabel("Conversion rate $m_{1}$" if effect == "m" else "Issued convertible debt $c_{1}$")
        plt.ylabel("Probability (cumulative)")
        plt.savefig(rf'/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/prbbank{i+1}complete', bbox_inches="tight", dpi=300) if save else None
        plt.show()

        plt.stackplot(m_range if effect == "m" else c_range, P_total_cp[:, :, i].T, colors=colors, alpha=0.75)
        plt.legend(["Bankrupt", "Conversion", "Healthy"] if single else ["Bankrupt", "Conversion Low", "Conversion High", "Healthy"]) 
        plt.title(f"Probability of bank {i+1} being in B, C and H (star network, shock c)" if single else f"Probability of bank {i+1} being in B, CL, CH and H (star network, shock c)")
        plt.xlabel("Conversion rate $m_{1}$" if effect == "m" else "Issued convertible debt $c_{1}$")
        plt.ylabel("Probability (cumulative)")
        plt.savefig(rf'/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/prbbank{i+1}coreperiphery', bbox_inches="tight", dpi=300) if save else None
        plt.show()

        plt.stackplot(m_range if effect == "m" else c_range, P_total_cp_p[:, :, i].T, colors=colors, alpha=0.75)
        plt.legend(["Bankrupt", "Conversion", "Healthy"] if single else ["Bankrupt", "Conversion Low", "Conversion High", "Healthy"]) 
        plt.title(f"Probability of bank {i+1} being in B, C and H (star network, shock p)" if single else f"Probability of bank {i+1} being in B, CL, CH and H (star network, shock p)")
        plt.xlabel("Conversion rate $m_{1}$" if effect == "m" else "Issued convertible debt $c_{1}$")
        plt.ylabel("Probability (cumulative)")
        plt.savefig(rf'/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/prbbank{i+1}coreperiphery_p', bbox_inches="tight", dpi=300) if save else None
        plt.show()
    
    return 0


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

def import_weight_matrices_R(filename, num_matrices=10):
    W_collection = []
    for i in range(num_matrices):
        if num_matrices > 1:
            path = filename + rf"_{i+1}.csv"
        else:
            path = filename
        W_matrix_i = pd.read_csv(path, index_col=0).to_numpy()
        W_matrix_i_normalized = W_matrix_i / np.sum(W_matrix_i, axis=0)
        W_collection.append(W_matrix_i_normalized)
    
    return W_collection
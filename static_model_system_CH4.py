import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from helper_functions import compute_equilibrium, import_weight_matrices_R
import itertools
import pickle

# Import generated W matrices from Csv-files generated in R
W_matrices_GV_cp = import_weight_matrices_R("Data EBA/W_matrix_GV_cp")                  #import liability matrices from csv files generated in R 

# Import other data and define model parameters 
dataset_EBA = pd.read_csv('Data EBA/dataset_EBA_cleaned.csv', index_col=0) / 1000       #scale by 1000 to prevent numerical issues

### Analysis 1: Fix shock X and vary m + gamma 
gamma_INT_range = np.linspace(0.0, 1.0, 21)
m_range = np.linspace(0, 2, 11)
m_range[0] = 0.01
n_combinations = gamma_INT_range.shape[0] * m_range.shape[0]

V_creditor_array = np.zeros((gamma_INT_range.shape[0], m_range.shape[0], len(W_matrices_GV_cp)))
V_creditor_ext_array = np.zeros((gamma_INT_range.shape[0], m_range.shape[0], len(W_matrices_GV_cp)))
V_equityholder_gain_B_array = np.zeros((gamma_INT_range.shape[0], m_range.shape[0], len(W_matrices_GV_cp)))
V_equityholder_loss_H_array = np.zeros((gamma_INT_range.shape[0], m_range.shape[0], len(W_matrices_GV_cp)))
V_equityholder_array = np.zeros((gamma_INT_range.shape[0], m_range.shape[0], len(W_matrices_GV_cp)))
fraction_bankrupt_array = np.zeros((gamma_INT_range.shape[0], m_range.shape[0], len(W_matrices_GV_cp)))
fraction_converting_array = np.zeros((gamma_INT_range.shape[0], m_range.shape[0], len(W_matrices_GV_cp)))
fraction_healthy_array = np.zeros((gamma_INT_range.shape[0], m_range.shape[0], len(W_matrices_GV_cp)))

compute_results = False
EXT_CoCo = False                    # external debt CoCo-ized or not

# Compute stock prices and sets when there is no CoCo-ization
n = dataset_EBA["Total External Assets"].shape[0]
c = np.zeros(n)
m = np.ones(n) 
l = c / m
a = ext_assets_shocked = dataset_EBA["Total External Assets"].to_numpy() * (1 - 0.03) - dataset_EBA["Total External Liabilities"].to_numpy()
B_noCoCo, C_noCoCo, H_noCoCo, s_noCoCo = compute_equilibrium(np.zeros(n), np.zeros(n), np.ones(n), np.zeros((n,n)), a)

core_banks = np.argsort(-dataset_EBA["Total Interbank Liabilities"].to_numpy())[:8]
non_core_banks = np.argsort(-dataset_EBA["Total Interbank Liabilities"].to_numpy())[8:]

# Vary the degree of CoCo-ization
if compute_results: 
    k=0
    for gamma_x, m_x in itertools.product(gamma_INT_range, m_range):
        index_gamma = k // m_range.shape[0]
        index_m = k - m_range.shape[0] * index_gamma
        print(f"Iteration {k+1}/{n_combinations}, {(gamma_x, m_x)}")
        gamma_INT = gamma_x                             # fraction of interbank liabilities CoCo-ized
        gamma_EXT = gamma_x if EXT_CoCo else 0          # fraction of external liabilities CoCo-ized
        X_shock = 0.03
        R_rate = 1/2
        c = dataset_EBA["Total Interbank Liabilities"].to_numpy() * gamma_INT + dataset_EBA["Total External Liabilities"].to_numpy() * gamma_EXT
        ext_assets_shocked = (dataset_EBA["Total External Assets"].to_numpy() * (1 - X_shock) + dataset_EBA["Total Interbank Assets"].to_numpy() * (1 - gamma_INT)) 
        ext_liabilities = (dataset_EBA["Total External Liabilities"].to_numpy() * (1-gamma_EXT) + dataset_EBA["Total Interbank Liabilities"].to_numpy() * (1-gamma_INT))  

        fraction_CoCo_liabilities_int = (dataset_EBA["Total Interbank Liabilities"].to_numpy() * gamma_INT ) / c if sum(c) != 0 else 0

        for w in range(1): #range(len(W_matrices_GV_cp)):
            n = dataset_EBA["Total External Assets"].shape[0]
            W_GV_corrected = W_matrices_GV_cp[w] * fraction_CoCo_liabilities_int

            a = (ext_assets_shocked - ext_liabilities) 
            n = W_matrices_GV_cp[w].shape[0]
            m = np.ones(n) * m_x       
            l = c / m   

            B, C, H, s = compute_equilibrium(l, c, m, W_GV_corrected, a)
                                                    
            if isinstance(s, int):
                print("Infeasible")
                print((gamma_x, m_x))
                continue

            V_creditor = np.sum(c * H + m*s * C) / np.sum(c) if np.sum(c) != 0 else 0
            V_equityholder_gain_B = np.sum(np.maximum(s[core_banks],0)) #np.sum(np.maximum(s,0) * B_noCoCo)
            V_equityholder_loss_H = np.sum(np.maximum(s[non_core_banks],0)) #np.sum(np.maximum(s[core_banks],0)) #np.sum(s * H_noCoCo - s_noCoCo*H_noCoCo)
            V_equityholder_total = np.sum(np.maximum(s,0)) #np.sum(np.maximum(s[non_core_banks],0))
            V_creditor_ext = np.sum(np.minimum(ext_assets_shocked * R_rate, ext_liabilities) * B
                                 + ext_liabilities * (H+C)
                                 + c * (1-fraction_CoCo_liabilities_int) * H
                                 + m*s * (1-fraction_CoCo_liabilities_int) * C) / np.sum(ext_liabilities + c * (1-fraction_CoCo_liabilities_int))

            fr_b = np.sum(B)/n
            fr_c = np.sum(C)/n
            fr_h = np.sum(H)/n
            sum_fr = fr_b + fr_c + fr_h
            if np.round(sum_fr, 6) != 1:
                print(sum_fr)
                print("Fractions do not add up, check inputs.")

            V_creditor_array[index_gamma, index_m, w] = V_creditor 
            V_creditor_ext_array[index_gamma, index_m, w] = V_creditor_ext
            V_equityholder_gain_B_array[index_gamma, index_m, w] =  V_equityholder_gain_B
            V_equityholder_loss_H_array[index_gamma, index_m, w] =  V_equityholder_loss_H
            V_equityholder_array[index_gamma, index_m, w] = V_equityholder_total
            fraction_bankrupt_array[index_gamma, index_m, w]  = fr_b 
            fraction_converting_array[index_gamma, index_m, w] = fr_c 
            fraction_healthy_array[index_gamma, index_m, w] = fr_h 
        k +=1
    
save_results = False
load_results = False
filename = 'Results_4_EBA_external_CoCo'
if save_results:
    with open(filename , 'wb') as f:
        pickle.dump([V_creditor_array, V_creditor_ext_array, V_equityholder_array, fraction_bankrupt_array, fraction_converting_array, fraction_healthy_array], f)
if load_results:
    with open(filename , 'rb') as f:
        V_creditor_array, V_creditor_ext_array, V_equityholder_array, fraction_bankrupt_array, fraction_converting_array, fraction_healthy_array = pickle.load(f)

# Plot of recovery rate and optimal gamma
if filename == 'Results_4_EBA_external_CoCo_RecRate':
    avg_V_creditor = V_creditor_ext_array.mean(axis=2)
    gamma_opt_list = []
    for i in range(11):
        gamma_max_idx = np.argmax(avg_V_creditor[:, i])
        gamma_max = gamma_INT_range[gamma_max_idx]
        gamma_opt_list.append(gamma_max)

    plt.plot(np.linspace(0, 1, 11), gamma_opt_list,'-o')
    plt.xticks(np.linspace(0, 1, 11))
    plt.yticks([0.0, 0.005, 0.01, 0.015])
    plt.xlabel(r"Recovery rate $R_i$")
    plt.ylabel(r"Optimal value of $\gamma$")
    plt.show()

plotting = False if filename != 'Results_4_EBA_external_CoCo_RecRate' else False
save = False
if plotting: 
    # Plot settings
    plt.rc('axes', labelsize=18)       
    plt.rc('xtick', labelsize=12)       
    plt.rc('ytick', labelsize=12)      
    plt.rc('legend', fontsize='small')

    X, Y = np.meshgrid(gamma_INT_range, m_range, indexing='ij')

    Z1 = V_creditor_array.mean(axis=2) * 10
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z1, cmap='viridis', edgecolor='none')
    plt.title(r"Fraction of initial CoCo value remaining in a 3% shock scenario")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$m$")
    ax.set_zlabel(r"$V_{\mathrm{creditors}}$")
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/EBA_3d_LCR', bbox_inches="tight", dpi=300) if save else None
    plt.show()

    Z2 = V_creditor_ext_array.mean(axis=2) * 10
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z2, cmap='viridis', edgecolor='none')
    plt.title(r"Fraction of remaining value for external creditors in a 3% shock scenario")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$m$")
    ax.set_zlabel(r"$V_{\mathrm{ext. creditors}}$")
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/EBA_3d_LCRext', bbox_inches="tight", dpi=300) if save else None
    plt.show()

    Z3 = V_equityholder_array.mean(axis=2) * 10
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z3, cmap='viridis', edgecolor='none')
    plt.title(r"Total value of the original shares in a 3% shock scenario")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$m$")
    ax.set_zlabel(r"$V_{\mathrm{eq.holders}}$")
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/EBA_3d_LEQ', bbox_inches="tight", dpi=300) if save else None
    plt.show()

    Z3a = V_equityholder_gain_B_array.mean(axis=2) * 10
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z3a, cmap='viridis', edgecolor='none')
    plt.title(r"Total value of the original shares in a 3% shock scenario")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$m$")
    ax.set_zlabel(r"$V_{\mathrm{eq.holders}}$")
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/EBA_3d_LEQ', bbox_inches="tight", dpi=300) if save else None
    plt.show()

    Z3b = V_equityholder_loss_H_array.mean(axis=2) * 10
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z3b, cmap='viridis', edgecolor='none')
    plt.title(r"Total value of the original shares in a 3% shock scenario")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$m$")
    ax.set_zlabel(r"$V_{\mathrm{eq.holders}}$")
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/EBA_3d_LEQ', bbox_inches="tight", dpi=300) if save else None
    plt.show()

    Z4 = fraction_bankrupt_array.mean(axis=2) * 10
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z4, cmap='viridis', edgecolor='none')
    plt.title(r"Fraction of banks defaulting in a 3% shock scenario")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$m$")
    ax.set_zlabel(r"$f_{D}$")
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/EBA_3d_fd', bbox_inches="tight", dpi=300) if save else None
    plt.show()

    Z5 = fraction_converting_array.mean(axis=2) * 10
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z5, cmap='viridis', edgecolor='none')
    plt.title(r"Fraction of banks converting in a 3% shock scenario")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$m$")
    ax.set_zlabel(r"$f_{C}$")
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/EBA_3d_fc', bbox_inches="tight", dpi=300) if save else None
    plt.show()

### Analysis 2: Fixed m + scenarios gamma and vary shock X 

# Scenarios
weakInt_CoCo_ization = {"gamma_internal" : 0.05, "gamma_external" : 0.0}
fullInt_CoCo_ization = {"gamma_internal" : 1.00, "gamma_external" : 0.0}
fullIntweakExt_CoCo_ization = {"gamma_internal" : 1.0, "gamma_external" : 0.05}
fullIntfullExt_CoCo_ization = {"gamma_internal" : 1.0, "gamma_external" : 1.0}
n_values_X = 100
X_shock_range = np.linspace(0, 0.25, n_values_X)

n = W_matrices_GV_cp[0].shape[0]
m = np.ones(n) 

def compute_metrics_scenario(n, m, W_matrices, scenario_gamma, scenarios_X):
    V_CoCo_debt_array = np.zeros((scenarios_X.shape[0], len(W_matrices_GV_cp)))
    V_ext_array = np.zeros((scenarios_X.shape[0], len(W_matrices_GV_cp)))
    V_Equity_array= np.zeros((scenarios_X.shape[0], len(W_matrices_GV_cp)))
    Bankrupt_proportion_array = np.zeros((scenarios_X.shape[0], len(W_matrices_GV_cp)))
    Converting_proportion_array = np.zeros((scenarios_X.shape[0], len(W_matrices_GV_cp)))
    Healthy_proportion_array = np.zeros((scenarios_X.shape[0], len(W_matrices_GV_cp)))

    c = dataset_EBA["Total Interbank Liabilities"].to_numpy() * scenario_gamma["gamma_internal"] + dataset_EBA["Total External Liabilities"].to_numpy() * scenario_gamma["gamma_external"]
    l = c / m   
    correction_factor_W = (dataset_EBA["Total Interbank Liabilities"].to_numpy() * scenario_gamma["gamma_internal"] ) / c if sum(c) != 0 else 0
    R_rate = 1/3

    for w in range(len(W_matrices)):
        print(f"Iteration {w}/{len(W_matrices)}", end='\r')
        W_matrix_corrected = W_matrices[w] * correction_factor_W
        for k, X in enumerate(scenarios_X): 
            if k % 10 == 0:
                print(k)
            ext_assets_shocked = (dataset_EBA["Total External Assets"].to_numpy() + dataset_EBA["Total Interbank Assets"].to_numpy() * (1 - scenario_gamma["gamma_internal"])) * (1 - X)
            ext_liabilities = (dataset_EBA["Total External Liabilities"].to_numpy() * (1-scenario_gamma["gamma_external"]) + dataset_EBA["Total Interbank Liabilities"].to_numpy() * (1-scenario_gamma["gamma_internal"]))       
            a = ext_assets_shocked - ext_liabilities     
            B, C, H, s = compute_equilibrium(l, c, m, W_matrix_corrected, a)
            if isinstance(s, int):
                print("Infeasible")
                continue

            V_CoCo_debt_array[k, w] = np.sum(c * H + m*s * C) / sum(c) if sum(c) != 0 else 0
            V_ext_array[k, w] = np.sum(np.minimum(ext_assets_shocked * R_rate, ext_liabilities) * B
                                 + ext_liabilities * (C+H)
                                 + c * (1-correction_factor_W) * H
                                 + m*s * (1-correction_factor_W) * C) / np.sum(ext_liabilities + c * (1-correction_factor_W))
            V_Equity_array[k, w] = np.sum(np.maximum(s, 0))
            Bankrupt_proportion_array[k, w]  = np.sum(B)/n
            Converting_proportion_array[k, w] = np.sum(C)/n
            Healthy_proportion_array[k, w] = np.sum(H)/n
            sum_fr = Bankrupt_proportion_array[k, w] + Converting_proportion_array[k, w] + Healthy_proportion_array[k, w]
            if np.round(sum_fr, 6) != 1:
                print(sum_fr)
                print("Error")
    
    return V_CoCo_debt_array, V_ext_array, V_Equity_array, Bankrupt_proportion_array, Converting_proportion_array, Healthy_proportion_array

compute_results = False
if compute_results:
    V_CoCo_debt_array_weakINT, V_ext_array_weakINT, V_Equity_array_weakINT, Bankrupt_proportion_array_weakINT, Converting_proportion_array_weakINT, Healthy_proportion_array_weakINT = compute_metrics_scenario(n, m, W_matrices_GV_cp, weakInt_CoCo_ization, X_shock_range)
    V_CoCo_debt_array_fullINT, V_ext_array_fullINT, V_Equity_array_fullINT, Bankrupt_proportion_array_fullINT, Converting_proportion_array_fullINT, Healthy_proportion_array_fullINT = compute_metrics_scenario(n, m, W_matrices_GV_cp, fullInt_CoCo_ization, X_shock_range)
    V_CoCo_debt_array_fullINTweakEXT, V_ext_array_fullINTweakEXT, V_Equity_array_fullINTweakEXT, Bankrupt_proportion_array_fullINTweakEXT, Converting_proportion_array_fullINTweakEXT, Healthy_proportion_array_fullINTweakEXT = compute_metrics_scenario(n, m, W_matrices_GV_cp, fullIntweakExt_CoCo_ization, X_shock_range)
    V_CoCo_debt_array_fullINTfullEXT, V_ext_array_fullINTfullEXT, V_Equity_array_fullINTfullEXT, Bankrupt_proportion_array_fullINTfullEXT, Converting_proportion_array_fullINTfullEXT, Healthy_proportion_array_fullINTfullEXT= compute_metrics_scenario(n, m, W_matrices_GV_cp, fullIntfullExt_CoCo_ization, X_shock_range)

save_results = False
load_results = True
if save_results:
    with open('Analysis EBA/Results_CH4_EBAdataset_varyShock_lowerR', 'wb') as f:
        pickle.dump([V_CoCo_debt_array_weakINT, V_ext_array_weakINT, V_Equity_array_weakINT, Bankrupt_proportion_array_weakINT, Converting_proportion_array_weakINT, Healthy_proportion_array_weakINT,
                     V_CoCo_debt_array_fullINT, V_ext_array_fullINT, V_Equity_array_fullINT, Bankrupt_proportion_array_fullINT, Converting_proportion_array_fullINT, Healthy_proportion_array_fullINT,
                     V_CoCo_debt_array_fullINTweakEXT, V_ext_array_fullINTweakEXT, V_Equity_array_fullINTweakEXT, Bankrupt_proportion_array_fullINTweakEXT, Converting_proportion_array_fullINTweakEXT, Healthy_proportion_array_fullINTweakEXT,
                     V_CoCo_debt_array_fullINTfullEXT, V_ext_array_fullINTfullEXT, V_Equity_array_fullINTfullEXT, Bankrupt_proportion_array_fullINTfullEXT, Converting_proportion_array_fullINTfullEXT, Healthy_proportion_array_fullINTfullEXT], f)
elif load_results:
    with open('Analysis EBA/Results_CH4_EBAdataset_varyShock_lowerR', 'rb') as f: 
        V_CoCo_debt_array_weakINT, V_ext_array_weakINT, V_Equity_array_weakINT, Bankrupt_proportion_array_weakINT, Converting_proportion_array_weakINT, Healthy_proportion_array_weakINT, V_CoCo_debt_array_fullINT, V_ext_array_fullINT, V_Equity_array_fullINT, Bankrupt_proportion_array_fullINT, Converting_proportion_array_fullINT, Healthy_proportion_array_fullINT, V_CoCo_debt_array_fullINTweakEXT, V_ext_array_fullINTweakEXT, V_Equity_array_fullINTweakEXT, Bankrupt_proportion_array_fullINTweakEXT, Converting_proportion_array_fullINTweakEXT, Healthy_proportion_array_fullINTweakEXT, V_CoCo_debt_array_fullINTfullEXT, V_ext_array_fullINTfullEXT, V_Equity_array_fullINTfullEXT, Bankrupt_proportion_array_fullINTfullEXT, Converting_proportion_array_fullINTfullEXT, Healthy_proportion_array_fullINTfullEXT = pickle.load(f)

plotting = True
save_plots = True
if plotting:
    # plt.plot(X_shock_range, V_CoCo_debt_array_weakINT.mean(axis=1))
    # plt.plot(X_shock_range, V_CoCo_debt_array_fullINT.mean(axis=1))
    # plt.plot(X_shock_range, V_CoCo_debt_array_fullINTweakEXT.mean(axis=1))
    # plt.plot(X_shock_range, V_CoCo_debt_array_fullINTfullEXT.mean(axis=1))
    # plt.legend(["Weak interbank", "Full interbank", "Full int., weak external", "Full int. and ext."])
    # plt.xlabel("Shock $X$")
    # plt.ylabel(r"$V_{\mathrm{creditors}}$")
    # plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/EBAVCreditor', bbox_inches="tight", dpi=300) if save_plots else None
    # plt.show()

    plt.plot(X_shock_range, V_ext_array_weakINT.mean(axis=1))
    plt.plot(X_shock_range, V_ext_array_fullINT.mean(axis=1))
    plt.plot(X_shock_range, V_ext_array_fullINTweakEXT.mean(axis=1))
    plt.plot(X_shock_range, V_ext_array_fullINTfullEXT.mean(axis=1))
    plt.legend(["Weak interbank", "Full interbank", "Full int., weak external", "Full int. and ext."])
    plt.xlabel("Shock $X$")
    plt.ylabel(r"$V_{\mathrm{ext.creditors}}$")
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/EBAVextCreditors_lowerR', bbox_inches="tight", dpi=300) if save_plots else None
    plt.show()

    # plt.plot(X_shock_range, V_Equity_array_weakINT.mean(axis=1))
    # plt.plot(X_shock_range, V_Equity_array_fullINT.mean(axis=1))
    # plt.plot(X_shock_range, V_Equity_array_fullINTweakEXT.mean(axis=1))
    # plt.plot(X_shock_range, V_Equity_array_fullINTfullEXT.mean(axis=1))
    # plt.legend(["Weak interbank", "Full interbank", "Full int., weak external", "Full int. and ext."])
    # plt.xlabel("Shock $X$")
    # plt.ylabel(r"$V_{\mathrm{eq.holders}}$")
    # plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/EBAVEQH', bbox_inches="tight", dpi=300) if save_plots else None
    # plt.show()

    # plt.plot(X_shock_range, Bankrupt_proportion_array_weakINT.mean(axis=1))
    # plt.plot(X_shock_range, Bankrupt_proportion_array_fullINT.mean(axis=1))
    # plt.plot(X_shock_range, Bankrupt_proportion_array_fullINTweakEXT.mean(axis=1)) 
    # plt.plot(X_shock_range, Bankrupt_proportion_array_fullINTfullEXT.mean(axis=1)) 
    # plt.legend(["Weak interbank", "Full interbank", "Full int., weak external", "Full int. and ext."])
    # plt.xlabel("Shock $X$")
    # plt.ylabel("Fraction of banks defaulting")
    # plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/EBAFD', bbox_inches="tight", dpi=300) if save_plots else None
    # plt.show()

    # plt.plot(X_shock_range, Converting_proportion_array_weakINT.mean(axis=1))
    # plt.plot(X_shock_range, Converting_proportion_array_fullINT.mean(axis=1))
    # plt.plot(X_shock_range, Converting_proportion_array_fullINTweakEXT.mean(axis=1))
    # plt.plot(X_shock_range, Converting_proportion_array_fullINTfullEXT.mean(axis=1))
    # plt.legend(["Weak interbank", "Full interbank", "Full int., weak external", "Full int. and ext."])
    # plt.xlabel("Shock $X$")
    # plt.ylabel("Fraction of banks converting")
    # plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/EBAFC', bbox_inches="tight", dpi=300) if save_plots else None
    # plt.show()

    # plt.plot(X_shock_range, Healthy_proportion_array_weakINT.mean(axis=1))
    # plt.plot(X_shock_range, Healthy_proportion_array_fullINT.mean(axis=1))
    # plt.plot(X_shock_range, Healthy_proportion_array_fullINTweakEXT.mean(axis=1))
    # plt.plot(X_shock_range, Healthy_proportion_array_fullINTfullEXT.mean(axis=1))
    # plt.legend(["Weak interbank", "Full interbank", "Full int., weak external", "Full int. and ext."])
    # plt.xlabel("Shock $X$")
    # plt.ylabel("Fraction of banks remaining healthy")
    # plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/EBAFH', bbox_inches="tight", dpi=300) if save_plots else None
    # plt.show()
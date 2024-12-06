import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from helper_functions import compute_equilibrium, import_weight_matrices_R
import itertools
import pickle

# Import generated W matrices from Csv-files generated in R
W_matrices_GV_cp = import_weight_matrices_R("Data EBA/W_matrix_GV_cp")                  #import liability matrices from csv files generated in R 

# Import other data and define model parameters 
dataset_EBA = pd.read_csv('Data EBA/dataset_EBA_cleaned.csv', index_col=0) / 1000       #scale by 1000 to prevent numerical issues


### Analysis 1: Fix shock X and vary m + gamma 
gamma_INT_range = np.linspace(0.01, 1.01, 11)
m_range = np.linspace(0.01, 2.01, 11)
n_combinations = gamma_INT_range.shape[0] * m_range.shape[0]

L_creditor_array = np.zeros((gamma_INT_range.shape[0], m_range.shape[0]))
L_equityholder_array = np.zeros((gamma_INT_range.shape[0], m_range.shape[0]))
fraction_bankrupt_array = np.zeros((gamma_INT_range.shape[0], m_range.shape[0]))
fraction_converting_array = np.zeros((gamma_INT_range.shape[0], m_range.shape[0]))
fraction_healthy_array = np.zeros((gamma_INT_range.shape[0], m_range.shape[0]))

compute_results = False
EXT_CoCo = False                 #external debt CoCo-ized or not
if compute_results: 
    k=0
    for gamma_x, m_x in itertools.product(gamma_INT_range, m_range):
        index_gamma = k // m_range.shape[0]
        index_m = k - m_range.shape[0] * index_gamma
        print(f"Iteration {k+1}/{n_combinations}, {(gamma_x, m_x)}")
        gamma_INT = gamma_x                         # fraction of interbank liabilities CoCo-ized
        gamma_EXT = gamma_x if EXT_CoCo else 0       # fraction of external liabilities CoCo-ized
        X_shock = 0.03    
        c = dataset_EBA["Total Interbank Liabilities"].to_numpy() * gamma_INT + dataset_EBA["Total External Liabilities"].to_numpy() * gamma_EXT
        correction_factor_W = (dataset_EBA["Total Interbank Liabilities"].to_numpy() * gamma_INT ) / c if sum(c) != 0 else 0

        L_creditor = 0 
        L_equityholder = 0
        fr_b = 0
        fr_c = 0
        fr_h = 0
        for w in range(len(W_matrices_GV_cp)):
            W_GV_corrected = W_matrices_GV_cp[w] * correction_factor_W

            a = dataset_EBA["Total External Assets"].to_numpy() * (1 - X_shock) - dataset_EBA["Total External Liabilities"].to_numpy() * (1 - gamma_EXT)           # note: because Ext. Assets = Ext. Liabilities, the net assets a are independent of gamma_CoCo
            a_non_shocked = dataset_EBA["Total External Assets"].to_numpy() - dataset_EBA["Total External Liabilities"].to_numpy() * (1 - gamma_EXT)
            n = W_matrices_GV_cp[w].shape[0]
            m = np.ones(n) * m_x       
            l = c / m                               # fair thresholds for existence and uniqueness of equilibrium

            B, C, H, s = compute_equilibrium(l, c, m, W_GV_corrected, a)
            if (B, C, H, s) == (0, 0 ,0 ,0):
                print("Infeasible")
                print((gamma_x, m_x))
                continue
            _, _, _, s_non_shocked = compute_equilibrium(l, c, m, W_GV_corrected, a_non_shocked)
            L_creditor += np.sum(c * np.array([1 if j in B else 0 for j in range(n)]) + (c - m*s) * np.array([1 if j in C else 0 for j in range(n)])) / sum(c) if sum(c) != 0 else 0
            L_equityholder += np.sum(np.maximum(s_non_shocked,0) - np.maximum(s, 0)) / sum(np.maximum(s_non_shocked, 0))
            fr_b += len(B)/n
            fr_c += len(C)/n
            fr_h += 1 - fr_b - fr_c

        L_creditor_array[index_gamma, index_m] = L_creditor / len(W_matrices_GV_cp)
        L_equityholder_array[index_gamma, index_m] = L_equityholder / len(W_matrices_GV_cp)
        fraction_bankrupt_array[index_gamma, index_m]  = fr_b / len(W_matrices_GV_cp)
        fraction_converting_array[index_gamma, index_m] = fr_c / len(W_matrices_GV_cp)
        fraction_healthy_array[index_gamma, index_m] = fr_h / len(W_matrices_GV_cp)
        k +=1
    
save_results = False
load_results = True
if save_results:
    with open('Results_CH4_EBAdataset_3d_int', 'wb') as f:
        pickle.dump([L_creditor_array , L_equityholder_array, fraction_bankrupt_array, fraction_converting_array, fraction_healthy_array], f)
if load_results:
    with open('Results_CH4_EBAdataset_3d_intext', 'rb') as f:
        L_creditor_array , L_equityholder_array, fraction_bankrupt_array, fraction_converting_array, fraction_healthy_array = pickle.load(f)

plotting = True        
save = True
if plotting: 
    # Plot settings
    plt.rc('axes', labelsize=18)       
    plt.rc('xtick', labelsize=12)       
    plt.rc('ytick', labelsize=12)      
    plt.rc('legend', fontsize='small')

    X, Y = np.meshgrid(gamma_INT_range, m_range, indexing='ij')

    Z1 = L_creditor_array
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z1, cmap='viridis', edgecolor='none')
    plt.title(r"Relative losses for creditors in a 3% shock scenario")
    ax.set_xlabel(r"$\gamma_{\mathrm{interbank}} = \gamma_{\mathrm{external}}$")
    ax.set_ylabel(r"$m$")
    ax.set_zlabel(r"$L_{\mathrm{creditors}}$")
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/EBA_3d_LCR', bbox_inches="tight", dpi=300) if save else None
    plt.show()

    Z2 = L_equityholder_array
    print(Z2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z2, cmap='viridis', edgecolor='none')
    plt.title(r"Relative losses for original shareholders in a 3% shock scenario")
    ax.set_xlabel(r"$\gamma_{\mathrm{interbank}} = \gamma_{\mathrm{external}}$")
    ax.set_ylabel(r"$m$")
    ax.set_zlabel(r"$L_{\mathrm{eq.holders}}$")
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/EBA_3d_LEQ', bbox_inches="tight", dpi=300) if save else None
    plt.show()

    Z3 = fraction_bankrupt_array
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z3, cmap='viridis', edgecolor='none')
    plt.title(r"Fraction of banks defaulting in a 3% shock scenario")
    ax.set_xlabel(r"$\gamma_{\mathrm{interbank}} = \gamma_{\mathrm{external}}$")
    ax.set_ylabel(r"$m$")
    ax.set_zlabel(r"$f_{D}$")
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/EBA_3d_fd', bbox_inches="tight", dpi=300) if save else None
    plt.show()

    Z4 = fraction_converting_array
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z4, cmap='viridis', edgecolor='none')
    plt.title(r"Fraction of banks converting in a 3% shock scenario")
    ax.set_xlabel(r"$\gamma_{\mathrm{interbank}} = \gamma_{\mathrm{external}}$")
    ax.set_ylabel(r"$m$")
    ax.set_zlabel(r"$f_{C}$")
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/EBA_3d_fc', bbox_inches="tight", dpi=300) if save else None
    plt.show()

### Analysis 2: Fixed m + scenarios gamma and vary shock X 

# Scenarios
weakInt_CoCo_ization = {"gamma_internal" : 0.05, "gamma_external" : 0.0}
fullInt_CoCo_ization = {"gamma_internal" : 0.25, "gamma_external" : 0.0}
fullIntweakExt_CoCo_ization = {"gamma_internal" : 0.75, "gamma_external" : 0.00}
fullIntfullExt_CoCo_ization = {"gamma_internal" : 1.0, "gamma_external" : 0.0}
n_values_X = 100
X_shock_range = np.linspace(0, 0.25, n_values_X)

n = W_matrices_GV_cp[0].shape[0]
m = np.ones(n) * 50

def compute_metrics_scenario(n, m, W_matrices, scenario_gamma, scenarios_X):
    Loss_CoCo_debt_array = np.zeros((scenarios_X.shape[0], len(W_matrices_GV_cp)))
    Loss_Equity_array= np.zeros((scenarios_X.shape[0], len(W_matrices_GV_cp)))
    Bankrupt_proportion_array = np.zeros((scenarios_X.shape[0], len(W_matrices_GV_cp)))
    Converting_proportion_array = np.zeros((scenarios_X.shape[0], len(W_matrices_GV_cp)))
    Healthy_proportion_array = np.zeros((scenarios_X.shape[0], len(W_matrices_GV_cp)))

    c = dataset_EBA["Total Interbank Liabilities"].to_numpy() * scenario_gamma["gamma_internal"] + dataset_EBA["Total External Liabilities"].to_numpy() * scenario_gamma["gamma_external"]
    l = c / m   
    correction_factor_W = (dataset_EBA["Total Interbank Liabilities"].to_numpy() * scenario_gamma["gamma_internal"] ) / c if sum(c) != 0 else 0
    a_full = dataset_EBA["Total External Assets"].to_numpy() - dataset_EBA["Total External Liabilities"].to_numpy() * (1 - scenario_gamma["gamma_external"])     
    _, _, _, s_initial = compute_equilibrium(l, c, m, W_matrices_GV_cp[0] * correction_factor_W, a_full)
    
    for w in range(len(W_matrices)):
        print(f"Iteration {w}/{len(W_matrices)}", end='\r')
        W_matrix_corrected = W_matrices[w] * correction_factor_W
        for k, X in enumerate(scenarios_X): 
            a = dataset_EBA["Total External Assets"].to_numpy() * (1 - X) - dataset_EBA["Total External Liabilities"].to_numpy() * (1 - scenario_gamma["gamma_external"])           # note: because Ext. Assets = Ext. Liabilities, the net assets a are independent of gamma_CoCo
            B, C, H, s = compute_equilibrium(l, c, m, W_matrix_corrected, a)
            if (B, C, H, s) == (0, 0 ,0 ,0):
                print("Infeasible")
                continue
            
            Loss_CoCo_debt_array[k, w] = np.sum(c * np.array([1 if j in B else 0 for j in range(n)]) + (c - m*s) * np.array([1 if j in C else 0 for j in range(n)])) #/ sum(c) if sum(c) != 0 else 0
            Loss_Equity_array[k, w] = np.sum(np.maximum(s, 0)) #np.sum(np.maximum(s_initial, 0)) - np.sum(np.maximum(s, 0)) 
            Bankrupt_proportion_array[k, w]  = len(B)/n
            Converting_proportion_array[k, w] = len(C)/n
            Healthy_proportion_array[k, w] = 1 - Bankrupt_proportion_array[k, w] - Converting_proportion_array[k, w]
    
    return Loss_CoCo_debt_array, Loss_Equity_array, Bankrupt_proportion_array, Converting_proportion_array, Healthy_proportion_array

compute_results = False
if compute_results:
    Loss_CoCo_debt_array_weakINT, Loss_Equity_array_weakINT, Bankrupt_proportion_array_weakINT, Converting_proportion_array_weakINT, Healthy_proportion_array_weakINT = compute_metrics_scenario(n, m, W_matrices_GV_cp, weakInt_CoCo_ization, X_shock_range)
    Loss_CoCo_debt_array_fullINT, Loss_Equity_array_fullINT, Bankrupt_proportion_array_fullINT, Converting_proportion_array_fullINT, Healthy_proportion_array_fullINT = compute_metrics_scenario(n, m, W_matrices_GV_cp, fullInt_CoCo_ization, X_shock_range)
    Loss_CoCo_debt_array_fullINTweakEXT, Loss_Equity_array_fullINTweakEXT, Bankrupt_proportion_array_fullINTweakEXT, Converting_proportion_array_fullINTweakEXT, Healthy_proportion_array_fullINTweakEXT = compute_metrics_scenario(n, m, W_matrices_GV_cp, fullIntweakExt_CoCo_ization, X_shock_range)
    Loss_CoCo_debt_array_fullINTfullEXT, Loss_Equity_array_fullINTfullEXT, Bankrupt_proportion_array_fullINTfullEXT, Converting_proportion_array_fullINTfullEXT, Healthy_proportion_array_fullINTfullEXT= compute_metrics_scenario(n, m, W_matrices_GV_cp, fullIntfullExt_CoCo_ization, X_shock_range)

save_results = False
load_results = False
if save_results:
    with open('Results_CH4_EBAdataset_varyShock', 'wb') as f:
        pickle.dump([Loss_CoCo_debt_array_weakINT, Loss_Equity_array_weakINT, Bankrupt_proportion_array_weakINT, Converting_proportion_array_weakINT, Healthy_proportion_array_weakINT,
                     Loss_CoCo_debt_array_fullINT, Loss_Equity_array_fullINT, Bankrupt_proportion_array_fullINT, Converting_proportion_array_fullINT, Healthy_proportion_array_fullINT,
                     Loss_CoCo_debt_array_fullINTweakEXT, Loss_Equity_array_fullINTweakEXT, Bankrupt_proportion_array_fullINTweakEXT, Converting_proportion_array_fullINTweakEXT, Healthy_proportion_array_fullINTweakEXT,
                     Loss_CoCo_debt_array_fullINTfullEXT, Loss_Equity_array_fullINTfullEXT, Bankrupt_proportion_array_fullINTfullEXT, Converting_proportion_array_fullINTfullEXT, Healthy_proportion_array_fullINTfullEXT], f)
if load_results:
    with open('Results_CH4_EBAdataset_varyShock', 'rb') as f: Loss_CoCo_debt_array_weakINT, Loss_Equity_array_weakINT, Bankrupt_proportion_array_weakINT, Converting_proportion_array_weakINT, Healthy_proportion_array_weakINT, Loss_CoCo_debt_array_fullINT, Loss_Equity_array_fullINT, Bankrupt_proportion_array_fullINT, Converting_proportion_array_fullINT, Healthy_proportion_array_fullINT, Loss_CoCo_debt_array_fullINTweakEXT, Loss_Equity_array_fullINTweakEXT, Bankrupt_proportion_array_fullINTweakEXT, Converting_proportion_array_fullINTweakEXT, Healthy_proportion_array_fullINTweakEXT, Loss_CoCo_debt_array_fullINTfullEXT, Loss_Equity_array_fullINTfullEXT, Bankrupt_proportion_array_fullINTfullEXT, Converting_proportion_array_fullINTfullEXT, Healthy_proportion_array_fullINTfullEXT = pickle.load(f)

plotting = False
save_plots = True
if plotting:
    plt.plot(X_shock_range, Loss_CoCo_debt_array_weakINT.mean(axis=1))
    plt.plot(X_shock_range, Loss_CoCo_debt_array_fullINT.mean(axis=1))
    plt.plot(X_shock_range, Loss_CoCo_debt_array_fullINTweakEXT.mean(axis=1))
    plt.plot(X_shock_range, Loss_CoCo_debt_array_fullINTfullEXT.mean(axis=1))
    plt.legend(["Weak interbank", "Full interbank", "Full int., weak external", "Full int. and ext."])
    plt.xlabel("Shock $X$")
    plt.ylabel("Losses Creditors")
    plt.title(r"Comparison of average loss for creditors in the 4 scenarios")
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/EBAlossCreditor', bbox_inches="tight", dpi=300) if save_plots else None
    plt.show()

    plt.plot(X_shock_range, Loss_Equity_array_weakINT.mean(axis=1))
    plt.plot(X_shock_range, Loss_Equity_array_fullINT.mean(axis=1))
    plt.plot(X_shock_range, Loss_Equity_array_fullINTweakEXT.mean(axis=1))
    plt.plot(X_shock_range, Loss_Equity_array_fullINTfullEXT.mean(axis=1))
    plt.legend(["Weak interbank", "Full interbank", "Full int., weak external", "Full int. and ext."])
    plt.xlabel("Shock $X$")
    plt.ylabel("Losses Original Equityholders")
    plt.title(r"Comparison of average loss for orignal equityholders in the 4 scenarios")
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/EBAlossEQH', bbox_inches="tight", dpi=300) if save_plots else None
    plt.show()

    plt.plot(X_shock_range, Bankrupt_proportion_array_weakINT.mean(axis=1))
    plt.plot(X_shock_range, Bankrupt_proportion_array_fullINT.mean(axis=1))
    plt.plot(X_shock_range, Bankrupt_proportion_array_fullINTweakEXT.mean(axis=1)) 
    plt.plot(X_shock_range, Bankrupt_proportion_array_fullINTfullEXT.mean(axis=1)) 
    plt.legend(["Weak interbank", "Full interbank", "Full int., weak external", "Full int. and ext."])
    plt.xlabel("Shock $X$")
    plt.ylabel("Fraction of banks defaulting")
    plt.title(r"Comparison of average fraction of banks defaulting in the 4 scenarios")
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/EBAFD', bbox_inches="tight", dpi=300) if save_plots else None
    plt.show()

    plt.plot(X_shock_range, Converting_proportion_array_weakINT.mean(axis=1))
    plt.plot(X_shock_range, Converting_proportion_array_fullINT.mean(axis=1))
    plt.plot(X_shock_range, Converting_proportion_array_fullINTweakEXT.mean(axis=1))
    plt.plot(X_shock_range, Converting_proportion_array_fullINTfullEXT.mean(axis=1))
    plt.legend(["Weak interbank", "Full interbank", "Full int., weak external", "Full int. and ext."])
    plt.xlabel("Shock $X$")
    plt.ylabel("Fraction of banks converting")
    plt.title(r"Comparison of average fraction of banks converting in the 4 scenarios")
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/EBAFC', bbox_inches="tight", dpi=300) if save_plots else None
    plt.show()

    plt.plot(X_shock_range, Healthy_proportion_array_weakINT.mean(axis=1))
    plt.plot(X_shock_range, Healthy_proportion_array_fullINT.mean(axis=1))
    plt.plot(X_shock_range, Healthy_proportion_array_fullINTweakEXT.mean(axis=1))
    plt.plot(X_shock_range, Healthy_proportion_array_fullINTfullEXT.mean(axis=1))
    plt.legend(["Weak interbank", "Full interbank", "Full int., weak external", "Full int. and ext."])
    plt.xlabel("Shock $X$")
    plt.ylabel("Fraction of banks remaining healthy")
    plt.title(r"Comparison of average fraction of banks remaining healthy in the 4 scenarios")
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/EBAFH', bbox_inches="tight", dpi=300) if save_plots else None
    plt.show()
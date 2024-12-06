import numpy as np
import matplotlib.pyplot as plt
from helper_functions import compute_sensitivity_contract_parameters, plot_results_sensitivity, compute_sensitivity_contract_parameters_multitranch
import pickle 

########################### Networks ####################################

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


########################### Analysis of systemic risk ####################################

### CONTAGION WITH SINGLE TRANCH ###

# Set up the problem 
n = 4
core_banks = [0]
W_ring = generate_network(n, type="ring", fixed=True, weights_fixed = 0.75)
W_complete = generate_network(n, type="complete", fixed=True, weights_fixed = 0.75)
W_cp = generate_network(n, type="core periphery", fixed=True, weights_fixed = 0.75, core_nodes = core_banks)
W_variants = [W_ring, W_complete, W_cp]

# Settings for initial asset values
b_rc = np.ones(n) * 2               
b_cp = np.ones(n) * 2           
b_cp[core_banks] = 5            

# Settings for shocks
np.random.seed(0)   
nbSimulations = 100
shocked_banks = [0]     
shocked_banks_periphery = [1]
beta = 2
atilde0 = 40
d0 = 20
X_shock = atilde0*(1 - np.random.beta(1, beta, size=(nbSimulations, 1))) - d0

## Impact of m and c on PD, PC, PH, Lc, Lcc, Leq
m_range = np.arange(0.01, 5.01, 0.1)
c_range = np.arange(0.01, 50.01, 1)            
effect = "m"

new_computation = False
if new_computation:
    P_total_ring, P_total_complete, P_total_cp, P_total_cp_p, L_creditors_ring, L_creditors_complete, L_creditors_cp, L_creditors_cp_p, \
            L_creditors_contagion_ring, L_creditors_contagion_complete, L_creditors_contagion_cp, L_creditors_contagion_cp_p, L_equityholders_ring, \
                L_equityholders_complete, L_equityholders_cp, L_equityholders_cp_p, L_equityholders_contagion_ring, L_equityholders_contagion_complete, \
                    L_equityholders_contagion_cp, L_equityholders_contagion_cp_p  = compute_sensitivity_contract_parameters(n, W_variants, core_banks, shocked_banks, shocked_banks_periphery, b_rc, b_cp, X_shock, m_range, c_range, effect)

save_results = False
load_results = False
if save_results:
    with open('Results_CH4_singletranch_c', 'wb') as f:
        pickle.dump([P_total_ring, P_total_complete, P_total_cp, P_total_cp_p, L_creditors_ring, L_creditors_complete, L_creditors_cp, L_creditors_cp_p, \
            L_creditors_contagion_ring, L_creditors_contagion_complete, L_creditors_contagion_cp, L_creditors_contagion_cp_p, L_equityholders_ring, \
                L_equityholders_complete, L_equityholders_cp, L_equityholders_cp_p, L_equityholders_contagion_ring, L_equityholders_contagion_complete, \
                    L_equityholders_contagion_cp, L_equityholders_contagion_cp_p], f)

if load_results:
    with open('Results_CH4_singletranch_m', 'rb') as f:
        P_total_ring, P_total_complete, P_total_cp, P_total_cp_p, L_creditors_ring, L_creditors_complete, L_creditors_cp, L_creditors_cp_p, \
            L_creditors_contagion_ring, L_creditors_contagion_complete, L_creditors_contagion_cp, L_creditors_contagion_cp_p, L_equityholders_ring, \
                L_equityholders_complete, L_equityholders_cp, L_equityholders_cp_p, L_equityholders_contagion_ring, L_equityholders_contagion_complete, \
                    L_equityholders_contagion_cp, L_equityholders_contagion_cp_p = pickle.load(f)

plot_results = False
single = True
if plot_results:
    plot_results_sensitivity(m_range, c_range, effect, single, P_total_ring, P_total_complete, P_total_cp, P_total_cp_p, L_creditors_ring, L_creditors_complete, L_creditors_cp, L_creditors_cp_p, \
            L_creditors_contagion_ring, L_creditors_contagion_complete, L_creditors_contagion_cp, L_creditors_contagion_cp_p, L_equityholders_ring, \
                L_equityholders_complete, L_equityholders_cp, L_equityholders_cp_p, L_equityholders_contagion_ring, L_equityholders_contagion_complete, \
                    L_equityholders_contagion_cp, L_equityholders_contagion_cp_p, save=False)


### CONTAGION WITH TWO TRANCES ###

### Two tranches
# Set up the problem 
n = 4
core_banks = [0]
W_ring = generate_network(n, type="ring", fixed=True, weights_fixed = 0.75)
W_complete = generate_network(n, type="complete", fixed=True, weights_fixed = 0.75)
W_cp = generate_network(n, type="core periphery", fixed=True, weights_fixed = 0.75, core_nodes = core_banks)
W_variants = [W_ring, W_complete, W_cp]

# Tranch parameters
zeta = 0.5              
xi = 0.3                   # half of CoCo converts at 30% share gain, half converts at 70% share gain

# Settings for initial asset values
b_rc = np.ones(n) * 2               
b_cp = np.ones(n) * 2           
b_cp[core_banks] = 5            

# Settings for shocks
np.random.seed(0)   
nbSimulations = 1_000
shocked_banks = [0]     
shocked_banks_periphery = [1]
beta = 2
atilde0 = 40
d0 = 20
X_shock = atilde0*(1 - np.random.beta(1, beta, size=(nbSimulations, 1))) - d0

## Impact of m and c on PD, PC, PH, Lc, Lcc, Leq
m_range = np.arange(0.01, 5.01, 0.1)
c_range = np.arange(0.01, 50.01, 1)            
effect = "c"

new_computation = False
if new_computation:
    P_total_ring, P_total_complete, P_total_cp, P_total_cp_p, L_creditors_ring, L_creditors_complete, L_creditors_cp, L_creditors_cp_p, \
            L_creditors_contagion_ring, L_creditors_contagion_complete, L_creditors_contagion_cp, L_creditors_contagion_cp_p, L_equityholders_ring, \
                L_equityholders_complete, L_equityholders_cp, L_equityholders_cp_p, L_equityholders_contagion_ring, L_equityholders_contagion_complete, \
                    L_equityholders_contagion_cp, L_equityholders_contagion_cp_p  = compute_sensitivity_contract_parameters_multitranch(n, W_variants, zeta, xi, core_banks, shocked_banks, shocked_banks_periphery, b_rc, b_cp, X_shock, m_range, c_range, effect)

save_results = False
load_results = False
if save_results:
    with open('Results_CH4_multitranch_c', 'wb') as f:
        pickle.dump([P_total_ring, P_total_complete, P_total_cp, P_total_cp_p, L_creditors_ring, L_creditors_complete, L_creditors_cp, L_creditors_cp_p, \
            L_creditors_contagion_ring, L_creditors_contagion_complete, L_creditors_contagion_cp, L_creditors_contagion_cp_p, L_equityholders_ring, \
                L_equityholders_complete, L_equityholders_cp, L_equityholders_cp_p, L_equityholders_contagion_ring, L_equityholders_contagion_complete, \
                    L_equityholders_contagion_cp, L_equityholders_contagion_cp_p], f)

if load_results:
    with open('Results_CH4_multitranch_c', 'rb') as f:
        P_total_ring, P_total_complete, P_total_cp, P_total_cp_p, L_creditors_ring, L_creditors_complete, L_creditors_cp, L_creditors_cp_p, \
            L_creditors_contagion_ring, L_creditors_contagion_complete, L_creditors_contagion_cp, L_creditors_contagion_cp_p, L_equityholders_ring, \
                L_equityholders_complete, L_equityholders_cp, L_equityholders_cp_p, L_equityholders_contagion_ring, L_equityholders_contagion_complete, \
                    L_equityholders_contagion_cp, L_equityholders_contagion_cp_p = pickle.load(f)

plot_results = False
single = False
if plot_results:
    plot_results_sensitivity(m_range, c_range, effect, single, P_total_ring, P_total_complete, P_total_cp, P_total_cp_p, L_creditors_ring, L_creditors_complete, L_creditors_cp, L_creditors_cp_p, \
            L_creditors_contagion_ring, L_creditors_contagion_complete, L_creditors_contagion_cp, L_creditors_contagion_cp_p, L_equityholders_ring, \
                L_equityholders_complete, L_equityholders_cp, L_equityholders_cp_p, L_equityholders_contagion_ring, L_equityholders_contagion_complete, \
                    L_equityholders_contagion_cp, L_equityholders_contagion_cp_p)


### COMPARE SINGLE TRANCH TO TWO TRANCH ###

compare_results= False
if compare_results: 
    with open('Results_CH4_singletranch_c', 'rb') as f:
        sP_total_ring, sP_total_complete, sP_total_cp, sP_total_cp_p, sL_creditors_ring, sL_creditors_complete, sL_creditors_cp, sL_creditors_cp_p, \
            sL_creditors_contagion_ring, sL_creditors_contagion_complete, sL_creditors_contagion_cp, sL_creditors_contagion_cp_p, sL_equityholders_ring, \
                sL_equityholders_complete, sL_equityholders_cp, sL_equityholders_cp_p, sL_equityholders_contagion_ring, sL_equityholders_contagion_complete, \
                        sL_equityholders_contagion_cp, sL_equityholders_contagion_cp_p = pickle.load(f)

    with open('Results_CH4_multitranch_c', 'rb') as f:
        mP_total_ring, mP_total_complete, mP_total_cp, mP_total_cp_p, mL_creditors_ring, mL_creditors_complete, mL_creditors_cp, mL_creditors_cp_p, \
            mL_creditors_contagion_ring, mL_creditors_contagion_complete, mL_creditors_contagion_cp, mL_creditors_contagion_cp_p, mL_equityholders_ring, \
                mL_equityholders_complete, mL_equityholders_cp, mL_equityholders_cp_p, mL_equityholders_contagion_ring, mL_equityholders_contagion_complete, \
                        mL_equityholders_contagion_cp, mL_equityholders_contagion_cp_p = pickle.load(f)

    diffL_creditor_ring = (mL_creditors_ring - sL_creditors_ring)
    diffL_creditor_complete = (mL_creditors_complete - sL_creditors_complete ) 
    diffL_creditor_cp = (mL_creditors_cp - sL_creditors_cp)
    diffL_creditor_cp_p = (mL_creditors_cp_p - sL_creditors_cp_p) 

    diffL_creditor_contagion_ring = (mL_creditors_contagion_ring - sL_creditors_contagion_ring) 
    diffL_creditor_contagion_complete = (mL_creditors_contagion_complete - sL_creditors_contagion_complete )
    diffL_creditor_contagion_cp = (mL_creditors_contagion_cp - sL_creditors_contagion_cp) 
    diffL_creditor_contagion_cp_p = (mL_creditors_contagion_cp_p - sL_creditors_contagion_cp_p) 

    diffL_equityholders_ring = (mL_equityholders_ring - sL_equityholders_ring) 
    diffL_equityholders_complete = (mL_equityholders_complete - sL_equityholders_complete ) 
    diffL_equityholders_cp = (mL_equityholders_cp - sL_equityholders_cp) 
    diffL_equityholders_cp_p = (mL_equityholders_cp_p - sL_equityholders_cp_p) 

    diffL_equityholders_contagion_ring = (mL_equityholders_contagion_ring - sL_equityholders_contagion_ring) 
    diffL_equityholders_contagion_complete = (mL_equityholders_contagion_complete - sL_equityholders_contagion_complete ) 
    diffL_equityholders_contagion_cp = (mL_equityholders_contagion_cp - sL_equityholders_contagion_cp) 
    diffL_equityholders_contagion_cp_p = (mL_equityholders_contagion_cp_p - sL_equityholders_contagion_cp_p) 

    diffLcr_shocked_ring = diffL_creditor_ring - diffL_creditor_contagion_ring 
    diffLcr_shocked_complete = diffL_creditor_complete - diffL_creditor_contagion_complete
    diffLcr_shocked_cp = diffL_creditor_cp - diffL_creditor_contagion_cp
    diffLcr_shocked_cp_p = diffL_creditor_cp_p - diffL_creditor_contagion_cp_p

    diffLeh_shocked_ring = diffL_equityholders_ring - diffL_equityholders_contagion_ring 
    diffLeh_shocked_complete = diffL_equityholders_complete - diffL_equityholders_contagion_complete
    diffLeh_shocked_cp = diffL_equityholders_cp - diffL_equityholders_contagion_cp
    diffLeh_shocked_cp_p = diffL_equityholders_cp_p - diffL_equityholders_contagion_cp_p

    save = False
    # Plot system level metrics
    plt.plot(m_range if effect == "m" else c_range, diffL_creditor_ring, linestyle=':')
    plt.plot(m_range if effect == "m" else c_range, diffL_creditor_complete, linestyle='-.')
    plt.plot(m_range if effect == "m" else c_range, diffL_creditor_cp, linestyle='--')
    plt.plot(m_range if effect == "m" else c_range, diffL_creditor_cp_p, linestyle='-')
    plt.legend(["Ring network", "Complete network", "Star network (core node)", "Star network (periphery node)"], fontsize='small')
    plt.title(r"Comparison of $L_{\mathrm{creditors}}$ for single- and multi-tranch CoCos")
    plt.xlabel("Conversion rate $m$" if effect == "m" else "Issued convertible debt $c$")
    plt.ylabel(r'$\Delta L_{\mathrm{creditors, total}}$')
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/losscreditors_diff', bbox_inches="tight", dpi=300) if save else None
    plt.show()

    plt.plot(m_range if effect == "m" else c_range, diffL_creditor_contagion_ring, linestyle=':')
    plt.plot(m_range if effect == "m" else c_range, diffL_creditor_contagion_complete, linestyle='-.')
    plt.plot(m_range if effect == "m" else c_range, diffL_creditor_contagion_cp, linestyle='--')
    plt.plot(m_range if effect == "m" else c_range, diffL_creditor_contagion_cp_p, linestyle='-')
    plt.legend(["Ring network", "Complete network", "Star network (core node)", "Star network (periphery node)"], fontsize='small')
    plt.title(r"Comparison of $L_{\mathrm{creditors, contagion}}$ for single- and multi-tranch CoCos")
    plt.xlabel("Conversion rate $m$" if effect == "m" else "Issued convertible debt $c$")
    plt.ylabel(r'$\Delta L_{\mathrm{creditors, contagion}}$')
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/losscreditorscontagion_diff', bbox_inches="tight", dpi=300) if save else None
    plt.show()

    plt.plot(m_range if effect == "m" else c_range, diffLcr_shocked_ring, linestyle=':')
    plt.plot(m_range if effect == "m" else c_range, diffLcr_shocked_complete, linestyle='-.')
    plt.plot(m_range if effect == "m" else c_range, diffLcr_shocked_cp, linestyle='--')
    plt.plot(m_range if effect == "m" else c_range, diffLcr_shocked_cp_p, linestyle='-')
    plt.legend(["Ring network", "Complete network", "Star network (core node)", "Star network (periphery node)"], fontsize='small') if save else None
    plt.title(r"Comparison of $L_{\mathrm{creditors, direct}}$ for single- and multi-tranch CoCos")
    plt.xlabel("Conversion rate $m$" if effect == "m" else "Issued convertible debt $c$")
    plt.ylabel(r'$\Delta L_{\mathrm{creditors, direct}}$')
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/losscreditorsdirect_diff', bbox_inches="tight", dpi=300) 
    plt.show()

    plt.plot(m_range if effect == "m" else c_range, diffL_equityholders_ring, linestyle=':')
    plt.plot(m_range if effect == "m" else c_range, diffL_equityholders_complete, linestyle='-.')
    plt.plot(m_range if effect == "m" else c_range, diffL_equityholders_cp, linestyle='--')
    plt.plot(m_range if effect == "m" else c_range, diffL_equityholders_cp_p, linestyle='-')
    plt.legend(["Ring network", "Complete network", "Star network (core node)", "Star network (periphery node)"], fontsize='small') if save else None
    plt.title(r"Comparison of $L_{\mathrm{eq.holders}}$ for single- and multi-tranch CoCos")
    plt.xlabel("Conversion rate $m$" if effect == "m" else "Issued convertible debt $c$")
    plt.ylabel(r'$\Delta L_{\mathrm{eq.holders}}$')
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/lossequityholders_diff', bbox_inches="tight", dpi=300) if save else None
    plt.show()

    plt.plot(m_range if effect == "m" else c_range, diffL_equityholders_contagion_ring, linestyle=':')
    plt.plot(m_range if effect == "m" else c_range, diffL_equityholders_contagion_complete, linestyle='-.')
    plt.plot(m_range if effect == "m" else c_range, diffL_equityholders_contagion_cp, linestyle='--')
    plt.plot(m_range if effect == "m" else c_range, diffL_equityholders_contagion_cp_p, linestyle='-')
    plt.legend(["Ring network", "Complete network", "Star network (core node)", "Star network (periphery node)"], fontsize='small')
    plt.title(r"Comparison of $L_{\mathrm{eq.holders, contagion}}$ for single- and multi-tranch CoCos")
    plt.xlabel("Conversion rate $m$" if effect == "m" else "Issued convertible debt $c$")
    plt.ylabel(r'$\Delta L_{\mathrm{eq.holders, contagion}}$')
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/lossequityholderscontagion_diff', bbox_inches="tight", dpi=300) if save else None
    plt.show()

    plt.plot(m_range if effect == "m" else c_range, diffLeh_shocked_ring, linestyle=':')
    plt.plot(m_range if effect == "m" else c_range, diffLeh_shocked_complete, linestyle='-.')
    plt.plot(m_range if effect == "m" else c_range, diffLeh_shocked_cp, linestyle='--')
    plt.plot(m_range if effect == "m" else c_range, diffLeh_shocked_cp_p, linestyle='-')
    plt.legend(["Ring network", "Complete network", "Star network (core node)", "Star network (periphery node)"], fontsize='small')
    plt.title(r"Comparison of $L_{\mathrm{eq.holders, direct}}$ for single- and multi-tranch CoCos")
    plt.xlabel("Conversion rate $m$" if effect == "m" else "Issued convertible debt $c$")
    plt.ylabel(r'$\Delta L_{\mathrm{eq.holders, direct}}$')
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/lossequityholdersdirect_diff', bbox_inches="tight", dpi=300) if save else None
    plt.show()
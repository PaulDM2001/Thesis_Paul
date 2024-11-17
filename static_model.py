import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from helper_functions import compute_equilibrium, generate_problem_instance, generate_2d_figure, generate_3d_figure, compute_systemic_risk_metrics, simulate_shocks_correlated
import pickle 

## Plot example 2D in thesis text 
# Fair case
c = [6, 6]
m = [3/4, 3/4]
l = np.array(c) / np.array(m) 
w = [0.5, 0.5]
limit_up = 20
limit_down = -20

# generate_2d_figure(c, l, m, w, limit_up, limit_down, save=False)

# Sub-fair case
l_sub = l - 3 
# generate_2d_figure(c, l_sub, m, w, limit_up, limit_down, save=False)

# Super-fair case
l_sup = l + 3 
# generate_2d_figure(c, l_sup, m, w, limit_up, limit_down, save=False)

## Plot example 3D in thesis text
# Fair case
c = np.array([6, 6, 6])
m = np.array([3/4, 3/4, 3/4])
l = np.array(c) / np.array(m) 
W = np.array([[0, 1/2, 1/2], [1/2, 0, 1/2], [1/2, 1/2, 0]])
n_samples = 10_000

limit_up = 30
limit_down = -30

# All sets
# generate_3d_figure(c, l, m, W, limit_up, limit_down, n_samples, save=False, save_name='3dplot_total', seed=True)

# Emphasise CCC
# generate_3d_figure(c, l, m, W, limit_up, limit_down, n_samples, specific_set='CCC', save=True, save_name='3dplot_CCC', seed=True)

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

# Set up the problem 
n = 4
core_banks = [0]
W_ring = generate_network(n, type="ring", fixed=True, weights_fixed = 0.75)
W_complete = generate_network(n, type="complete", fixed=True, weights_fixed = 0.75)
W_cp = generate_network(n, type="core periphery", fixed=True, weights_fixed = 0.75, core_nodes = core_banks)

# Settings for initial asset values
b = np.ones(n) * 0.1                #2
b_cp = np.ones(n) * 0.1             #2
b_cp[core_banks] = 0.5              #5

# Settings for shocks
np.random.seed(0)   
nbSimulations = 1_00
shocked_banks = [0]     
shocked_banks_periphery = [1]
beta = 2
z_1 = 40
z_2 = 20
shock = z_1*(1 - np.random.beta(1, beta, size=(nbSimulations, 1))) - z_2

## Impact of m and c on PD, PC, PH, Lc, Lcc, Leq
m_range = np.arange(0.01, 5.01, 0.1)
c_range = np.arange(0.01, 50.01, 1)            
effect = "m"

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

CCC_t_ring = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
CCC_t_complete = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
CCC_t_cp = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
CCC_t_cp_p = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))

new_computation = False
if new_computation:
    for k, x in enumerate(m_range if effect == "m" else c_range):
        print(f"Iteration {k}/{m_range.shape[0]}", end='\r')
       
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
        a_rc = np.tile(l_rc + c_rc - np.matmul(W_ring, c_rc) + b, (nbSimulations, 1))
        a_cp = np.tile(l_cp + c_cp - np.matmul(W_cp, c_cp) + b_cp, (nbSimulations, 1))
        a_cp_p = np.tile(l_cp + c_cp - np.matmul(W_cp, c_cp) + b_cp, (nbSimulations, 1))

        a_rc[:, shocked_banks] = np.minimum(shock, a_rc[:, shocked_banks])
        a_cp[:, shocked_banks] = np.minimum(shock * (n-1), a_cp[:, shocked_banks])
        a_cp_p[:, shocked_banks_periphery] = np.minimum(shock, a_cp_p[:, shocked_banks_periphery])

        # Compute initial prices
        s0_rc = l_rc + b
        s0_cp = l_cp + b_cp

        # Compute systemic risk metrics
        Prb_array_ring, CCC_ring, L_creditors_array_ring, L_creditors_contagion_array_ring, L_equityholders_array_ring = compute_systemic_risk_metrics(l_rc, c_rc, m, W_ring, s0_rc, shocked_banks, a_rc)
        Prb_array_cplt, CCC_cplt, L_creditors_array_cplt, L_creditors_contagion_array_cplt, L_equityholders_array_cplt = compute_systemic_risk_metrics(l_rc, c_rc, m, W_complete, s0_rc, shocked_banks, a_rc)
        Prb_array_cp, CCC_cp, L_creditors_array_cp, L_creditors_contagion_array_cp, L_equityholders_array_cp = compute_systemic_risk_metrics(l_cp, c_cp, m, W_cp, s0_cp, shocked_banks, a_cp)
        Prb_array_cp_p, CCC_cp_p, L_creditors_array_cp_p, L_creditors_contagion_array_cp_p, L_equityholders_array_cp_p = compute_systemic_risk_metrics(l_cp, c_cp, m, W_cp, s0_cp, shocked_banks_periphery, a_cp_p)

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

        CCC_t_ring[k] = CCC_ring
        CCC_t_complete[k] = CCC_cplt
        CCC_t_cp[k] = CCC_cp
        CCC_t_cp_p[k] = CCC_cp_p

save_results = False if new_computation else False
if save_results:
    with open('Results_CH3_baseline_c_new', 'wb') as f:
        pickle.dump([P_total_ring, P_total_complete, P_total_cp, P_total_cp_p, L_creditors_ring, L_creditors_complete, L_creditors_cp, 
                     L_creditors_cp_p, L_creditors_contagion_ring, L_creditors_contagion_complete, L_creditors_contagion_cp, L_creditors_contagion_cp_p, 
                     L_equityholders_ring, L_equityholders_complete, L_equityholders_cp, L_equityholders_cp_p, CCC_t_ring,CCC_t_complete, CCC_t_cp, CCC_t_cp_p ], f)

load_saved_results = False if new_computation else True
if load_saved_results:
    with open('Results_CH3_baseline_c_new', 'rb') as f:
        P_total_ring, P_total_complete, P_total_cp, P_total_cp_p, L_creditors_ring, L_creditors_complete, L_creditors_cp, L_creditors_cp_p, L_creditors_contagion_ring, L_creditors_contagion_complete, L_creditors_contagion_cp, L_creditors_contagion_cp_p, L_equityholders_ring, L_equityholders_complete, L_equityholders_cp, L_equityholders_cp_p, CCC_t_ring,CCC_t_complete, CCC_t_cp, CCC_t_cp_p = pickle.load(f)


# Plot settings
plt.rc('axes', labelsize=12)       
plt.rc('xtick', labelsize=10)       
plt.rc('ytick', labelsize=10)      
plt.rc('legend', fontsize='small')

plotting = False
save = False
if plotting:    
    # Plot system level metrics
    plt.plot(m_range[:50] if effect == "m" else c_range, L_creditors_ring[:50], linestyle=':')
    plt.plot(m_range[:50] if effect == "m" else c_range, L_creditors_complete[:50], linestyle='-.')
    plt.plot(m_range[:50] if effect == "m" else c_range, L_creditors_cp[:50], linestyle='--')
    plt.plot(m_range[:50] if effect == "m" else c_range, L_creditors_cp_p[:50], linestyle='-')
    plt.legend(["Ring network", "Complete network", "Star network (core node)", "Star network (periphery node)"], fontsize='small')
    plt.title("Systemic losses for creditors (total)")
    plt.xlabel("Conversion rate $m$" if effect == "m" else "Issued convertible debt $c$")
    plt.ylabel(r'$L_{\mathrm{creditors, total}}$')
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/losscreditors', bbox_inches="tight", dpi=300) if save else None
    plt.show()

    plt.plot(m_range[:50] if effect == "m" else c_range, L_creditors_contagion_ring[:50], linestyle=':')
    plt.plot(m_range[:50] if effect == "m" else c_range, L_creditors_contagion_complete[:50], linestyle='-.')
    plt.plot(m_range[:50] if effect == "m" else c_range, L_creditors_contagion_cp[:50], linestyle='--')
    plt.plot(m_range[:50] if effect == "m" else c_range, L_creditors_contagion_cp_p[:50], linestyle='-')
    plt.legend(["Ring network", "Complete network", "Star network (core node)", "Star network (periphery node)"], fontsize='small')
    plt.title("Systemic losses for creditors (due to contagion)")
    plt.xlabel("Conversion rate $m$" if effect == "m" else "Issued convertible debt $c$")
    plt.ylabel(r'$L_{\mathrm{creditors, contagion}}$')
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/losscreditorscontagion', bbox_inches="tight", dpi=300) if save else None
    plt.show()

    plt.plot(m_range[:50] if effect == "m" else c_range, L_equityholders_ring[:50], linestyle=':')
    plt.plot(m_range[:50] if effect == "m" else c_range, L_equityholders_complete[:50], linestyle='-.')
    plt.plot(m_range[:50] if effect == "m" else c_range, L_equityholders_cp[:50], linestyle='--')
    plt.plot(m_range[:50] if effect == "m" else c_range, L_equityholders_cp_p[:50], linestyle='-')
    plt.legend(["Ring network", "Complete network", "Star network (core node)", "Star network (periphery node)"], fontsize='small')
    plt.title("Systemic losses for equityholders")
    plt.xlabel("Conversion rate $m$" if effect == "m" else "Issued convertible debt $c$")
    plt.ylabel(r'$L_{\mathrm{equityholders}}$')
    plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/lossequityholder', bbox_inches="tight", dpi=300) if save else None
    plt.show()

    # Plot probability metrics
    for i in range(n):
        plt.stackplot(m_range[:50] if effect == "m" else c_range, P_total_ring[:50, :, i].T)
        plt.legend(["Bankrupt", "Conversion", "Healthy"])
        plt.title(f"Probability of bank {i+1} being in B, C and H (ring network)")
        plt.xlabel("Conversion rate $m$" if effect == "m" else "Issued convertible debt $c$")
        plt.ylabel("Probability (cumulative)")
        plt.savefig(rf'/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/prbbank{i+1}ring', bbox_inches="tight", dpi=300) if save else None
        plt.show()

        plt.stackplot(m_range[:50] if effect == "m" else c_range, P_total_complete[:50, :, i].T)
        plt.legend(["Bankrupt", "Conversion", "Healthy"])
        plt.title(f"Probability of bank {i+1} being in B, C and H (complete network)")
        plt.xlabel("Conversion rate $m$" if effect == "m" else "Issued convertible debt $c$")
        plt.ylabel("Probability (cumulative)")
        plt.savefig(rf'/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/prbbank{i+1}complete', bbox_inches="tight", dpi=300) if save else None
        plt.show()

        plt.stackplot(m_range[:50] if effect == "m" else c_range, P_total_cp[:50, :, i].T)
        plt.legend(["Bankrupt", "Conversion", "Healthy"])
        plt.title(f"Probability of bank {i+1} being in B, C and H (star network, core node)")
        plt.xlabel("Conversion rate $m$" if effect == "m" else "Issued convertible debt $c$")
        plt.ylabel("Probability (cumulative)")
        plt.savefig(rf'/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/prbbank{i+1}coreperiphery', bbox_inches="tight", dpi=300) if save else None
        plt.show()

        plt.stackplot(m_range[:50] if effect == "m" else c_range, P_total_cp_p[:50, :, i].T)
        plt.legend(["Bankrupt", "Conversion", "Healthy"])
        plt.title(f"Probability of bank {i+1} being in B, C and H (star network, periphery node)")
        plt.xlabel("Conversion rate $m$" if effect == "m" else "Issued convertible debt $c$")
        plt.ylabel("Probability (cumulative)")
        plt.savefig(rf'/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/prbbank{i+1}coreperiphery_p', bbox_inches="tight", dpi=300) if save else None
        plt.show()

########################### Analysis of systemic risk (II) ####################################

# Set up the problem 
n = 4
core_banks = [0]
W_ring = generate_network(n, type="ring", fixed=True, weights_fixed = 0.75)
W_complete = generate_network(n, type="complete", fixed=True, weights_fixed = 0.75)
W_cp = generate_network(n, type="core periphery", fixed=True, weights_fixed = 0.75, core_nodes = core_banks)

# Settings for initial asset values
b = np.ones(n) * 0.1                #2
b_cp = np.ones(n) * 0.1             #2
b_cp[core_banks] = 0.5              #5

# Settings for shocks
nbSimulations = 1_000
beta = np.array([0.5, 2, 3.5, 0.9])
rho = 0.4
z_1 = 40
z_2 = 20
X_shock = simulate_shocks_correlated(nbSimulations, beta, rho) 
a_shocked = z_1*(1 - X_shock) - z_2


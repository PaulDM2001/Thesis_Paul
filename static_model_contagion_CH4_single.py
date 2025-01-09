import numpy as np
import matplotlib.pyplot as plt
import time
from helper_functions import generate_network, compute_equilibrium, plot_results_sensitivity
import pickle 

########################### Analysis of systemic risk ####################################

### CONTAGION WITH SINGLE TRANCH ###

# Set up the problem 
n = 4
core_banks = [0]
W_ring = generate_network(n, type="ring", fixed=True, weights_fixed = 0.75)
W_complete = generate_network(n, type="complete", fixed=True, weights_fixed = 0.75)
W_cp = generate_network(n, type="core periphery", fixed=True, weights_fixed = 0.75, core_nodes = core_banks)
W_variants = [W_ring, W_complete, W_cp]

# Settings for shocks
np.random.seed(0)   
nbSimulations = 1000
shocked_banks = [0]     
shocked_banks_periphery = [1]
beta = 1.5
atilde0 = 34
d0 = 17
X_shock = atilde0*(1 - np.random.beta(1, beta, size=(nbSimulations, 1))) - d0

## Impact of m and c on PD, PC, PH, Lc, Lcc, Leq
m_range = np.linspace(0.0, 5.0, 51)
m_range[0] = 0.01
c_range = np.linspace(0.0, 50.0, 51) + 1e-3      #perturb by 1e-3 to prevent numerical issue
c_range[0] = 0.01
effect = "m"

def compute_systemic_risk_metrics(l, c, m, W, s_initial, shocked_banks, a_simulations):
    B_array = np.zeros((a_simulations.shape[0], a_simulations.shape[1]))
    C_array = np.zeros((a_simulations.shape[0], a_simulations.shape[1]))
    H_array = np.zeros((a_simulations.shape[0], a_simulations.shape[1]))
    Prb_array = np.zeros((3, a_simulations.shape[1]))
    value_creditors_array = np.zeros((a_simulations.shape[0], 1))
    value_creditors_contagion_array = np.zeros((a_simulations.shape[0], 1))
    value_equityholders_array = np.zeros((a_simulations.shape[0], 1))
    value_equityholders_contagion_array = np.zeros((a_simulations.shape[0], 1))

    nbInfeasible = 0
    for i in range(a_simulations.shape[0]):  
        # Solve problem
        B, C, H, s = compute_equilibrium(l, c, m, W, a_simulations[i, :])
        if isinstance(s, int):
            nbInfeasible +=1
            continue

        B_array[i, :] = B
        C_array[i, :] = C
        H_array[i, :] = H
        SA = np.array([0 if i in shocked_banks else 1 for i in range(a_simulations.shape[1])])

        value_creditors_array[i] = np.sum(c * H + m*s * C) 
        value_creditors_contagion_array[i] = np.sum(c * H * SA + m*s * C * SA) 
        value_equityholders_array[i] = np.sum(s_initial - s * (1 - B))
        value_equityholders_contagion_array[i] = np.sum((s_initial - s * (1 - B)) * SA)
        
    Prb_array[0, :] = B_array.mean(axis=0)
    Prb_array[1, :] = C_array.mean(axis=0)
    Prb_array[2, :] = H_array.mean(axis=0)
    
    if nbInfeasible > 0:
        print(f"Warning: {nbInfeasible} simulations lead to an infeasible outcome of the MILP-solver.")
    
    return Prb_array, value_creditors_array, value_creditors_contagion_array, value_equityholders_array, value_equityholders_contagion_array

def compute_sensitivity_contract_parameters(n, W_variants, core_banks, shocked_banks, shocked_banks_periphery, X_shock, m_range, c_range, effect):
    P_total_ring = np.zeros((m_range.shape[0], 3, n))
    P_total_complete = np.zeros((m_range.shape[0], 3, n))
    P_total_cp = np.zeros((m_range.shape[0], 3, n))
    P_total_cp_p = np.zeros((m_range.shape[0], 3, n))

    L_creditors_ring = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], X_shock.shape[0]))
    L_creditors_complete = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], X_shock.shape[0]))
    L_creditors_cp = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], X_shock.shape[0]))
    L_creditors_cp_p = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], X_shock.shape[0]))

    L_creditors_contagion_ring = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], X_shock.shape[0]))
    L_creditors_contagion_complete = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], X_shock.shape[0]))
    L_creditors_contagion_cp = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], X_shock.shape[0]))
    L_creditors_contagion_cp_p = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], X_shock.shape[0]))

    L_creditors_direct_ring = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], X_shock.shape[0]))
    L_creditors_direct_complete = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], X_shock.shape[0]))
    L_creditors_direct_cp = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], X_shock.shape[0]))
    L_creditors_direct_cp_p = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], X_shock.shape[0]))

    L_equityholders_ring = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], X_shock.shape[0]))
    L_equityholders_complete = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], X_shock.shape[0]))
    L_equityholders_cp = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], X_shock.shape[0]))
    L_equityholders_cp_p = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], X_shock.shape[0]))

    L_equityholders_contagion_ring = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], X_shock.shape[0]))
    L_equityholders_contagion_complete = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], X_shock.shape[0]))
    L_equityholders_contagion_cp = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], X_shock.shape[0]))
    L_equityholders_contagion_cp_p = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], X_shock.shape[0]))

    L_equityholders_direct_ring = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], X_shock.shape[0]))
    L_equityholders_direct_complete = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], X_shock.shape[0]))
    L_equityholders_direct_cp = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], X_shock.shape[0]))
    L_equityholders_direct_cp_p = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], X_shock.shape[0]))

    st = time.time()
    et = time.time()
    for k, x in enumerate(m_range if effect == "m" else c_range):
        print(f"Iteration {k}/{m_range.shape[0]}, elapsed time: {np.round(et-st,2)} seconds")#, end='\r')
        SA = np.array([0 if i in shocked_banks else 1 for i in range(n)])
        SAp = np.array([0 if i in shocked_banks_periphery else 1 for i in range(n)])
        # Determine parameters based on chosen effect
        if effect == "m": 
            m = np.ones(n) 
            m[shocked_banks] = x
            m_cp_p = np.ones(n) 
            m_cp_p[shocked_banks_periphery] = x
            c_rc = np.ones(n) * 12
            c_cp = np.ones(n) * 12
            c_cp[core_banks] = 12 * (n - len(core_banks))
            c_cp_p = c_cp

        elif effect == "c": 
            m = np.ones(n) 
            m_cp_p = np.ones(n) 
            c_rc = np.ones(n) * 12
            c_rc[shocked_banks] = x
            c_cp = np.ones(n) * 12
            c_cp[shocked_banks] = x
            c_cp[core_banks] *= (n - len(core_banks))
            c_cp_p = np.ones(n) * 12
            c_cp_p[shocked_banks_periphery] = x 
            c_cp_p[core_banks] *= (n - len(core_banks))

        else:
            raise Exception("No valid effect chosen. Calculation stops")

        # Compute thresholds
        l_rc = np.round(c_rc / m, 6)
        l_cp = np.round(c_cp / m, 6)
        l_cp_p = np.round(c_cp_p / m_cp_p, 6)

        # Compute initial assets and shocked assets
        a_init_rc = np.array([17.0, 17.0, 17.0, 17.0])
        a_init_cp = np.array([51.0, 17.0, 17.0, 17.0])
        a_rc = np.tile(a_init_rc, (X_shock.shape[0], 1)) 
        a_cp = np.tile(a_init_cp, (X_shock.shape[0], 1))
        a_cp_p = np.tile(a_init_cp, (X_shock.shape[0], 1))

        a_rc[:, shocked_banks] = X_shock
        a_cp[:, shocked_banks] = X_shock * (n-len(core_banks))
        a_cp_p[:, shocked_banks_periphery] = X_shock

        # Compute initial prices
        _, C_init_r, H_init_r, s_init_r = compute_equilibrium(l_rc, c_rc, m, W_variants[0], a_init_rc)
        _, C_init_c, H_init_c, s_init_c = compute_equilibrium(l_rc, c_rc, m, W_variants[1], a_init_rc)
        _, C_init_cp, H_init_cp, s_init_cp = compute_equilibrium(l_cp, c_cp, m, W_variants[2], a_init_cp)
        _, C_init_cp_p, H_init_cp_p, s_init_cp_p = compute_equilibrium(l_cp_p, c_cp_p, m_cp_p, W_variants[2], a_init_cp)

        init_value_creditors_r = np.sum(c_rc * H_init_r + m*s_init_r * C_init_r) 
        init_value_creditors_c = np.sum(c_rc * H_init_c + m*s_init_c* C_init_c)
        init_value_creditors_cp = np.sum(c_cp * H_init_cp + m*s_init_cp* C_init_cp) 
        init_value_creditors_cp_p = np.sum(c_cp_p * H_init_cp_p + m_cp_p*s_init_cp_p* C_init_cp_p) 
        
        init_value_creditors_contagion_r = np.sum(c_rc * H_init_r * SA + m*s_init_r * C_init_r * SA)
        init_value_creditors_contagion_c = np.sum(c_rc * H_init_c * SA + m*s_init_c * C_init_c * SA) 
        init_value_creditors_contagion_cp = np.sum(c_cp * H_init_cp * SA + m*s_init_cp* C_init_cp * SA) 
        init_value_creditors_contagion_cp_p = np.sum(c_cp_p * H_init_cp_p * SAp+ m_cp_p*s_init_cp_p* C_init_cp_p * SAp) 

        # Compute systemic risk metrics
        Prb_array_ring, L_creditors_array_ring, L_creditors_contagion_array_ring, L_equityholders_array_ring, L_equityholders_contagion_array_ring = compute_systemic_risk_metrics(l_rc, c_rc, m, W_variants[0], s_init_r, shocked_banks, np.round(a_rc, 6))
        Prb_array_cplt, L_creditors_array_cplt, L_creditors_contagion_array_cplt, L_equityholders_array_cplt, L_equityholders_contagion_array_cplt  = compute_systemic_risk_metrics(l_rc, c_rc, m, W_variants[1], s_init_c, shocked_banks, np.round(a_rc, 6))
        Prb_array_cp, L_creditors_array_cp, L_creditors_contagion_array_cp, L_equityholders_array_cp, L_equityholders_contagion_array_cp  = compute_systemic_risk_metrics(l_cp, c_cp, m, W_variants[2], s_init_cp, shocked_banks, np.round(a_cp, 6))
        Prb_array_cp_p, L_creditors_array_cp_p, L_creditors_contagion_array_cp_p, L_equityholders_array_cp_p, L_equityholders_contagion_array_cp_p  = compute_systemic_risk_metrics(l_cp_p, c_cp_p, m_cp_p, W_variants[2], s_init_cp_p, shocked_banks_periphery, np.round(a_cp_p, 6))

        # Store systemic risk metrics
        P_total_ring[k, :, :] = Prb_array_ring
        P_total_complete[k, :, :] = Prb_array_cplt
        P_total_cp[k, :, :] = Prb_array_cp
        P_total_cp_p[k, :, :] = Prb_array_cp_p

        L_creditors_ring[k, :] = (init_value_creditors_r - L_creditors_array_ring.squeeze()) / init_value_creditors_r 
        L_creditors_complete[k, :] =  (init_value_creditors_c - L_creditors_array_cplt.squeeze()) / init_value_creditors_c
        L_creditors_cp[k, :] = (init_value_creditors_cp - L_creditors_array_cp.squeeze()) / init_value_creditors_cp
        L_creditors_cp_p[k, :] = (init_value_creditors_cp_p - L_creditors_array_cp_p.squeeze()) / init_value_creditors_cp_p

        L_creditors_contagion_ring[k, :] = (init_value_creditors_contagion_r - L_creditors_contagion_array_ring.squeeze()) / init_value_creditors_r        
        L_creditors_contagion_complete[k, :] = (init_value_creditors_contagion_c - L_creditors_contagion_array_cplt.squeeze()) / init_value_creditors_c
        L_creditors_contagion_cp[k, :] = (init_value_creditors_contagion_cp - L_creditors_contagion_array_cp.squeeze()) / init_value_creditors_cp
        L_creditors_contagion_cp_p[k, :] = (init_value_creditors_contagion_cp_p - L_creditors_contagion_array_cp_p.squeeze()) / init_value_creditors_cp_p 

        L_creditors_direct_ring[k, :] = ((init_value_creditors_r -  init_value_creditors_contagion_r) - (L_creditors_array_ring.squeeze() - L_creditors_contagion_array_ring.squeeze())) / init_value_creditors_r                  
        L_creditors_direct_complete[k, :] = ((init_value_creditors_c -  init_value_creditors_contagion_c) - (L_creditors_array_cplt.squeeze() - L_creditors_contagion_array_cplt.squeeze())) / init_value_creditors_c
        L_creditors_direct_cp[k, :] = ((init_value_creditors_cp -  init_value_creditors_contagion_cp) - (L_creditors_array_cp.squeeze() - L_creditors_contagion_array_cp.squeeze())) / init_value_creditors_cp
        L_creditors_direct_cp_p[k, :] = ((init_value_creditors_cp_p -  init_value_creditors_contagion_cp_p) - (L_creditors_array_cp_p.squeeze() - L_creditors_contagion_array_cp_p.squeeze())) / init_value_creditors_cp_p 

        L_equityholders_ring[k, :] = L_equityholders_array_ring.squeeze() / np.sum(s_init_r)
        L_equityholders_complete[k, :] = L_equityholders_array_cplt.squeeze() / np.sum(s_init_c)
        L_equityholders_cp[k, :] = L_equityholders_array_cp.squeeze() / np.sum(s_init_cp)
        L_equityholders_cp_p[k, :] = L_equityholders_array_cp_p.squeeze() / np.sum(s_init_cp_p)

        L_equityholders_contagion_ring[k, :] = L_equityholders_contagion_array_ring.squeeze() / np.sum(s_init_r)
        L_equityholders_contagion_complete[k, :] = L_equityholders_contagion_array_cplt.squeeze() / np.sum(s_init_c)
        L_equityholders_contagion_cp[k, :] = L_equityholders_contagion_array_cp.squeeze() / np.sum(s_init_cp)
        L_equityholders_contagion_cp_p[k, :] = L_equityholders_contagion_array_cp_p.squeeze() / np.sum(s_init_cp_p)

        L_equityholders_direct_ring[k, :] = (L_equityholders_array_ring.squeeze() - L_equityholders_contagion_array_ring.squeeze()) / np.sum(s_init_r)
        L_equityholders_direct_complete[k, :] = (L_equityholders_array_cplt.squeeze() - L_equityholders_contagion_array_cplt.squeeze()) / np.sum(s_init_c)
        L_equityholders_direct_cp[k, :] = (L_equityholders_array_cp.squeeze() - L_equityholders_contagion_array_cp.squeeze()) / np.sum(s_init_cp)
        L_equityholders_direct_cp_p[k, :] = (L_equityholders_array_cp_p.squeeze() - L_equityholders_contagion_array_cp_p.squeeze()) / np.sum(s_init_cp_p)

        
        # Time
        et = time.time()

    return P_total_ring, P_total_complete, P_total_cp, P_total_cp_p, L_creditors_ring, L_creditors_complete, L_creditors_cp, L_creditors_cp_p, \
        L_creditors_contagion_ring, L_creditors_contagion_complete, L_creditors_contagion_cp, L_creditors_contagion_cp_p, L_creditors_direct_ring, \
        L_creditors_direct_complete, L_creditors_direct_cp, L_creditors_direct_cp_p, L_equityholders_ring, L_equityholders_complete, L_equityholders_cp, \
        L_equityholders_cp_p, L_equityholders_contagion_ring, L_equityholders_contagion_complete, L_equityholders_contagion_cp, L_equityholders_contagion_cp_p, \
        L_equityholders_direct_ring, L_equityholders_direct_complete, L_equityholders_direct_cp, L_equityholders_direct_cp_p

new_computation = False
if new_computation:
   P_total_ring, P_total_complete, P_total_cp, P_total_cp_p, L_creditors_ring, L_creditors_complete, L_creditors_cp, L_creditors_cp_p, \
        L_creditors_contagion_ring, L_creditors_contagion_complete, L_creditors_contagion_cp, L_creditors_contagion_cp_p, L_creditors_direct_ring, \
        L_creditors_direct_complete, L_creditors_direct_cp, L_creditors_direct_cp_p, L_equityholders_ring, L_equityholders_complete, L_equityholders_cp, \
        L_equityholders_cp_p, L_equityholders_contagion_ring, L_equityholders_contagion_complete, L_equityholders_contagion_cp, L_equityholders_contagion_cp_p, \
        L_equityholders_direct_ring, L_equityholders_direct_complete, L_equityholders_direct_cp, L_equityholders_direct_cp_p = compute_sensitivity_contract_parameters(n, W_variants, core_banks, shocked_banks, shocked_banks_periphery, X_shock, m_range, c_range, effect)

save_results = False
load_results = True
if save_results:
    with open('Analysis contagion/Results_4_contagion_single_m_1000sim', 'wb') as f:
        pickle.dump([P_total_ring, P_total_complete, P_total_cp, P_total_cp_p, L_creditors_ring, L_creditors_complete, L_creditors_cp, L_creditors_cp_p, \
        L_creditors_contagion_ring, L_creditors_contagion_complete, L_creditors_contagion_cp, L_creditors_contagion_cp_p, L_creditors_direct_ring, \
        L_creditors_direct_complete, L_creditors_direct_cp, L_creditors_direct_cp_p, L_equityholders_ring, L_equityholders_complete, L_equityholders_cp, \
        L_equityholders_cp_p, L_equityholders_contagion_ring, L_equityholders_contagion_complete, L_equityholders_contagion_cp, L_equityholders_contagion_cp_p, \
        L_equityholders_direct_ring, L_equityholders_direct_complete, L_equityholders_direct_cp, L_equityholders_direct_cp_p], f)

if load_results:
    with open('Analysis contagion/Results_4_contagion_single_m_1000sim', 'rb') as f:
        P_total_ring, P_total_complete, P_total_cp, P_total_cp_p, L_creditors_ring, L_creditors_complete, L_creditors_cp, L_creditors_cp_p, \
        L_creditors_contagion_ring, L_creditors_contagion_complete, L_creditors_contagion_cp, L_creditors_contagion_cp_p, L_creditors_direct_ring, \
        L_creditors_direct_complete, L_creditors_direct_cp, L_creditors_direct_cp_p, L_equityholders_ring, L_equityholders_complete, L_equityholders_cp, \
        L_equityholders_cp_p, L_equityholders_contagion_ring, L_equityholders_contagion_complete, L_equityholders_contagion_cp, L_equityholders_contagion_cp_p, \
        L_equityholders_direct_ring, L_equityholders_direct_complete, L_equityholders_direct_cp, L_equityholders_direct_cp_p = pickle.load(f)

plot_results = True
single = True
if plot_results:
    plot_results_sensitivity(m_range, c_range, effect, single, P_total_ring, P_total_complete, P_total_cp, P_total_cp_p, L_creditors_ring, L_creditors_complete, L_creditors_cp, L_creditors_cp_p, \
        L_creditors_contagion_ring, L_creditors_contagion_complete, L_creditors_contagion_cp, L_creditors_contagion_cp_p, L_creditors_direct_ring, \
        L_creditors_direct_complete, L_creditors_direct_cp, L_creditors_direct_cp_p, L_equityholders_ring, L_equityholders_complete, L_equityholders_cp, \
        L_equityholders_cp_p, L_equityholders_contagion_ring, L_equityholders_contagion_complete, L_equityholders_contagion_cp, L_equityholders_contagion_cp_p, \
        L_equityholders_direct_ring, L_equityholders_direct_complete, L_equityholders_direct_cp, L_equityholders_direct_cp_p, save=True)
    
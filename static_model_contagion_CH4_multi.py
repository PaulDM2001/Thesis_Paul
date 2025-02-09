import numpy as np
import matplotlib.pyplot as plt
import time
from helper_functions import generate_network, compute_equilibrium_multiple_thresholds, plot_results_sensitivity
import pickle 

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
xi = np.array([0.3, 1.0, 1.0, 1.0])
xi_cp_p = np.array([1.0, 0.3, 1.0, 1.0])
zeta = np.array([0.5, 1.0, 1.0, 1.0])
zeta_cp_p = np.array([1.0, 0.5, 1.0, 1.0])           

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
c_range = np.linspace(0.0, 50, 51) + 1e-3
c_range[0] = 0.01         
effect = "c"

def compute_systemic_risk_metrics_multitranch(l, c, m, W, s_initial, shocked_banks, a_simulations):
    B_array = np.zeros((a_simulations.shape[0], a_simulations.shape[1]))
    G2_array = np.zeros((a_simulations.shape[0], a_simulations.shape[1]))
    G1_array = np.zeros((a_simulations.shape[0], a_simulations.shape[1]))
    H_array = np.zeros((a_simulations.shape[0], a_simulations.shape[1]))
    Prb_array = np.zeros((4, a_simulations.shape[1]))
    value_creditors_array = np.zeros((a_simulations.shape[0]))
    value_creditors_contagion_array = np.zeros((a_simulations.shape[0]))
    value_equityholders_array = np.zeros((a_simulations.shape[0]))
    value_equityholders_contagion_array = np.zeros((a_simulations.shape[0]))

    nbInfeasible = 0
    for i in range(a_simulations.shape[0]):  
        # Solve problem
        B, G2, G1, H, s = compute_equilibrium_multiple_thresholds(l, c, m, W, a_simulations[i, :], s_initial, combination_multi_single=True)
        if isinstance(s, int):
            nbInfeasible +=1
            continue

        B_array[i, :] = B
        G2_array[i, :] = G2
        G1_array[i, :] = G1
        H_array[i, :] = H

        SA = np.array([0 if i in shocked_banks else 1 for i in range(a_simulations.shape[1])])

        value_creditors_array[i] = np.sum((c[:, 0] + c[:, 1]) * H + (c[:, 1] + m[:, 0]*s) * G1 + (m[:, 0] + m[:, 1]) * s * G2)
        value_creditors_contagion_array[i] = np.sum(((c[:, 0] + c[:, 1]) * H+ (c[:, 1] + m[:, 0]*s) * G1 + (m[:, 0] + m[:, 1]) * s * G2)*SA) 
        value_equityholders_array[i] = np.sum(s_initial - s * (1 - B))
        value_equityholders_contagion_array[i] = np.sum((s_initial - s * (1 - B)) * SA)
    
    Prb_array[0, :] = B_array.mean(axis=0)
    Prb_array[1, :] = G2_array.mean(axis=0)
    Prb_array[2, :] = G1_array.mean(axis=0)
    Prb_array[3, :] = H_array.mean(axis=0)
    
    if nbInfeasible > 0:
        print(f"Warning: {nbInfeasible} simulations lead to an infeasible outcome of the MILP-solver.")
    
    return Prb_array, value_creditors_array, value_creditors_contagion_array, value_equityholders_array, value_equityholders_contagion_array


def compute_sensitivity_contract_parameters_multitranch(n, W_variants, zeta, zeta_cp_p, xi, xi_cp_p, core_banks, shocked_banks, shocked_banks_periphery, X_shock, m_range, c_range, effect):
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
    
    L_creditors_direct_ring = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_creditors_direct_complete = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_creditors_direct_cp = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_creditors_direct_cp_p = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))

    L_equityholders_ring = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_equityholders_complete = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_equityholders_cp = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_equityholders_cp_p = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))

    L_equityholders_contagion_ring = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_equityholders_contagion_complete = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_equityholders_contagion_cp = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_equityholders_contagion_cp_p = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))

    L_equityholders_direct_ring = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_equityholders_direct_complete = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_equityholders_direct_cp = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))
    L_equityholders_direct_cp_p = np.zeros((m_range.shape[0] if effect == "m" else c_range.shape[0], 1))

    st = time.time()
    et = time.time()
    for k, x in enumerate(m_range if effect == "m" else c_range):
        print(f"Iteration {k}/{m_range.shape[0]}, elapsed time: {np.round(et-st,2)} seconds", end='\n')
        SA = np.array([0 if i in shocked_banks else 1 for i in range(n)])
        SAp = np.array([0 if i in shocked_banks_periphery else 1 for i in range(n)])

        # Determine parameters based on chosen effect
        if effect == "m": 
            m_total = np.ones(n) 
            m_total[shocked_banks] = x
            m_total_cp_p = np.ones(n) 
            m_total_cp_p[shocked_banks_periphery] = x
            c_coeff = 12
            c_total_rc = np.ones(n) * c_coeff 
            c_total_cp = np.ones(n) * c_coeff 
            c_total_cp[core_banks] = c_coeff * (n - len(core_banks))
            c_total_cp_p = c_total_cp

        elif effect == "c": 
            m_total = np.ones(n) 
            m_total_cp_p = np.ones(n) 
            c_total_rc = np.ones(n) * 12 
            c_total_rc[shocked_banks] = x
            c_total_cp = np.ones(n) * 12
            c_total_cp[shocked_banks] = x
            c_total_cp[core_banks] = c_total_cp[core_banks] * (n - len(core_banks))
            c_total_cp_p = np.ones(n) * 12 
            c_total_cp_p[shocked_banks_periphery] = x 
            c_total_cp_p[core_banks] *= (n - len(core_banks))

        else:
            raise Exception("No valid effect chosen. Calculation stops")

        m_low = np.round(m_total * (1 - xi), 6)
        m_high = np.round(m_total * xi, 6)
        c_low_rc = np.round(c_total_rc * (1 - zeta), 6)
        c_high_rc = np.round(c_total_rc* zeta, 6)

        c_low_cp = c_total_cp * (1 - zeta)
        c_high_cp = c_total_cp * zeta

        m_low_cp_p = np.round(m_total_cp_p * (1 - xi_cp_p), 6)
        m_high_cp_p = np.round(m_total_cp_p * xi_cp_p, 6)
        c_low_cp_p = np.round(c_total_cp_p  * (1 - zeta_cp_p), 6)
        c_high_cp_p = np.round(c_total_cp_p * zeta_cp_p, 6)

        l_low_rc = np.round(c_low_rc / m_low, 6)
        l_high_rc = np.round(c_high_rc / m_high, 6)
        l_low_cp = np.round(c_low_cp / m_low, 6)
        l_high_cp = np.round(c_high_cp / m_high, 6)
        l_low_cp_p = np.round(c_low_cp_p / m_low_cp_p, 6)
        l_high_cp_p = np.round(c_high_cp_p / m_high_cp_p, 6)
  
        l_rc = np.vstack((l_high_rc, l_low_rc)).T
        l_rc[1:, 1] = 0 # Set thresholds of second tranch to zero for the non-shocked banks, division gives NaN  
        l_cp = np.vstack((l_high_cp, l_low_cp)).T
        l_cp[1:, 1] = 0 # Set thresholds of second tranch to zero for the non-shocked banks, division gives NaN    
        l_cp_p = np.vstack((l_high_cp_p, l_low_cp_p)).T
        l_cp_p[0, 1] = 0 # Set thresholds of second tranch to zero for the non-shocked banks, division gives NaN    
        l_cp_p[2:, 1] = 0 # Set thresholds of second tranch to zero for the non-shocked banks, division gives NaN    

        m = np.vstack((m_high, m_low)).T
        m_cp_p = np.vstack((m_high_cp_p, m_low_cp_p)).T

        c_rc = np.vstack((c_high_rc, c_low_rc)).T
        c_cp = np.vstack((c_high_cp, c_low_cp)).T
        c_cp_p = np.vstack((c_high_cp_p, c_low_cp_p)).T

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
        _, G2_init_r, G1_init_r, H_init_r, s_init_r = compute_equilibrium_multiple_thresholds(np.round(l_rc, 6), np.round(c_rc, 6), np.round(m, 6), np.tile(W_variants[0], (2, 1, 1)), np.round(a_init_rc, 6), combination_multi_single=True)
        _, G2_init_c, G1_init_c, H_init_c, s_init_c = compute_equilibrium_multiple_thresholds(np.round(l_rc, 6), np.round(c_rc, 6), np.round(m, 6), np.tile(W_variants[1], (2, 1, 1)), np.round(a_init_rc, 6), combination_multi_single=True)
        _, G2_init_cp, G1_init_cp, H_init_cp, s_init_cp = compute_equilibrium_multiple_thresholds(l_cp, c_cp, m, np.tile(W_variants[2], (2, 1, 1)), np.round(a_init_cp, 6), combination_multi_single=True)
        _, G2_init_cp_p, G1_init_cp_p, H_init_cp_p, s_init_cp_p = compute_equilibrium_multiple_thresholds(l_cp_p, c_cp_p, m_cp_p, np.tile(W_variants[2], (2, 1, 1)), np.round(a_init_cp, 6), combination_multi_single=True)

        init_value_creditors_r = np.sum((c_rc[:, 0] + c_rc[:, 1]) * H_init_r + (c_rc[:, 1] + m[:, 0]*s_init_r) * G1_init_r + (m[:, 0]*s_init_r + m[:, 1]*s_init_r) * G2_init_r) 
        init_value_creditors_c = np.sum((c_rc[:, 0] + c_rc[:, 1]) * H_init_c + (c_rc[:, 1] + m[:, 0]*s_init_c) * G1_init_c + (m[:, 0] + m[:, 1]) * s_init_c * G2_init_c) 
        init_value_creditors_cp = np.sum((c_cp[:, 0] + c_cp[:, 1]) * H_init_cp + (c_cp[:, 1] + m[:, 0]*s_init_cp) * G1_init_cp + (m[:, 0]*s_init_cp + m[:, 1]*s_init_cp) * G2_init_cp) 
        init_value_creditors_cp_p = np.sum((c_cp_p[:, 0] + c_cp_p[:, 1]) * H_init_cp_p + (c_cp_p[:, 1] + m_cp_p[:, 0]*s_init_cp_p) * G1_init_cp_p + (m_cp_p[:, 0] + m_cp_p[:, 1]) * s_init_cp_p * G2_init_cp_p) 
        
        init_value_creditors_contagion_r = np.sum(((c_rc[:, 0] + c_rc[:, 1]) * H_init_r + (c_rc[:, 1] + m[:, 0]*s_init_r) * G1_init_r + (m[:, 0]*s_init_r + m[:, 1]*s_init_r) * G2_init_r)*SA)
        init_value_creditors_contagion_c = np.sum(((c_rc[:, 0] + c_rc[:, 1]) * H_init_c + (c_rc[:, 1] + m[:, 0]*s_init_c) * G1_init_c + (m[:, 0] + m[:, 1]) * s_init_c * G2_init_c)*SA) 
        init_value_creditors_contagion_cp = np.sum(((c_cp[:, 0] + c_cp[:, 1]) * H_init_cp + (c_cp[:, 1] + m[:, 0]*s_init_cp) * G1_init_cp + (m[:, 0]*s_init_cp + m[:, 1]*s_init_cp) * G2_init_cp)*SA) 
        init_value_creditors_contagion_cp_p = np.sum(((c_cp_p[:, 0] + c_cp_p[:, 1]) * H_init_cp_p + (c_cp_p[:, 1] + m_cp_p[:, 0]*s_init_cp_p) * G1_init_cp_p+ (m_cp_p[:, 0] + m_cp_p[:, 1]) * s_init_cp_p * G2_init_cp_p)*SAp)
    
        # Compute systemic risk metrics        
        Prb_array_ring, L_creditors_array_ring, L_creditors_contagion_array_ring, L_equityholders_array_ring, L_equityholders_contagion_array_ring = compute_systemic_risk_metrics_multitranch(l_rc, c_rc, m,  np.tile(W_variants[0], (2, 1, 1)), s_init_r, shocked_banks, np.round(a_rc, 6))
        Prb_array_cplt, L_creditors_array_cplt, L_creditors_contagion_array_cplt, L_equityholders_array_cplt, L_equityholders_contagion_array_cplt  = compute_systemic_risk_metrics_multitranch(l_rc, c_rc, m, np.tile(W_variants[1], (2, 1, 1)), s_init_c, shocked_banks, np.round(a_rc, 6))
        Prb_array_cp, L_creditors_array_cp, L_creditors_contagion_array_cp, L_equityholders_array_cp, L_equityholders_contagion_array_cp = compute_systemic_risk_metrics_multitranch(l_cp, c_cp, m, np.tile(W_variants[2], (2, 1, 1)), s_init_cp, shocked_banks, np.round(a_cp, 6))
        Prb_array_cp_p, L_creditors_array_cp_p, L_creditors_contagion_array_cp_p, L_equityholders_array_cp_p, L_equityholders_contagion_array_cp_p = compute_systemic_risk_metrics_multitranch(l_cp_p, c_cp_p, m_cp_p, np.tile(W_variants[2], (2, 1, 1)), s_init_cp_p, shocked_banks_periphery, np.round(a_cp_p, 6))

        # Store systemic risk metrics
        P_total_ring[k, :, :] = Prb_array_ring
        P_total_complete[k, :, :] = Prb_array_cplt
        P_total_cp[k, :, :] = Prb_array_cp
        P_total_cp_p[k, :, :] = Prb_array_cp_p

        L_creditors_ring[k] = (init_value_creditors_r - L_creditors_array_ring.mean()) / init_value_creditors_r 
        L_creditors_complete[k] = (init_value_creditors_c - L_creditors_array_cplt.mean()) / init_value_creditors_c
        L_creditors_cp[k] =  (init_value_creditors_cp - L_creditors_array_cp.mean()) / init_value_creditors_cp
        L_creditors_cp_p[k] =  (init_value_creditors_cp_p - L_creditors_array_cp_p.mean()) / init_value_creditors_cp_p

        L_creditors_contagion_ring[k] = (init_value_creditors_contagion_r - L_creditors_contagion_array_ring.mean()) / init_value_creditors_r 
        L_creditors_contagion_complete[k] = (init_value_creditors_contagion_c - L_creditors_contagion_array_cplt.mean()) / init_value_creditors_contagion_c
        L_creditors_contagion_cp[k] = (init_value_creditors_contagion_cp - L_creditors_contagion_array_cp.mean()) / init_value_creditors_cp
        L_creditors_contagion_cp_p[k] =  (init_value_creditors_contagion_cp_p - L_creditors_contagion_array_cp_p.mean()) / init_value_creditors_cp_p

        L_creditors_direct_ring[k] = ((init_value_creditors_r -  init_value_creditors_contagion_r) - (L_creditors_array_ring.mean() - L_creditors_contagion_array_ring.mean())) / init_value_creditors_r                  
        L_creditors_direct_complete[k] = ((init_value_creditors_c -  init_value_creditors_contagion_c) - (L_creditors_array_cplt.mean() - L_creditors_contagion_array_cplt.mean())) / init_value_creditors_c
        L_creditors_direct_cp[k] = ((init_value_creditors_cp -  init_value_creditors_contagion_cp) - (L_creditors_array_cp.mean() - L_creditors_contagion_array_cp.mean())) / init_value_creditors_cp
        L_creditors_direct_cp_p[k] = ((init_value_creditors_cp_p -  init_value_creditors_contagion_cp_p) - (L_creditors_array_cp_p.mean() - L_creditors_contagion_array_cp_p.mean())) / init_value_creditors_cp_p 

        L_equityholders_ring[k] = L_equityholders_array_ring.mean() / np.sum(s_init_r)
        L_equityholders_complete[k] = L_equityholders_array_cplt.mean() / np.sum(s_init_c)
        L_equityholders_cp[k] = L_equityholders_array_cp.mean() / np.sum(s_init_cp)
        L_equityholders_cp_p[k] = L_equityholders_array_cp_p.mean()/ np.sum(s_init_cp_p)

        L_equityholders_contagion_ring[k] = L_equityholders_contagion_array_ring.mean() / np.sum(s_init_r)
        L_equityholders_contagion_complete[k] = L_equityholders_contagion_array_cplt.mean() / np.sum(s_init_c)
        L_equityholders_contagion_cp[k] = L_equityholders_contagion_array_cp.mean() / np.sum(s_init_cp)
        L_equityholders_contagion_cp_p[k] = L_equityholders_contagion_array_cp_p.mean() / np.sum(s_init_cp_p)

        L_equityholders_direct_ring[k] = (L_equityholders_array_ring.mean() - L_equityholders_contagion_array_ring.mean()) / np.sum(s_init_r)
        L_equityholders_direct_complete[k] = (L_equityholders_array_cplt.mean() - L_equityholders_contagion_array_cplt.mean()) / np.sum(s_init_c)
        L_equityholders_direct_cp[k] = (L_equityholders_array_cp.mean() - L_equityholders_contagion_array_cp.mean()) / np.sum(s_init_cp)
        L_equityholders_direct_cp_p[k] = (L_equityholders_array_cp_p.mean() - L_equityholders_contagion_array_cp_p.mean()) / np.sum(s_init_cp_p)
        
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
        L_equityholders_direct_ring, L_equityholders_direct_complete, L_equityholders_direct_cp, L_equityholders_direct_cp_p  = compute_sensitivity_contract_parameters_multitranch(n, W_variants, zeta, zeta_cp_p, xi, xi_cp_p, core_banks, shocked_banks, shocked_banks_periphery, X_shock, m_range, c_range, effect)

save_results = False
load_results = True
if save_results:
    with open('Analysis contagion/Results_4_contagion_multi_c_1000sim', 'wb') as f:
        pickle.dump([P_total_ring, P_total_complete, P_total_cp, P_total_cp_p, L_creditors_ring, L_creditors_complete, L_creditors_cp, L_creditors_cp_p, \
        L_creditors_contagion_ring, L_creditors_contagion_complete, L_creditors_contagion_cp, L_creditors_contagion_cp_p, L_creditors_direct_ring, \
        L_creditors_direct_complete, L_creditors_direct_cp, L_creditors_direct_cp_p, L_equityholders_ring, L_equityholders_complete, L_equityholders_cp, \
        L_equityholders_cp_p, L_equityholders_contagion_ring, L_equityholders_contagion_complete, L_equityholders_contagion_cp, L_equityholders_contagion_cp_p, \
        L_equityholders_direct_ring, L_equityholders_direct_complete, L_equityholders_direct_cp, L_equityholders_direct_cp_p], f)

if load_results:
    with open('Analysis contagion/Results_4_contagion_multi_c_1000sim', 'rb') as f:
        P_total_ring, P_total_complete, P_total_cp, P_total_cp_p, L_creditors_ring, L_creditors_complete, L_creditors_cp, L_creditors_cp_p, \
        L_creditors_contagion_ring, L_creditors_contagion_complete, L_creditors_contagion_cp, L_creditors_contagion_cp_p, L_creditors_direct_ring, \
        L_creditors_direct_complete, L_creditors_direct_cp, L_creditors_direct_cp_p, L_equityholders_ring, L_equityholders_complete, L_equityholders_cp, \
        L_equityholders_cp_p, L_equityholders_contagion_ring, L_equityholders_contagion_complete, L_equityholders_contagion_cp, L_equityholders_contagion_cp_p, \
        L_equityholders_direct_ring, L_equityholders_direct_complete, L_equityholders_direct_cp, L_equityholders_direct_cp_p = pickle.load(f)

plot_results = True
single = False
if plot_results:
    plot_results_sensitivity(m_range, c_range, effect, single, P_total_ring, P_total_complete, P_total_cp, P_total_cp_p, L_creditors_ring, L_creditors_complete, L_creditors_cp, L_creditors_cp_p, \
        L_creditors_contagion_ring, L_creditors_contagion_complete, L_creditors_contagion_cp, L_creditors_contagion_cp_p, L_creditors_direct_ring, \
        L_creditors_direct_complete, L_creditors_direct_cp, L_creditors_direct_cp_p, L_equityholders_ring, L_equityholders_complete, L_equityholders_cp, \
        L_equityholders_cp_p, L_equityholders_contagion_ring, L_equityholders_contagion_complete, L_equityholders_contagion_cp, L_equityholders_contagion_cp_p, \
        L_equityholders_direct_ring, L_equityholders_direct_complete, L_equityholders_direct_cp, L_equityholders_direct_cp_p)


### COMPARE SINGLE TRANCH TO TWO TRANCH ###

compare_results = False
if compare_results: 
    with open('Analysis contagion/Results_4_contagion_single_c_1000sim', 'rb') as f:
        sP_total_ring, sP_total_complete, sP_total_cp, sP_total_cp_p, sL_creditors_ring, sL_creditors_complete, sL_creditors_cp, sL_creditors_cp_p, \
        sL_creditors_contagion_ring, sL_creditors_contagion_complete, sL_creditors_contagion_cp, sL_creditors_contagion_cp_p, sL_creditors_direct_ring, \
        sL_creditors_direct_complete, sL_creditors_direct_cp, sL_creditors_direct_cp_p, sL_equityholders_ring, sL_equityholders_complete, sL_equityholders_cp, \
        sL_equityholders_cp_p, sL_equityholders_contagion_ring, sL_equityholders_contagion_complete, sL_equityholders_contagion_cp, sL_equityholders_contagion_cp_p, \
        sL_equityholders_direct_ring, sL_equityholders_direct_complete, sL_equityholders_direct_cp, sL_equityholders_direct_cp_p = pickle.load(f)

    with open('Analysis contagion/Results_4_contagion_multi_c_1000sim', 'rb') as f:
        mP_total_ring, mP_total_complete, mP_total_cp, mP_total_cp_p, mL_creditors_ring, mL_creditors_complete, mL_creditors_cp, mL_creditors_cp_p, \
        mL_creditors_contagion_ring, mL_creditors_contagion_complete, mL_creditors_contagion_cp, mL_creditors_contagion_cp_p, mL_creditors_direct_ring, \
        mL_creditors_direct_complete, mL_creditors_direct_cp, mL_creditors_direct_cp_p, mL_equityholders_ring, mL_equityholders_complete, mL_equityholders_cp, \
        mL_equityholders_cp_p, mL_equityholders_contagion_ring, mL_equityholders_contagion_complete, mL_equityholders_contagion_cp, mL_equityholders_contagion_cp_p, \
        mL_equityholders_direct_ring, mL_equityholders_direct_complete, mL_equityholders_direct_cp, mL_equityholders_direct_cp_p = pickle.load(f)

    diffL_creditor_ring = (mL_creditors_ring.squeeze() - sL_creditors_ring.mean(axis=1))
    diffL_creditor_complete = (mL_creditors_complete.squeeze() - sL_creditors_complete.mean(axis=1) ) 
    diffL_creditor_cp = (mL_creditors_cp.squeeze() - sL_creditors_cp.mean(axis=1))
    diffL_creditor_cp_p = (mL_creditors_cp_p.squeeze() - sL_creditors_cp_p.mean(axis=1)) 

    diffL_creditor_contagion_ring = (mL_creditors_contagion_ring.squeeze() - sL_creditors_contagion_ring.mean(axis=1)) 
    diffL_creditor_contagion_complete = (mL_creditors_contagion_complete.squeeze() - sL_creditors_contagion_complete.mean(axis=1) )
    diffL_creditor_contagion_cp = (mL_creditors_contagion_cp.squeeze() - sL_creditors_contagion_cp.mean(axis=1)) 
    diffL_creditor_contagion_cp_p = (mL_creditors_contagion_cp_p.squeeze() - sL_creditors_contagion_cp_p.mean(axis=1)) 

    diffL_equityholders_ring = (mL_equityholders_ring.squeeze() - sL_equityholders_ring.mean(axis=1)) 
    diffL_equityholders_complete = (mL_equityholders_complete.squeeze() - sL_equityholders_complete.mean(axis=1) ) 
    diffL_equityholders_cp = (mL_equityholders_cp.squeeze() - sL_equityholders_cp.mean(axis=1)) 
    diffL_equityholders_cp_p = (mL_equityholders_cp_p.squeeze() - sL_equityholders_cp_p.mean(axis=1)) 

    diffL_equityholders_contagion_ring = (mL_equityholders_contagion_ring.squeeze() - sL_equityholders_contagion_ring.mean(axis=1)) 
    diffL_equityholders_contagion_complete = (mL_equityholders_contagion_complete.squeeze() - sL_equityholders_contagion_complete.mean(axis=1) ) 
    diffL_equityholders_contagion_cp = (mL_equityholders_contagion_cp.squeeze() - sL_equityholders_contagion_cp.mean(axis=1)) 
    diffL_equityholders_contagion_cp_p = (mL_equityholders_contagion_cp_p.squeeze() - sL_equityholders_contagion_cp_p.mean(axis=1)) 

    diffL_creditor_direct_ring = (mL_creditors_direct_ring.squeeze() - sL_creditors_direct_ring.mean(axis=1)) 
    diffL_creditor_direct_complete = (mL_creditors_direct_complete.squeeze() - sL_creditors_direct_complete.mean(axis=1) )
    diffL_creditor_direct_cp = (mL_creditors_direct_cp.squeeze() - sL_creditors_direct_cp.mean(axis=1)) 
    diffL_creditor_direct_cp_p = (mL_creditors_direct_cp_p.squeeze() - sL_creditors_direct_cp_p.mean(axis=1)) 

    diffL_equityholders_direct_ring = (mL_equityholders_direct_ring.squeeze() - sL_equityholders_direct_ring.mean(axis=1)) 
    diffL_equityholders_direct_complete = (mL_equityholders_direct_complete.squeeze() - sL_equityholders_direct_complete.mean(axis=1) ) 
    diffL_equityholders_direct_cp = (mL_equityholders_direct_cp.squeeze() - sL_equityholders_direct_cp.mean(axis=1)) 
    diffL_equityholders_direct_cp_p = (mL_equityholders_direct_cp_p.squeeze() - sL_equityholders_direct_cp_p.mean(axis=1)) 

    save = True
    # Plot system level metrics
    plt.plot(m_range if effect == "m" else c_range, diffL_creditor_ring, linestyle=':')
    plt.plot(m_range if effect == "m" else c_range, diffL_creditor_complete, linestyle='-.')
    plt.plot(m_range if effect == "m" else c_range, diffL_creditor_cp, linestyle='--')
    plt.plot(m_range if effect == "m" else c_range, diffL_creditor_cp_p, linestyle='-')
    plt.legend(["Ring network", "Complete network", "Star network (core node)", "Star network (periphery node)"], fontsize='small')
    plt.xlabel("Conversion rate $m_1$" if effect == "m" else "Issued convertible debt $c_1$")
    plt.ylabel(r'$\Delta L_{\mathrm{Creditors}}$')
    plt.savefig('Images/Images comparison multi and single/comparison_L_creditor_m' if effect == "m" else 'Images/Images comparison multi and single/comparison_L_creditor_c' , bbox_inches="tight", dpi=300) if save else None
    plt.show()

    plt.plot(m_range if effect == "m" else c_range, diffL_creditor_contagion_ring, linestyle=':')
    plt.plot(m_range if effect == "m" else c_range, diffL_creditor_contagion_complete, linestyle='-.')
    plt.plot(m_range if effect == "m" else c_range, diffL_creditor_contagion_cp, linestyle='--')
    plt.plot(m_range if effect == "m" else c_range, diffL_creditor_contagion_cp_p, linestyle='-')
    plt.legend(["Ring network", "Complete network", "Star network (core node)", "Star network (periphery node)"], fontsize='small')
    plt.xlabel("Conversion rate $m_1$" if effect == "m" else "Issued convertible debt $c_1$")
    plt.ylabel(r'$\Delta L_{\mathrm{CreditorsC}}$')
    plt.savefig('Images/Images comparison multi and single/comparison_L_creditorcontagion_m' if effect == "m" else 'Images/Images comparison multi and single/comparison_L_creditorcontagion_c' , bbox_inches="tight", dpi=300) if save else None
    plt.show()

    plt.plot(m_range if effect == "m" else c_range, diffL_creditor_direct_ring, linestyle=':')
    plt.plot(m_range if effect == "m" else c_range, diffL_creditor_direct_complete, linestyle='-.')
    plt.plot(m_range if effect == "m" else c_range, diffL_creditor_direct_cp, linestyle='--')
    plt.plot(m_range if effect == "m" else c_range, diffL_creditor_direct_cp_p, linestyle='-')
    plt.legend(["Ring network", "Complete network", "Star network (core node)", "Star network (periphery node)"], fontsize='small') if save else None
    plt.xlabel("Conversion rate $m_1$" if effect == "m" else "Issued convertible debt $c_1$")
    plt.ylabel(r'$\Delta L_{\mathrm{CreditorsD}}$')
    plt.savefig('Images/Images comparison multi and single/comparison_L_creditordirect_m' if effect == "m" else 'Images/Images comparison multi and single/comparison_L_creditordirect_c' , bbox_inches="tight", dpi=300) if save else None
    plt.show()

    plt.plot(m_range if effect == "m" else c_range, diffL_equityholders_ring, linestyle=':')
    plt.plot(m_range if effect == "m" else c_range, diffL_equityholders_complete, linestyle='-.')
    plt.plot(m_range if effect == "m" else c_range, diffL_equityholders_cp, linestyle='--')
    plt.plot(m_range if effect == "m" else c_range, diffL_equityholders_cp_p, linestyle='-')
    plt.legend(["Ring network", "Complete network", "Star network (core node)", "Star network (periphery node)"], fontsize='small') if save else None
    plt.xlabel("Conversion rate $m_1$" if effect == "m" else "Issued convertible debt $c_1$")
    plt.ylabel(r'$\Delta L_{\mathrm{EQH}}$')
    plt.savefig('Images/Images comparison multi and single/comparison_L_eqholder_m' if effect == "m" else 'Images/Images comparison multi and single/comparison_L_eqholder_c' , bbox_inches="tight", dpi=300) if save else None
    plt.show()

    plt.plot(m_range if effect == "m" else c_range, diffL_equityholders_contagion_ring, linestyle=':')
    plt.plot(m_range if effect == "m" else c_range, diffL_equityholders_contagion_complete, linestyle='-.')
    plt.plot(m_range if effect == "m" else c_range, diffL_equityholders_contagion_cp, linestyle='--')
    plt.plot(m_range if effect == "m" else c_range, diffL_equityholders_contagion_cp_p, linestyle='-')
    plt.legend(["Ring network", "Complete network", "Star network (core node)", "Star network (periphery node)"], fontsize='small')
    plt.xlabel("Conversion rate $m_1$" if effect == "m" else "Issued convertible debt $c_1$")
    plt.ylabel(r'$\Delta L_{\mathrm{EQHC}}$')
    plt.savefig('Images/Images comparison multi and single/comparison_L_eqholdercontagion_m' if effect == "m" else 'Images/Images comparison multi and single/comparison_L_eqholdercontagion_c' , bbox_inches="tight", dpi=300) if save else None
    plt.show()

    plt.plot(m_range if effect == "m" else c_range, diffL_equityholders_direct_ring, linestyle=':')
    plt.plot(m_range if effect == "m" else c_range, diffL_equityholders_direct_complete, linestyle='-.')
    plt.plot(m_range if effect == "m" else c_range, diffL_equityholders_direct_cp, linestyle='--')
    plt.plot(m_range if effect == "m" else c_range, diffL_equityholders_direct_cp_p, linestyle='-')
    plt.legend(["Ring network", "Complete network", "Star network (core node)", "Star network (periphery node)"], fontsize='small')
    plt.xlabel("Conversion rate $m_1$" if effect == "m" else "Issued convertible debt $c_1$")
    plt.ylabel(r'$\Delta L_{\mathrm{EQHD}}$')
    plt.savefig('Images/Images comparison multi and single/comparison_L_eqholderdirect_m' if effect == "m" else 'Images/Images comparison multi and single/comparison_L_eqholderdirect_c' , bbox_inches="tight", dpi=300) if save else None
    plt.show()
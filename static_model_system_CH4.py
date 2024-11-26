import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from helper_functions import generate_2d_figure, generate_3d_figure

# Import generated W matrices from Csv-files generated in R
W_matrix_GV = pd.read_csv("W_matrix_GV.csv", index_col=0).to_numpy()
W_matrix_MD = pd.read_csv("W_matrix_MD.csv", index_col=0).to_numpy()
W_matrix_ME = pd.read_csv("W_matrix_ME.csv", index_col=0).to_numpy()

W_matrix_GV_normalized = W_matrix_GV / np.sum(W_matrix_GV, axis=0)
W_matrix_MD_normalized = W_matrix_MD / np.sum(W_matrix_MD, axis=0)
W_matrix_ME_normalized = W_matrix_ME / np.sum(W_matrix_ME, axis=0)

########################### Analysis of systemic risk (II) ####################################

# Set up the problem 
n = 4
core_banks = [0]
W_ring = generate_network(n, type="ring", fixed=True, weights_fixed = 0.75)
W_complete = generate_network(n, type="complete", fixed=True, weights_fixed = 0.75)
W_cp = generate_network(n, type="core periphery", fixed=True, weights_fixed = 0.75, core_nodes = core_banks)

# Settings for shocks
np.random.seed(0)   
nbSimulations = 100
shocked_banks = [i for i in range(n)]     
beta = np.array([2, 2, 2, 2])
rho = 0.0
a_shocked = 0
a_shocked_cp = 0

## Impact of m and c on PD, PC, PH, Lc, Lcc, Leq
m_range = np.arange(0.01, 5.01, 0.5)
c_range = np.arange(0.01, 50.01, 5)  
rho_range = np.arange(0, 1.0, 0.1)   
beta_range = np.arange(30, 80, 5)  

effect = "m"

def systemic_shock_analyis_contract_param(n, W_variants, core_banks, shocked_banks, a_shocked, a_shocked_cp, m_range, c_range, effect):
    P_total_ring = np.zeros((rho_range.shape[0], m_range.shape[0], 3, n))
    P_total_complete = np.zeros((rho_range.shape[0], m_range.shape[0], 3, n))
    P_total_cp = np.zeros((rho_range.shape[0], m_range.shape[0], 3, n))

    L_creditors_ring = np.zeros((rho_range.shape[0], m_range.shape[0] if effect == "m" else c_range.shape[0]))
    L_creditors_complete = np.zeros((rho_range.shape[0], m_range.shape[0] if effect == "m" else c_range.shape[0]))
    L_creditors_cp = np.zeros((rho_range.shape[0], m_range.shape[0] if effect == "m" else c_range.shape[0]))

    L_equityholders_ring = np.zeros((rho_range.shape[0], m_range.shape[0] if effect == "m" else c_range.shape[0]))
    L_equityholders_complete = np.zeros((rho_range.shape[0], m_range.shape[0] if effect == "m" else c_range.shape[0]))
    L_equityholders_cp = np.zeros((rho_range.shape[0], m_range.shape[0] if effect == "m" else c_range.shape[0]))

    st = time.time()
    et = time.time()
    for j, rho in enumerate(beta_range):
        atilde0 = rho
        d0 = 20
        X_shock = simulate_shocks_correlated(nbSimulations, beta, rho=0) 
        a_shocked = atilde0*(1 - X_shock) - d0
        a_shocked_cp = a_shocked
        a_shocked_cp[:, core_banks] = a_shocked_cp[:, core_banks] * (n - len(core_banks))
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

            # Compute initial prices
            _, _, _, s0_rc = compute_equilibrium(l_rc, c_rc, m, W_variants[0], np.array([20]*n))
            _, _, _, s0_cp = compute_equilibrium(l_rc, c_rc, m, W_variants[2], np.array([60, 20, 20, 20]))
            
            # Compute systemic risk metrics
            Prb_array_ring, L_creditors_array_ring, _, L_equityholders_array_ring, _ = compute_systemic_risk_metrics(l_rc, c_rc, m, W_variants[0], s0_rc, shocked_banks, a_shocked)
            Prb_array_cplt, L_creditors_array_cplt, _, L_equityholders_array_cplt, _  = compute_systemic_risk_metrics(l_rc, c_rc, m, W_variants[1], s0_rc, shocked_banks, a_shocked)
            Prb_array_cp, L_creditors_array_cp, _, L_equityholders_array_cp, _  = compute_systemic_risk_metrics(l_cp, c_cp, m, W_variants[2], s0_cp, shocked_banks, a_shocked_cp)

            # Store systemic risk metrics
            P_total_ring[j, k, :, :] = Prb_array_ring
            P_total_complete[j, k, :, :] = Prb_array_cplt
            P_total_cp[j, k, :, :] = Prb_array_cp

            L_creditors_ring[j,k] = L_creditors_array_ring.mean()
            L_creditors_complete[j,k] = L_creditors_array_cplt.mean()
            L_creditors_cp[j,k] = L_creditors_array_cp.mean()

            L_equityholders_ring[j,k] = L_equityholders_array_ring.mean()
            L_equityholders_complete[j,k] = L_equityholders_array_cplt.mean()
            L_equityholders_cp[j,k] = L_equityholders_array_cp.mean()
            
            et = time.time()

    return P_total_ring, P_total_complete, P_total_cp, L_creditors_ring, L_creditors_complete, L_creditors_cp, L_equityholders_ring, L_equityholders_complete, L_equityholders_cp

P_total_ring, P_total_complete, P_total_cp, L_creditors_ring, L_creditors_complete, L_creditors_cp, L_equityholders_ring, L_equityholders_complete, L_equityholders_cp = systemic_shock_analyis_contract_param(n, W_variants, core_banks, shocked_banks, a_shocked, a_shocked_cp, m_range, c_range, effect)

X, Y = np.meshgrid(m_range, beta_range)
Z = L_creditors_ring #P_total_ring.mean(axis=3)[:, :, 0]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
plt.show()

# Plot probability metrics
colors=['#6A1C1C', '#DAA520', '#006400', '#000080']

plt.plot(m_range if effect == "m" else c_range, P_total_ring.mean(axis=2)[:, 0], linestyle=':')
plt.plot(m_range if effect == "m" else c_range, P_total_complete.mean(axis=2)[:, 0], linestyle='-.')
plt.plot(m_range if effect == "m" else c_range, P_total_cp[:, 0, 0], linestyle='--')
plt.plot(m_range if effect == "m" else c_range, P_total_cp[:, :, 1:4].mean(axis=2)[:, 0], linestyle='--')
plt.legend(["Ring network", "Complete network", "Star network (c)", "Star network (p)"], fontsize='small')
plt.show()

plt.plot(m_range if effect == "m" else c_range, P_total_ring.mean(axis=2)[:, 1], linestyle=':')
plt.plot(m_range if effect == "m" else c_range, P_total_complete.mean(axis=2)[:, 1], linestyle='-.')
plt.plot(m_range if effect == "m" else c_range, P_total_cp[:, 1, 0], linestyle='--')
plt.plot(m_range if effect == "m" else c_range, P_total_cp[:, :, 1:4].mean(axis=2)[:, 1], linestyle='--')
plt.legend(["Ring network", "Complete network", "Star network (c)", "Star network (p)"], fontsize='small')
plt.show()

plt.plot(m_range if effect == "m" else c_range, P_total_ring.mean(axis=2)[:, 2], linestyle=':')
plt.plot(m_range if effect == "m" else c_range, P_total_complete.mean(axis=2)[:, 2], linestyle='-.')
plt.plot(m_range if effect == "m" else c_range, P_total_cp[:, 2, 0], linestyle='--')
plt.plot(m_range if effect == "m" else c_range, P_total_cp[:, :, 1:4].mean(axis=2)[:, 2], linestyle='--')
plt.legend(["Ring network", "Complete network", "Star network (c)", "Star network (p)"], fontsize='small')
plt.show()


# Plot settings
plt.rc('axes', labelsize=12)       
plt.rc('xtick', labelsize=10)       
plt.rc('ytick', labelsize=10)      
plt.rc('legend', fontsize='small')

# Plot system level metrics
save = False
plt.plot(m_range if effect == "m" else c_range, L_creditors_ring, linestyle=':')
plt.plot(m_range if effect == "m" else c_range, L_creditors_complete, linestyle='-.')
plt.plot(m_range if effect == "m" else c_range, L_creditors_cp, linestyle='--')
plt.legend(["Ring network", "Complete network", "Star network (core node)", "Star network (periphery node)"], fontsize='small')
plt.title("Systemic losses for creditors (total)")
plt.xlabel("Conversion rate $m$" if effect == "m" else "Issued convertible debt $c$")
plt.ylabel(r'$L_{\mathrm{creditors, total}}$')
plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/losscreditors', bbox_inches="tight", dpi=300) if save else None
plt.show()

plt.plot(m_range if effect == "m" else c_range, L_equityholders_ring, linestyle=':')
plt.plot(m_range if effect == "m" else c_range, L_equityholders_complete, linestyle='-.')
plt.plot(m_range if effect == "m" else c_range, L_equityholders_cp, linestyle='--')
plt.legend(["Ring network", "Complete network", "Star network (core node)", "Star network (periphery node)"], fontsize='small')
plt.title("Systemic losses for equityholders")
plt.xlabel("Conversion rate $m$" if effect == "m" else "Issued convertible debt $c$")
plt.ylabel(r'$L_{\mathrm{equityholders}}$')
plt.savefig('/Users/pauldemoor/Documents/MSc QFAS/MSc QFAS 2024-2025 thesis/Code/images/lossequityholder', bbox_inches="tight", dpi=300) if save else None
plt.show()
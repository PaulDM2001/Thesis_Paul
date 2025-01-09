import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

def BalanceSheet_scenario(C_0, A_IBr_0, A_IBc_0, A_IBe_0, A_ID_0, DEP_0, L_IBd_0, L_IBc_0, E_0, parameters_int_rate, parameters_spread, parameters_industry, parameters_cash, parameters_debt, parameters_deposits, W_matrix_ID, W_matrix_IB, W_matrix_CoCo, W_matrix_E, T, N_steps, shock_scenario=0):
    # Time grid
    dt = T / N_steps
    n_banks = C_0.shape[0]
    n_industries = parameters_industry["I_0"].shape[0]

    # Interest rate and Industry Indices
    credit_coefficients = {"Bad": 1.00, "Moderate": 0.57, "Good": 0.38}
    r_base = np.zeros((N_steps + 1))
    r_base[0] = parameters_int_rate["r_0"]
    spread = np.zeros((N_steps + 1))
    spread[0] = parameters_spread["s_0"]
    r_debt = np.zeros((n_banks, N_steps + 1))
    r_debt[:, 0] = r_base[0] + credit_coefficients["Good"] * spread[0]
    I_paths = np.zeros((n_industries, N_steps + 1))
    I_paths[:, 0] = parameters_industry["I_0"] 

    # Liability Side
    IB_debt = np.zeros((n_banks, N_steps + 1))
    IB_debt[:, 0] = L_IBd_0
    IB_CoCo_debt = np.zeros((n_banks, N_steps + 1))
    IB_CoCo_debt[:, 0] = L_IBc_0
    Deposit_paths = np.zeros((n_banks, N_steps + 1))
    Deposit_paths[:, 0] = DEP_0
    Equity_paths = np.zeros((n_banks, N_steps + 1))
    Equity_paths[:, 0] = E_0

    # Asset side
    Cash_paths = np.zeros((n_banks, N_steps + 1))
    Cash_paths[:, 0] = C_0
    A_IB_paths = np.zeros((n_banks, N_steps + 1))
    A_IB_paths[:, 0] = A_IBr_0
    A_IBC_paths = np.zeros((n_banks, N_steps + 1))
    A_IBC_paths[:, 0] = A_IBc_0
    A_ID_paths = np.zeros((n_banks, N_steps + 1))
    A_ID_paths[:, 0] = A_ID_0
    A_EQC_paths = np.zeros((n_banks, N_steps + 1))
    A_EQC_paths[:, 0] = A_IBe_0

    TA_paths = np.zeros((n_banks, N_steps + 1))
    TA_paths[:, 0] = C_0 + A_IBr_0 + A_IBc_0 + A_ID_0 + A_IBe_0

    # Credit ratings
    Z_paths = np.zeros((n_banks, N_steps + 1))
    Z_paths[:, 0] = 1.2 * Cash_paths[:, 0] / TA_paths[:, 0] + 0.6 * Equity_paths[:, 0] / (Deposit_paths[:, 0] + IB_debt[:, 0])
    Rating = np.zeros((n_banks, N_steps + 1))
    Rating[:, 0] = 2

    # CoCo trigger measure 
    ETA_ratio = np.zeros((n_banks, N_steps + 1))
    ETA_ratio[:, 0] = Equity_paths[:, 0] / TA_paths[:, 0]

    # Set of Defaulted, Converted, Healthy banks
    Defaulted_Banks = []
    Converted_Banks = []
    Healthy_banks = [i for i in range(n_banks)] 

    ######## Simulate a path ########
    for t in range(1, N_steps+1):
       
        ################# Interest rate and spread #################
        Z1 = np.random.randn()
        r_base[t] = r_base[t-1] + parameters_int_rate["alpha_r"] * (parameters_int_rate["r_bar"]-r_base[t-1]) * dt + parameters_int_rate["sigma_r"] * np.sqrt(dt) * np.sqrt(max(0, r_base[t-1])) * Z1  

        Z2 = np.random.randn()
        spread[t] = spread[t-1] + parameters_spread["alpha_s"] * (parameters_spread["s_bar"]-spread[t-1]) * dt + parameters_spread["sigma_s"] * np.sqrt(dt) * np.sqrt(max(0, spread[t-1])) * Z2 

        # Credit ratings
        if t in [63, 126, 189]:
            if t == 63:
                benchmark = Z_paths[:, 0]
            moving_average_Z = np.mean(Z_paths[:, (t-63):t], axis=1)
            Rating[:, t] = (moving_average_Z >= 0.90 * benchmark) * (moving_average_Z <= 1.10 * benchmark) * Rating[:, t-1] + (moving_average_Z < 0.90 * benchmark) * (np.maximum(Rating[:, t-1] - 1, 0)) + (moving_average_Z > 1.10 * benchmark) * (np.minimum(Rating[:, t-1] + 1, 2))
            if np.any(Rating[:, t] != Rating[:, t-1]):
                benchmark = moving_average_Z * (Rating[:, t] != Rating[:, t-1]) + benchmark * (Rating[:, t] == Rating[:, t-1])

        else:
            Rating[:, t] = Rating[:, t-1]
        
        r_debt[:, t] = r_base[t] + ((Rating[:, t]==2)*credit_coefficients["Good"] + (Rating[:, t]==1)*credit_coefficients["Moderate"] + (Rating[:, t]==0)*credit_coefficients["Bad"]) * spread[t] 
        
        
        ################# Interbank debt and CoCo debt #################
        IB_debt[:, t] = IB_debt[:, t-1] * (1 - parameters_debt["duration"] * (r_debt[:, t] - r_debt[:, t-1]) + (1/2) * parameters_debt["convexity"] * (r_debt[:, t] - r_debt[:, t-1]) ** 2)
        IB_debt[Defaulted_Banks, t] = 0 
        
        IB_CoCo_debt[:, t] = IB_CoCo_debt[:, t-1] * (1 - parameters_debt["duration"] * (r_debt[:, t] - r_debt[:, t-1]) + (1/2) * parameters_debt["convexity"] * (r_debt[:, t] - r_debt[:, t-1]) ** 2)
        IB_CoCo_debt[Defaulted_Banks, t] = 0

        
        ################# Cash Account #################
        
        # Shocks to Cash Account
        mu_mvn = np.zeros(n_banks)
        corr_matrix = np.ones((n_banks, n_banks)) * parameters_cash["shock_C_corr"] + np.identity(n_banks) * (1 - parameters_cash["shock_C_corr"])
        samples_gaussian_copula = stats.norm.cdf(np.random.multivariate_normal(mu_mvn, corr_matrix))
        dN = stats.poisson.ppf(samples_gaussian_copula, mu=parameters_cash["lambda_C"] * dt)
        
        if dN.any() > 0:
            gaussian_shock = np.random.normal(loc = parameters_cash["shock_C_mu"], scale = parameters_cash["shock_C_sigma"], size=n_banks)
            dJ = gaussian_shock * dN
        else:
            dJ = 0
        
        if shock_scenario == 3:
            dJ -= 0.15 * np.random.binomial(1, 0.5, size=n_banks) if t == 20 else 0
   
        # Interest on Cash Account
        Cash_paths[:, t] = Cash_paths[:, t-1] * (1 + r_base[t]/100 * dt + dJ)
        Cash_paths[Defaulted_Banks, t] = 0

        
        ################# Industry Debt Assets #################
        
        # Shock to Industry Indices
        mu_mvn_I = np.zeros(n_industries)
        corr_matrix_I = np.ones((n_industries, n_industries)) * parameters_industry["shock_I_corr"] + np.identity(n_industries) * (1 - parameters_industry["shock_I_corr"])
        samples_gaussian_copula_I = stats.norm.cdf(np.random.multivariate_normal(mu_mvn_I, corr_matrix_I))
        dN_I = stats.poisson.ppf(samples_gaussian_copula_I, mu=parameters_industry["lambda_I"] * dt)

        if dN_I.any() > 0:
            gaussian_shock_I = np.random.normal(loc = parameters_industry["shock_I_mu"], scale = parameters_industry["shock_I_sigma"], size=n_industries)
            dJ_I = gaussian_shock_I * dN_I
        else:
            dJ_I = 0
        
        if shock_scenario == 1:
            dJ_I -= 0.10 * np.random.binomial(1, 0.5, size=n_industries) if t == 20 else 0
        elif shock_scenario == 2:
            dJ_I -= 0.10 * np.random.binomial(1, 0.5, size=n_industries) if t == 20 else 0
            dJ_I -= 0.10 * np.random.binomial(1, 0.5, size=n_industries) if t == 200 else 0
        elif shock_scenario == 3:
            dJ_I -= 0.10 * np.random.binomial(1, 0.5, size=n_industries) if t == 20 else 0

        # Difussion 
        dZ = np.random.multivariate_normal(mean=[0]*n_industries, cov=parameters_industry["cov_I"])
        dW = np.sqrt(dt) * dZ
        # Evolution
        I_paths[:, t] = I_paths[:, t-1] * (1 + parameters_industry["sigma_I"] * dW + dJ_I) + parameters_industry["alpha_I"] * (parameters_industry["I_bar"] - I_paths[:, t-1]) * dt  
        A_ID_paths[:, t] = np.matmul(I_paths[:, t], W_matrix_ID) * A_ID_paths[:, 0] 
        A_ID_paths[Defaulted_Banks, t] = 0


        ################# Interbank Assets #################
        A_IB_paths[:, t] = parameters_debt["factor_IB"] * np.matmul(W_matrix_IB, IB_debt[:, t])
        A_IB_paths[Defaulted_Banks, t] = 0
        A_IBC_paths[:, t] = np.matmul(W_matrix_CoCo, IB_CoCo_debt[:, t])
        A_IBC_paths[Defaulted_Banks, t] = 0
        A_EQC_paths[:, t] = np.matmul(W_matrix_E, Equity_paths[:, t-1])

        
        ################# Total Assets and Deposits #################         
        TA_paths[:, t] = Cash_paths[:, t] + A_ID_paths[:, t] + A_IB_paths[:, t] + A_IBC_paths[:, t] + A_EQC_paths[:, t] 

        Deposit_paths[(Healthy_banks + Converted_Banks), t] = Deposit_paths[(Healthy_banks + Converted_Banks), t-1] * (1 + parameters_deposits["g_D"] * (parameters_deposits["target_AD"] - TA_paths[(Healthy_banks + Converted_Banks), t-1]/Deposit_paths[(Healthy_banks + Converted_Banks), t-1]) * dt + np.sqrt(dt) * np.random.randn(len((Healthy_banks + Converted_Banks))) * 0.00)
        Deposit_paths[Defaulted_Banks, t] = 0


        ################# Equity #################
        Equity_paths[:, t] = TA_paths[:, t] - Deposit_paths[:, t] - IB_debt[:, t] - IB_CoCo_debt[:, t]
        
        ################# Conversion and Default #################
        conversions = False
        if conversions: 
            # Conversions
            ETA_ratio[Healthy_banks, t] = Equity_paths[Healthy_banks, t] / TA_paths[Healthy_banks, t] 
            ETA_ratio[Converted_Banks, t] = Equity_paths[Converted_Banks, t] / TA_paths[Converted_Banks, t] 
            Converting_t = [i for i in range(n_banks) if ETA_ratio[i, t] <= 0.04 and i not in Converted_Banks]
            Converted_Banks.extend(Converting_t)

            # Canceling of CoCo debt:
            if len(Converting_t) > 0:
                q_factor = 1
                for i in Converting_t:
                    W_matrix_E[:, i] = W_matrix_CoCo[i, :] * IB_CoCo_debt[i, t] * q_factor / (Equity_paths[i, t] + IB_CoCo_debt[i, t])
                IB_CoCo_debt[Converting_t, t] = 0
                A_EQC_paths[:, t] = np.matmul(W_matrix_E, Equity_paths[:, t])
                TA_paths[:, t] = Cash_paths[:, t] + A_ID_paths[:, t] + A_IB_paths[:, t] + A_IBC_paths[:, t] + A_EQC_paths[:, t] 
                Equity_paths[:, t] = TA_paths[:, t] - Deposit_paths[:, t] - IB_debt[:, t] - IB_CoCo_debt[:, t]
                
        # Defaults
        Default_t = [i for i in range(n_banks) if Equity_paths[i, t] < 0 and i not in Defaulted_Banks]
        Defaulted_Banks.extend(Default_t)
        
        # Healthy
        Healthy_banks = [i for i in range(n_banks) if i not in (Defaulted_Banks + Converted_Banks)]
        
        # Recovery of credit given default
        if len(Default_t) > 0:
            R_rate = 0.6
            Cash_paths[Healthy_banks, t] += np.sum(np.matmul(W_matrix_IB[Default_t].T, IB_debt[Default_t, t][:, np.newaxis]), axis=1)[Healthy_banks] * R_rate
            IB_debt[Default_t, t] = 0
            A_IB_paths[:, t] = parameters_debt["factor_IB"] * np.matmul(W_matrix_IB, IB_debt[:, t])

            Equity_paths[Default_t, t] = 0
        
        ################# Z-score #################
        Z_paths[(Healthy_banks + Converted_Banks), t] = 1.2 * Cash_paths[(Healthy_banks + Converted_Banks), t] / TA_paths[(Healthy_banks + Converted_Banks), t] + 0.6 * Equity_paths[(Healthy_banks + Converted_Banks), t] / (Deposit_paths[(Healthy_banks + Converted_Banks), t] + IB_debt[(Healthy_banks + Converted_Banks), t])
        Z_paths[Defaulted_Banks, t] = 0
    
    # plt.clf()
    # for i in range(0, 35):
    #     plt.plot(Equity_paths[i, :])

    # for i in range(0, 35):
    #     plt.plot(Rating[i, :])

    # for i in range(35, 40):
    #     plt.plot(Equity_paths[i, :])

    return Defaulted_Banks, Equity_paths

### Model parameters
np.random.seed(0)
## General parametrs
nb_steps = 252                  # Time steps
T = 1                           # Time horizon
n_l = 5                         # Large banks
n_s = 35                        # Small banks
n_banks = n_l + n_s             # Total number of banks
n_industries = 10                # Industrial sectors 

## Parameters Interest Rate and Spread
param_dict_base_rate = {"r_0" : 1.50, "alpha_r": 10.62, "r_bar": 1.47, "sigma_r": 0.13}
param_dict_spread = {"s_0" : 3.43, "alpha_s": 13.01, "s_bar": 3.43, "sigma_s": 0.14}

## Parameters Industry Loan Assets
cov_I = np.ones((n_industries, n_industries)) * 0.0
np.fill_diagonal(cov_I, 1)
param_dict_industry_debt = {"I_0": np.ones(n_industries) * 1, "alpha_I": np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), "I_bar" : np.array([0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35]), "sigma_I":np.array([0.02, 0.02, 0.05, 0.05, 0.08, 0.08, 0.11, 0.11, 0.14, 0.14]),
                            "lambda_I": 0.04, "shock_I_mu": -0.05, "shock_I_sigma": 0.05, "shock_I_corr": 0.10,  "cov_I": cov_I}

## Parameters Cash
param_dict_cash = {"lambda_C": 0.04, "shock_C_mu": -0.05, "shock_C_sigma": 0.05, "shock_C_corr": 0.10}

## Parametrs Interbank Debt Liabilities
param_dict_interbank_debt = {"duration": 1.5, "convexity": 1.5**2, "duration_CoCo": 1.5, "convexity_CoCo": 1.5**2, "factor_IB": 0.7}

## Parameters Deposits
param_dict_deposits = {"g_D": 0.5, "target_AD": 1.25}


## Parameters Initial Balance Sheets
# Small banks
C_0s = np.ones(n_s) * 25
A_IBr_0s = np.ones(n_s) * 7
A_IBc_0s = np.ones(n_s) * 0
A_IBe_0s = np.ones(n_s) * 0
A_ID_0s = np.ones(n_s) * 68

DEP_0s = np.ones(n_s) * 80
L_IBd_0s = np.ones(n_s) * 10
L_IBc_0s = np.ones(n_s) * 0
E_0s = np.ones(n_s) * 10

# Large banks
mult_large = 20 
C_0l = np.ones(n_l) * 30 * mult_large 
A_IBr_0l = np.ones(n_l) * 7 * mult_large 
A_IBc_0l = np.ones(n_l) * 0 * mult_large 
A_IBe_0l = np.ones(n_l) * 0  
A_ID_0l = np.ones(n_l) * 63 * mult_large 

DEP_0l = np.ones(n_l) * 80 * mult_large 
L_IBd_0l = np.ones(n_l) * 10 * mult_large 
L_IBc_0l = np.ones(n_l) * 0 * mult_large 
E_0l = np.ones(n_l) * 10 * mult_large 

# Merged
C_0 = np.hstack((C_0s, C_0l))
A_IBr_0 = np.hstack((A_IBr_0s, A_IBr_0l))
A_IBc_0 = np.hstack((A_IBc_0s, A_IBc_0l))
A_IBe_0 = np.hstack((A_IBe_0s, A_IBe_0l))
A_ID_0 = np.hstack((A_ID_0s, A_ID_0l))

DEP_0 = np.hstack((DEP_0s, DEP_0l))
L_IBd_0 = np.hstack((L_IBd_0s, L_IBd_0l))
L_IBc_0 = np.hstack((L_IBc_0s, L_IBc_0l))
E_0 = np.hstack((E_0s, E_0l))
 
# Interbank network
W_matrix_IBd = pd.read_csv('W_matrix_CH5_ME.csv', index_col=0).to_numpy()
W_matrix_IBd_norm = W_matrix_IBd / np.sum(W_matrix_IBd, axis=0)

# Industru debt holdings
W_matrix_ID = np.zeros((n_industries, n_banks))
for j in range(n_banks):
    if j < 35: 
        k = 3
        nonzero_rows = np.random.choice(n_industries, k, replace=False)
    else:
        k = n_industries
        nonzero_rows = np.random.choice(n_industries, k, replace=False)
    
    W_matrix_ID[nonzero_rows, j] = 1/k

## Simulation analysis
np.random.seed(0)
# Parameters
nb_Simulations = 1000
shock_scenario = 0                                      #scenarios 0 (no shocks), 1 (industrial shock day 20), 2 (industrial shock day 20 and 200), 3 (cash shock and industrial shock day 20)
bankruptcy_array_small = np.zeros((nb_Simulations))
bankruptcy_array_large = np.zeros((nb_Simulations))
equity_array_small = np.zeros((nb_steps+1, nb_Simulations))
equity_array_large = np.zeros((nb_steps+1, nb_Simulations))

for i in range(nb_Simulations):
    if i % 10 == 0:
        print(f"Iteration {i}/{nb_Simulations}")
    # Run one simulation
    Bankrupt_banks, Equity_paths = BalanceSheet_scenario(C_0, A_IBr_0, A_IBc_0, A_IBe_0, A_ID_0, DEP_0, L_IBd_0, L_IBc_0, E_0, parameters_int_rate = param_dict_base_rate, parameters_spread = param_dict_spread, parameters_industry = param_dict_industry_debt, 
                                    parameters_cash = param_dict_cash, parameters_debt = param_dict_interbank_debt, parameters_deposits = param_dict_deposits, W_matrix_ID = W_matrix_ID, W_matrix_IB = W_matrix_IBd_norm, W_matrix_CoCo = W_matrix_IBd_norm, 
                                    W_matrix_E = np.zeros((n_banks, n_banks)), T = T, N_steps = nb_steps, shock_scenario = shock_scenario)
    
    # Compute and save metrics
    nb_Bankrupt_small = len([i for i in range(n_s) if (i in Bankrupt_banks and i<35)])
    nb_Bankrupt_large = len([i for i in range(n_l) if (i in Bankrupt_banks and i>=35)])
    small_banks_Equity = np.sum(Equity_paths[:35], axis=0)
    large_banks_Equity = np.sum(Equity_paths[35:], axis=0)

    bankruptcy_array_small[i] = nb_Bankrupt_small
    bankruptcy_array_large[i] = nb_Bankrupt_large
    equity_array_small[:, i] = small_banks_Equity 
    equity_array_large[:, i] = large_banks_Equity

# Compute VaR
alpha = 0.05
VaR_equity_small_banks = np.quantile(equity_array_small, q=alpha, axis=1) / equity_array_small[0, 0]
VaR_equity_large_banks = np.quantile(equity_array_large, q=alpha, axis=1) / equity_array_large[0, 0]


print(f"Average number of bankruptcies small banks: {bankruptcy_array_small.mean()}")
print(f"Average number of bankruptcies large banks: {bankruptcy_array_large.mean()}")

plt.plot(VaR_equity_small_banks)
plt.show()
plt.plot(VaR_equity_large_banks)
plt.show()
print(f"Equity VaR small banks period T: {VaR_equity_small_banks[-1]}")
print(f"Equity VaR large banks period T: {VaR_equity_large_banks[-1]}")
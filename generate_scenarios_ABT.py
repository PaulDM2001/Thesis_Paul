import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import pickle
import time 

def BalanceSheet_scenarios(parameters_int_rate, parameters_spread, parameters_industry, parameters_cash, n_banks, n_l, T, N_steps, nbSimulations, shock_scenario):
    
    # Initialize RNG
    rng = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)

    n_industries = parameters_industry["I_0"].shape[0]

    r_base_array = np.zeros((nbSimulations, N_steps + 1))
    spread_array = np.zeros((nbSimulations, 4, N_steps + 1))
    I_array = np.zeros((nbSimulations, n_industries, N_steps + 1))
    Cash_shock_array = np.zeros((nbSimulations, n_banks, N_steps + 1))

    for i in range(nbSimulations):
        if i % 100 == 0:
            print(f"Iteration {i}/{nb_Simulations}")

        # Time grid
        dt = T / N_steps

        # Scenarios 
        r_base = np.zeros((N_steps + 1))
        r_base[0] = parameters_int_rate["r_0"]
        spread = np.zeros((4, N_steps + 1))
        spread[:, 0] = parameters_spread["s_0"]
        I_paths = np.zeros((n_industries, N_steps + 1))
        I_paths[:, 0] = parameters_industry["I_0"] 
        Cash_shock = np.zeros((n_banks, N_steps + 1))
        Cash_shock[:, 0] = 0

        ######## Simulate a path ########
        for t in range(1, N_steps+1):
        
            ################# Interest rate and spread #################
            r_base[t] = r_base[t-1] + parameters_int_rate["alpha_r"] * (parameters_int_rate["r_bar"]-r_base[t-1]) * dt + parameters_int_rate["sigma_r"] * np.sqrt(dt) * np.sqrt(max(0, r_base[t-1])) * rng.normal()
            spread[:, t] = spread[:, t-1] + parameters_spread["alpha_s"] * (parameters_spread["s_bar"]-spread[:, t-1]) * dt + parameters_spread["sigma_s"] * np.sqrt(dt) * np.sqrt(np.maximum(0, spread[:, t-1])) * rng.normal(size = 4)
            
            ################# Cash Account #################
            
            # Shocks to Cash Account
            mu_mvn = np.zeros(n_banks)
            corr_matrix = np.ones((n_banks, n_banks)) * parameters_cash["shock_C_corr"] + np.identity(n_banks) * (1 - parameters_cash["shock_C_corr"])
            samples_gaussian_copula = stats.norm.cdf(rng.multivariate_normal(mu_mvn, corr_matrix))
            dN_cash = stats.poisson.ppf(samples_gaussian_copula, mu=parameters_cash["lambda_C"] * dt)
            
            dJ_cash = np.zeros(n_banks)
            if dN_cash.any() > 0:
                gaussian_shock = rng.normal(loc = parameters_cash["shock_C_mu"], scale = parameters_cash["shock_C_sigma"], size=n_banks)
                dJ_cash += gaussian_shock * dN_cash
            
            if t == 20 and shock_scenario == 3:
                selected_banks = rng2.binomial(1, 1/2, size=n_l)          # rng2 to prevent issues with internal state of RNG
                dJ_cash[35:] -= 0.15 * selected_banks 
    
            # Interest on Cash Account
            Cash_shock[:, t] = dJ_cash
            
            ################# Industry Debt Assets #################
            
            # Shock to Industry Indices
            mu_mvn_I = np.zeros(n_industries)
            corr_matrix_I = np.ones((n_industries, n_industries)) * parameters_industry["shock_I_corr"] + np.identity(n_industries) * (1 - parameters_industry["shock_I_corr"])
            samples_gaussian_copula_I = stats.norm.cdf(rng.multivariate_normal(mu_mvn_I, corr_matrix_I))
            dN_I = stats.poisson.ppf(samples_gaussian_copula_I, mu=parameters_industry["lambda_I"] * dt)

            dJ_I = np.zeros(n_industries)
            if dN_I.any() > 0:
                gaussian_shock_I = rng.normal(loc = parameters_industry["shock_I_mu"], scale = parameters_industry["shock_I_sigma"], size=n_industries)
                dJ_I = gaussian_shock_I * dN_I
            
            if t == 20:
                selected_industries = rng.binomial(n=1, p=1/3, size=n_industries) 
                if shock_scenario in [1, 2, 3]:
                    dJ_I -= 0.10 * selected_industries
    
            if t == 200:
                selected_industries = rng.binomial(n=1, p=1/3, size=n_industries) 
                if shock_scenario in [2]:
                    dJ_I -= 0.10 * selected_industries

            # Difussion 
            dZ = rng.multivariate_normal(mean=[0]*n_industries, cov=parameters_industry["cov_I"])
            dW = np.sqrt(dt) * dZ

            # Evolution
            I_paths[:, t] = I_paths[:, t-1] * (1 + parameters_industry["sigma_I"] * dW + dJ_I) + parameters_industry["alpha_I"] * (parameters_industry["I_bar"] - I_paths[:, t-1]) * dt  

        r_base_array[i, :] = r_base
        spread_array[i, :, :] = spread
        I_array[i, :, :] = I_paths
        Cash_shock_array[i, :, :] = Cash_shock

    return r_base_array, spread_array, I_array, Cash_shock_array


def Compute_evolution_system(C_0, A_IBr_0, A_IBc_0, A_IBe_0, A_ID_0, DEP_0, L_IBd_0, L_IBc_0, E_0, scen_r_base, scen_spread, scen_I, scen_C, parameters_debt, parameters_deposits, W_matrix_ID, W_matrix_IB, W_matrix_CoCo, CoCo_thresholds, W_matrix_E, T, N_steps, nbSimulations):
    st = time.time()
    bankruptcy_array_small = np.zeros((nbSimulations))
    bankruptcy_array_large = np.zeros((nbSimulations))
    conversion_array_small = np.zeros((nbSimulations))
    conversion_array_large = np.zeros((nbSimulations))
    status_array_small = np.zeros((3, N_steps+1, nbSimulations))
    status_array_large = np.zeros((3, N_steps+1, nbSimulations))
    equity_array_small = np.zeros((N_steps+1, nbSimulations))
    equity_array_large = np.zeros((N_steps+1, nbSimulations))

    for idx in range(nbSimulations):
        if idx % 100 == 0:
            ct = time.time()
            print(f"Iteration {idx}/{nbSimulations}, Elapsed time: {ct - st:.2f} s")

        # Time grid
        dt = T / N_steps
        n_banks = C_0.shape[0]

        # Interest rate and Industry Indices
        r_debt = np.zeros((n_banks, N_steps + 1))
        r_debt[:, 0] = scen_r_base[idx, 0] + scen_spread[idx, 2, 0]

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
        W_matrix_E = np.zeros((n_banks, n_banks))

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
        Status_over_time = np.zeros((n_banks, N_steps + 1))
        Defaulted_Banks = []
        Converted_Banks = []
        Healthy_banks = [i for i in range(n_banks)] 

        ######## Simulate a path ########
        for t in range(1, N_steps+1):
        
            # Credit ratings
            if t in [63, 126, 189]:
                if t == 63:
                    benchmark = Z_paths[:, 0]
                moving_average_Z = np.mean(Z_paths[:, (t-63):t], axis=1)
                Rating[:, t] = (moving_average_Z >= 0.90 * benchmark) * (moving_average_Z <= 1.10 * benchmark) * Rating[:, t-1] + (moving_average_Z < 0.90 * benchmark) * (np.maximum(Rating[:, t-1] - 1, 0)) + (moving_average_Z > 1.10 * benchmark) * (np.minimum(Rating[:, t-1] + 1, 3))
                if np.any(Rating[:, t] != Rating[:, t-1]):
                    benchmark = moving_average_Z * (Rating[:, t] != Rating[:, t-1]) + benchmark * (Rating[:, t] == Rating[:, t-1])

            else:
                Rating[:, t] = Rating[:, t-1]
            
            r_debt[:, t] = scen_r_base[idx, t] + (Rating[:, t]==3)*scen_spread[idx, 3, t] + (Rating[:, t]==2)*scen_spread[idx, 2, t] + (Rating[:, t]==1)*scen_spread[idx, 1, t] + (Rating[:, t]==0)*scen_spread[idx, 0, t]
            
            
            ################# Interbank debt and CoCo debt #################
            IB_debt[:, t] = IB_debt[:, t-1] * (1 - parameters_debt["duration"] * (r_debt[:, t] - r_debt[:, t-1]) + (1/2) * parameters_debt["convexity"] * (r_debt[:, t] - r_debt[:, t-1]) ** 2)
            IB_debt[Defaulted_Banks, t] = 0 
            
            IB_CoCo_debt[:, t] = IB_CoCo_debt[:, t-1] * (1 - parameters_debt["duration"] * (r_debt[:, t] - r_debt[:, t-1]) + (1/2) * parameters_debt["convexity"] * (r_debt[:, t] - r_debt[:, t-1]) ** 2)
            IB_CoCo_debt[Defaulted_Banks, t] = 0

            
            ################# Cash Account #################
            Cash_paths[:, t] = Cash_paths[:, t-1] * (1 + scen_r_base[idx, t] * dt + scen_C[idx, :, t])
            Cash_paths[Defaulted_Banks, t] = 0

            ################# Industry Debt Assets #################
            A_ID_paths[:, t] = np.matmul(scen_I[idx, :, t], W_matrix_ID) * A_ID_paths[:, 0] 
            A_ID_paths[Defaulted_Banks, t] = 0


            ################# Interbank Assets #################
            A_IB_paths[:, t] = parameters_debt["factor_IB"] * np.matmul(W_matrix_IB, IB_debt[:, t])
            A_IB_paths[Defaulted_Banks, t] = 0
            if np.any(A_IBC_paths[:, t-1]>0):
                A_IBC_paths[:, t] = np.matmul(W_matrix_CoCo, IB_CoCo_debt[:, t])
                A_IBC_paths[Defaulted_Banks, t] = 0
                A_EQC_paths[:, t] = np.matmul(W_matrix_E, Equity_paths[:, t-1])

        
            ################# Total Assets #################         
            TA_paths[:, t] = Cash_paths[:, t] + A_ID_paths[:, t] + A_IB_paths[:, t] + A_IBC_paths[:, t] + A_EQC_paths[:, t] 


            ################# Deposits #################     
            Deposit_paths[(Healthy_banks + Converted_Banks), t] = Deposit_paths[(Healthy_banks + Converted_Banks), t-1] * (1 + parameters_deposits["g_D"] * (parameters_deposits["target_AD"] - TA_paths[(Healthy_banks + Converted_Banks), t-1]/Deposit_paths[(Healthy_banks + Converted_Banks), t-1]) * dt)
            Deposit_paths[Defaulted_Banks, t] = 0


            ################# Equity #################
            Equity_paths[:, t] = TA_paths[:, t] - Deposit_paths[:, t] - IB_debt[:, t] - IB_CoCo_debt[:, t]
            

            ################# Conversions #################
            if np.any(L_IBc_0 > 0): 
                # Conversions
                ETA_ratio[(Healthy_banks + Converted_Banks), t] = Equity_paths[(Healthy_banks + Converted_Banks), t] / TA_paths[(Healthy_banks + Converted_Banks), t] 

                Converting_structural = [i for i in range(n_banks) if ETA_ratio[i, t] <= CoCo_thresholds[i] and i not in Converted_Banks and i not in Defaulted_Banks]
                Converted_Banks.extend(Converting_structural)
                Healthy_banks = [i for i in range(n_banks) if i not in (Defaulted_Banks + Converted_Banks)]
                
                Converting_t = Converting_structural
                while len(Converting_t) > 0:    # Keep converting until no more conversions are possible
                    
                    # Compute weights of equity for creditors (only used when there is interbank CoCo debt)
                    for k in Converting_t:
                        q_max = (Equity_paths[k, t] + IB_CoCo_debt[k, t]) / IB_CoCo_debt[k, t]
                        W_matrix_E[:, k] = W_matrix_CoCo[:, k] * IB_CoCo_debt[k, t] * np.minimum(parameters_debt["CoCo_q"], q_max) / (Equity_paths[k, t] + IB_CoCo_debt[k, t])
                    
                    # Cancel CoCo liabilities
                    IB_CoCo_debt[Converting_t, t] = 0
                    Equity_paths[Converting_t, t] = TA_paths[Converting_t, t] - Deposit_paths[Converting_t, t] - IB_debt[Converting_t, t] - IB_CoCo_debt[Converting_t, t]
                    
                    # Cross-holding
                    if np.any(A_IBC_paths[:, 0]>0):
                        A_IBC_paths[:, t] = np.matmul(W_matrix_CoCo, IB_CoCo_debt[:, t])
                        A_EQC_paths[:, t] = np.matmul(W_matrix_E, Equity_paths[:, t])
                        TA_paths[:, t] = Cash_paths[:, t] + A_ID_paths[:, t] + A_IB_paths[:, t] + A_IBC_paths[:, t] + A_EQC_paths[:, t] 
                        Equity_paths[:, t] = TA_paths[:, t] - Deposit_paths[:, t] - IB_debt[:, t] - IB_CoCo_debt[:, t]
                    
                    # Check for further conversions
                    ETA_ratio[(Healthy_banks + Converted_Banks), t] = Equity_paths[(Healthy_banks + Converted_Banks), t] / TA_paths[(Healthy_banks + Converted_Banks), t]
                    Converting_t = [i for i in range(n_banks) if ETA_ratio[i, t] <= CoCo_thresholds[i] and i not in Converted_Banks and i not in Defaulted_Banks]
                    Converted_Banks.extend(Converting_t)
                    Healthy_banks = [i for i in range(n_banks) if i not in (Defaulted_Banks + Converted_Banks)]

            ################# Defaults #################
            Default_structural = [i for i in range(n_banks) if Equity_paths[i, t] < 0 and i not in Defaulted_Banks]
            Defaulted_Banks.extend(Default_structural)
            Converted_Banks = list(set(Converted_Banks) - set(Defaulted_Banks))
            Healthy_banks = [i for i in range(n_banks) if i not in (Defaulted_Banks + Converted_Banks)]
            
            # Defaults
            Default_t = Default_structural
            while len(Default_t) > 0:   # Keep going until no banks is in default
                # Recovery of assets from defaulted banks 
                for k in Default_t:
                    recovered_debt_assets = np.minimum(np.maximum(TA_paths[k, t] - Deposit_paths[k, t], 0), IB_debt[k, t]) * parameters_debt["Recovery_rate"]
                    Cash_paths[(Healthy_banks + Converted_Banks), t] += parameters_debt["factor_IB"] * W_matrix_IB[(Healthy_banks + Converted_Banks), k] * recovered_debt_assets
                    IB_debt[k, t] = 0
                    Equity_paths[k, t] = 0
                
                # Update other banks 
                A_IB_paths[:, t] = parameters_debt["factor_IB"] * np.matmul(W_matrix_IB, IB_debt[:, t])
                TA_paths[:, t] = Cash_paths[:, t] + A_ID_paths[:, t] + A_IB_paths[:, t] + A_IBC_paths[:, t] + A_EQC_paths[:, t] 
                Equity_paths[:, t] = TA_paths[:, t] - Deposit_paths[:, t] - IB_debt[:, t] - IB_CoCo_debt[:, t]
                
                # Check for additional conversions 
                if np.any(L_IBc_0 > 0): 
                    ETA_ratio[(Healthy_banks + Converted_Banks), t] = Equity_paths[(Healthy_banks + Converted_Banks), t] / TA_paths[(Healthy_banks + Converted_Banks), t] 
                    Converting_due_to_default = [i for i in range(n_banks) if ETA_ratio[i, t] <= CoCo_thresholds[i] and i not in Converted_Banks and i not in Defaulted_Banks]
                    Converted_Banks.extend(Converting_due_to_default)
                    Healthy_banks = [i for i in range(n_banks) if i not in (Defaulted_Banks + Converted_Banks)]
                    
                    Converting_t = Converting_due_to_default
                    while len(Converting_t) > 0:    # Keep converting until no more conversions are possible
                        
                        # Compute weights of equity for creditors (only used when there is interbank CoCo debt)
                        for k in Converting_t:
                            q_max = (Equity_paths[k, t] + IB_CoCo_debt[k, t]) / IB_CoCo_debt[k, t]
                            W_matrix_E[:, k] = W_matrix_CoCo[:, k] * IB_CoCo_debt[k, t] * np.minimum(parameters_debt["CoCo_q"], q_max) / (Equity_paths[k, t] + IB_CoCo_debt[k, t])
                        
                        # Cancel CoCo liabilities
                        IB_CoCo_debt[Converting_t, t] = 0
                        Equity_paths[Converting_t, t] = TA_paths[Converting_t, t] - Deposit_paths[Converting_t, t] - IB_debt[Converting_t, t] - IB_CoCo_debt[Converting_t, t]
                        
                        # Cross-holding
                        if np.any(A_IBC_paths[:, 0]>0):
                            A_IBC_paths[:, t] = np.matmul(W_matrix_CoCo, IB_CoCo_debt[:, t])
                            A_EQC_paths[:, t] = np.matmul(W_matrix_E, Equity_paths[:, t])
                            TA_paths[:, t] = Cash_paths[:, t] + A_ID_paths[:, t] + A_IB_paths[:, t] + A_IBC_paths[:, t] + A_EQC_paths[:, t] 
                            Equity_paths[:, t] = TA_paths[:, t] - Deposit_paths[:, t] - IB_debt[:, t] - IB_CoCo_debt[:, t]
                        
                        # Check for further conversions
                        ETA_ratio[(Healthy_banks + Converted_Banks), t] = Equity_paths[(Healthy_banks + Converted_Banks), t] / TA_paths[(Healthy_banks + Converted_Banks), t]
                        Converting_t = [i for i in range(n_banks) if ETA_ratio[i, t] <= CoCo_thresholds[i] and i not in Converted_Banks and i not in Defaulted_Banks]
                        Converted_Banks.extend(Converting_t)
                        Healthy_banks = [i for i in range(n_banks) if i not in (Defaulted_Banks + Converted_Banks)]

                # Check for additional defaults
                Default_t = [i for i in (Healthy_banks + Converted_Banks) if Equity_paths[i, t] < 0 and i not in Defaulted_Banks]
                Defaulted_Banks.extend(Default_t)
                Converted_Banks = list(set(Converted_Banks) - set(Default_t))
                Healthy_banks = [i for i in range(n_banks) if i not in (Defaulted_Banks + Converted_Banks)]

            ### Check ###
            if len(Healthy_banks) + len(Converted_Banks) + len(Defaulted_Banks) != n_banks: 
                print(f"Error: {len(Healthy_banks) + len(Converted_Banks) + len(Defaulted_Banks)} out of {n_banks} assigned.")

            ################# Z-score #################
            Z_paths[(Healthy_banks + Converted_Banks), t] = 1.2 * Cash_paths[(Healthy_banks + Converted_Banks), t] / TA_paths[(Healthy_banks + Converted_Banks), t] + 0.6 * Equity_paths[(Healthy_banks + Converted_Banks), t] / (Deposit_paths[(Healthy_banks + Converted_Banks), t] + IB_debt[(Healthy_banks + Converted_Banks), t])
            Z_paths[Defaulted_Banks, t] = 0

            Status_over_time[:, t] = np.array([2 if i in Defaulted_Banks else 0 for i in range(n_banks)]) + np.array([1 if i in Converted_Banks else 0 for i in range(n_banks)]) 

        status_array_small[0, :, idx] = np.mean((Status_over_time[:35, :] == 0), axis=0)
        status_array_small[1, :, idx] = np.mean((Status_over_time[:35, :] == 1), axis=0)
        status_array_small[2, :, idx] = np.mean((Status_over_time[:35, :] == 2), axis=0)

        status_array_large[0, :, idx] = np.mean((Status_over_time[35:, :] == 0), axis=0)  
        status_array_large[1, :, idx] = np.mean((Status_over_time[35:, :] == 1), axis=0)  
        status_array_large[2, :, idx] = np.mean((Status_over_time[35:, :] == 2), axis=0)  

        bankruptcy_array_small[idx] = len([i for i in range(n_banks) if (i in Defaulted_Banks and i<35)])
        bankruptcy_array_large[idx] = len([i for i in range(n_banks) if (i in Defaulted_Banks and i>=35)])
        conversion_array_small[idx] = len([i for i in range(n_banks) if (i in Converted_Banks and i<35)])
        conversion_array_large[idx] = len([i for i in range(n_banks) if (i in Converted_Banks and i>=35)])
        equity_array_small[:, idx] = np.sum(Equity_paths[:35], axis=0)
        equity_array_large[:, idx] = np.sum(Equity_paths[35:], axis=0)

    return  bankruptcy_array_small, bankruptcy_array_large, conversion_array_small, conversion_array_large, status_array_small, status_array_large , equity_array_small, equity_array_large

#################### GENERATE SCENARIO DATA #################### 

## General parametrs
nb_steps = 252                  # Time steps
T = 1                           # Time horizon
n_l = 5                         # Large banks
n_s = 35                        # Small banks
n_banks = n_l + n_s             # Total number of banks
n_industries = 10               # Industrial sectors 

## Parameters Interest Rate and Spread
param_dict_base_rate = {"r_0" : 0.047763, "alpha_r": 2.338321, "r_bar": 0.047763, "sigma_r": 0.046260}
param_dict_spread = {"s_0" : np.array([0.009294,  0.007216, 0.005192, 0.003920]), "alpha_s": np.array([0.754983, 1.024749, 1.597957, 2.564611]), "s_bar": np.array([0.009294,  0.007216, 0.005192, 0.003920]), "sigma_s": np.array([0.023673, 0.025400, 0.022777, 0.027028])}

## Parameters Industry Loan Assets
cov_I = np.ones((n_industries, n_industries)) * 0.0
np.fill_diagonal(cov_I, 1)
param_dict_industry_debt = {"I_0": np.ones(n_industries) * 1, "alpha_I": np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), "I_bar" : np.array([0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35]), "sigma_I":np.array([0.02, 0.02, 0.05, 0.05, 0.08, 0.08, 0.11, 0.11, 0.14, 0.14]),
                            "lambda_I": 0.5, "shock_I_mu": -0.05, "shock_I_sigma": 0.05, "shock_I_corr": 0.10,  "cov_I": cov_I}

## Parameters Cash
param_dict_cash = {"lambda_C": 0.5, "shock_C_mu": -0.05, "shock_C_sigma": 0.05, "shock_C_corr": 0.10}


### Generate scenarios and save 
nb_Simulations = 2000

compute_scenario_data = False
if compute_scenario_data:
    r_base_array0, spread_array0, I_array0, Cash_shock_array0 = BalanceSheet_scenarios(parameters_int_rate = param_dict_base_rate, parameters_spread = param_dict_spread, parameters_industry = param_dict_industry_debt, parameters_cash = param_dict_cash, 
                                                                                n_banks = n_banks, n_l = n_l, T = T, N_steps = nb_steps, nbSimulations = nb_Simulations, shock_scenario = 0)

    r_base_array1, spread_array1, I_array1, Cash_shock_array1 = BalanceSheet_scenarios(parameters_int_rate = param_dict_base_rate, parameters_spread = param_dict_spread, parameters_industry = param_dict_industry_debt, parameters_cash = param_dict_cash, 
                                                                                n_banks = n_banks, n_l = n_l, T = T, N_steps = nb_steps, nbSimulations = nb_Simulations, shock_scenario = 1)

    r_base_array2, spread_array2, I_array2, Cash_shock_array2 = BalanceSheet_scenarios(parameters_int_rate = param_dict_base_rate, parameters_spread = param_dict_spread, parameters_industry = param_dict_industry_debt, parameters_cash = param_dict_cash, 
                                                                                n_banks = n_banks, n_l = n_l, T = T, N_steps = nb_steps, nbSimulations = nb_Simulations, shock_scenario = 2)

    r_base_array3, spread_array3, I_array3, Cash_shock_array3 = BalanceSheet_scenarios(parameters_int_rate = param_dict_base_rate, parameters_spread = param_dict_spread, parameters_industry = param_dict_industry_debt, parameters_cash = param_dict_cash, 
                                                                                n_banks = n_banks, n_l = n_l, T = T, N_steps = nb_steps, nbSimulations = nb_Simulations, shock_scenario = 3)

    # CHECKS
    print(np.sum(r_base_array0 - r_base_array1))
    print(np.sum(spread_array0 - spread_array1))
    print(np.sum(r_base_array1 - r_base_array2))
    print(np.sum(r_base_array1 - r_base_array3))
    print(np.sum(spread_array1 - spread_array2))
    print(np.sum(spread_array1 - spread_array3))
    print(np.sum(I_array1 - I_array3))

    # ## Save scenarios
    with open(fr'Analysis ABT2/Scenarios/scenario0', 'wb') as f:
        pickle.dump([r_base_array0, spread_array0, I_array0, Cash_shock_array0], f)
    with open(fr'Analysis ABT2/Scenarios/scenario1', 'wb') as f:
        pickle.dump([r_base_array1, spread_array1, I_array1, Cash_shock_array1], f)
    with open(fr'Analysis ABT2/Scenarios/scenario2', 'wb') as f:
        pickle.dump([r_base_array2, spread_array2, I_array2, Cash_shock_array2], f)
    with open(fr'Analysis ABT2/Scenarios/scenario3', 'wb') as f:
        pickle.dump([r_base_array3, spread_array3, I_array3, Cash_shock_array3], f)


#################### GENERATE BALANCE SHEET DATA #################### 

# Sensitivty analysis           
R_rate = 0.6                    # Recovery rate (easy to set for all specifications)
alpha_trigger = 0.4             # Trigger (easy to set for all specifications)

## Parametrs Interbank Debt Liabilities
param_dict_interbank_debt = {"duration": 1.5, "convexity": 1.5**2, "duration_CoCo": 1.5, "convexity_CoCo": 1.5**2, "factor_IB": 7/10, "Recovery_rate": R_rate, "CoCo_q": 0.0}

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
W_matrix_IBd = pd.read_csv('Analysis ABT2/W_matrix_CH5_ME.csv', index_col=0).to_numpy()
W_matrix_IBd_norm = W_matrix_IBd / np.sum(W_matrix_IBd, axis=0)

# Industru debt holdings
generate_weight_matrix = False
if generate_weight_matrix:
    W_matrix_ID = np.zeros((n_industries, n_banks))
    for j in range(n_banks):
        if j < 35: 
            k = 3
            nonzero_rows = np.random.choice(n_industries, k, replace=False)
        else:
            k = n_industries
            nonzero_rows = np.random.choice(n_industries, k, replace=False)
        
        W_matrix_ID[nonzero_rows, j] = 1/k
    with open(fr'Analysis ABT2/weight_matrix_industry', 'wb') as f:
        pickle.dump(W_matrix_ID, f)
else: 
    with open(fr'Analysis ABT2/weight_matrix_industry', 'rb') as f:
        W_matrix_ID = pickle.load(f)

# CoCo thresholds
alpha_th = alpha_trigger
thresholds = alpha_th * (E_0 / (C_0 + A_IBr_0 + A_IBc_0 + A_IBe_0 + A_ID_0))  

compute_results = False
nSim = 2_000
if compute_results: 
    for s in [1, 3]:
        with open(fr'Analysis ABT2/Scenarios/scenario{s}', 'rb') as f:
            scenarios_r_base, scenarios_spread, scenarios_Ind, scenarios_Cash = pickle.load(f)

        bankruptcy_array_small, bankruptcy_array_large, conversion_array_small, conversion_array_large, status_array_small, status_array_large , equity_array_small, equity_array_large = Compute_evolution_system(C_0, A_IBr_0, A_IBc_0, A_IBe_0, A_ID_0, DEP_0, L_IBd_0, L_IBc_0, E_0, scen_r_base = scenarios_r_base, scen_spread = scenarios_spread, scen_I = scenarios_Ind, scen_C = scenarios_Cash, 
                                                                                                                                                                                            parameters_debt = param_dict_interbank_debt, parameters_deposits = param_dict_deposits, W_matrix_ID = W_matrix_ID, W_matrix_IB = W_matrix_IBd_norm, W_matrix_CoCo = W_matrix_IBd_norm, 
                                                                                                                                                                                            CoCo_thresholds = thresholds, W_matrix_E = np.zeros((n_banks, n_banks)), T = T, N_steps = nb_steps, nbSimulations = nSim)
        
        save_results = True
        if save_results:
            with open(fr'Analysis ABT2/Baseline/Results_NoCoCo_scen{s}', 'wb') as f:
                pickle.dump([bankruptcy_array_small, bankruptcy_array_large, conversion_array_small, conversion_array_large, status_array_small, status_array_large , equity_array_small, equity_array_large], f)

# #################################################################################################################################################################################################################################

## Parametrs Interbank Debt Liabilities
param_dict_interbank_debt = {"duration": 1.5, "convexity": 1.5**2, "duration_CoCo": 1.5, "convexity_CoCo": 1.5**2, "factor_IB": 10/10, "Recovery_rate": R_rate, "CoCo_q": 0.0}

## Parameters Initial Balance Sheets
# Small banks
C_0s = np.ones(n_s) * 25
A_IBr_0s = np.ones(n_s) * 7
A_IBc_0s = np.ones(n_s) * 0
A_IBe_0s = np.ones(n_s) * 0
A_ID_0s = np.ones(n_s) * 68

DEP_0s = np.ones(n_s) * 80
L_IBd_0s = np.ones(n_s) * 7
L_IBc_0s = np.ones(n_s) * 3
E_0s = np.ones(n_s) * 10

# Large banks
mult_large = 20 
C_0l = np.ones(n_l) * 30 * mult_large 
A_IBr_0l = np.ones(n_l) * 7 * mult_large 
A_IBc_0l = np.ones(n_l) * 0 * mult_large 
A_IBe_0l = np.ones(n_l) * 0  
A_ID_0l = np.ones(n_l) * 63 * mult_large 

DEP_0l = np.ones(n_l) * 80 * mult_large 
L_IBd_0l = np.ones(n_l) * 7 * mult_large 
L_IBc_0l = np.ones(n_l) * 3 * mult_large 
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

# CoCo thresholds
alpha_th = alpha_trigger
thresholds = alpha_th * (E_0 / (C_0 + A_IBr_0 + A_IBc_0 + A_IBe_0 + A_ID_0))        # threshold equals 0.4 of the equity ratio

compute_results = False
if compute_results: 
    for s in [1, 3]:
        with open(fr'Analysis ABT2/Scenarios/scenario{s}', 'rb') as f:
            scenarios_r_base, scenarios_spread, scenarios_Ind, scenarios_Cash = pickle.load(f)

        bankruptcy_array_small, bankruptcy_array_large, conversion_array_small, conversion_array_large, status_array_small, status_array_large , equity_array_small, equity_array_large = Compute_evolution_system(C_0, A_IBr_0, A_IBc_0, A_IBe_0, A_ID_0, DEP_0, L_IBd_0, L_IBc_0, E_0, scen_r_base = scenarios_r_base, scen_spread = scenarios_spread, scen_I = scenarios_Ind, scen_C = scenarios_Cash, 
                                                                                                                                                                                            parameters_debt = param_dict_interbank_debt, parameters_deposits = param_dict_deposits, W_matrix_ID = W_matrix_ID, W_matrix_IB = W_matrix_IBd_norm, W_matrix_CoCo = W_matrix_IBd_norm, 
                                                                                                                                                                                            CoCo_thresholds = thresholds, W_matrix_E = np.zeros((n_banks, n_banks)), T = T, N_steps = nb_steps, nbSimulations = nSim)

        save_results = True
        if save_results:
            with open(fr'Analysis ABT2/Baseline/Results_CoCoNoCross_scen{s}', 'wb') as f:
                pickle.dump([bankruptcy_array_small, bankruptcy_array_large, conversion_array_small, conversion_array_large, status_array_small, status_array_large , equity_array_small, equity_array_large], f)

# #################################################################################################################################################################################################################################

## Parameters Initial Balance Sheets
# Small banks
C_0s = np.ones(n_s) * 25
A_IBr_0s = np.ones(n_s) * 4
A_IBc_0s = np.ones(n_s) * 3
A_IBe_0s = np.ones(n_s) * 0
A_ID_0s = np.ones(n_s) * 68

DEP_0s = np.ones(n_s) * 80
L_IBd_0s = np.ones(n_s) * 7
L_IBc_0s = np.ones(n_s) * 3
E_0s = np.ones(n_s) * 10

# Large banks
mult_large = 20 
C_0l = np.ones(n_l) * 30 * mult_large 
A_IBr_0l = np.ones(n_l) * 4 * mult_large 
A_IBc_0l = np.ones(n_l) * 3 * mult_large 
A_IBe_0l = np.ones(n_l) * 0  
A_ID_0l = np.ones(n_l) * 63 * mult_large 

DEP_0l = np.ones(n_l) * 80 * mult_large 
L_IBd_0l = np.ones(n_l) * 7 * mult_large 
L_IBc_0l = np.ones(n_l) * 3 * mult_large 
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
 
# CoCo thresholds
alpha_th = alpha_trigger
thresholds = alpha_th * (E_0 / (C_0 + A_IBr_0 + A_IBc_0 + A_IBe_0 + A_ID_0))        # threshold equals 0.4 of the equity ratio

compute_results = False
if compute_results: 
    for q in [0, 1/2, 1, 3/2]:
        for s in [1, 3]:
            param_dict_interbank_debt = {"duration": 1.5, "convexity": 1.5**2, "duration_CoCo": 1.5, "convexity_CoCo": 1.5**2, "factor_IB": 4/7, "Recovery_rate": R_rate, "CoCo_q": q}
            with open(fr'Analysis ABT2/Scenarios/scenario{s}', 'rb') as f:
                scenarios_r_base, scenarios_spread, scenarios_Ind, scenarios_Cash = pickle.load(f)

            bankruptcy_array_small, bankruptcy_array_large, conversion_array_small, conversion_array_large, status_array_small, status_array_large , equity_array_small, equity_array_large = Compute_evolution_system(C_0, A_IBr_0, A_IBc_0, A_IBe_0, A_ID_0, DEP_0, L_IBd_0, L_IBc_0, E_0, scen_r_base = scenarios_r_base, scen_spread = scenarios_spread, scen_I = scenarios_Ind, scen_C = scenarios_Cash, 
                                                                                                                                                                                                parameters_debt = param_dict_interbank_debt, parameters_deposits = param_dict_deposits, W_matrix_ID = W_matrix_ID, W_matrix_IB = W_matrix_IBd_norm, W_matrix_CoCo = W_matrix_IBd_norm, 
                                                                                                                                                                                                CoCo_thresholds = thresholds, W_matrix_E = np.zeros((n_banks, n_banks)), T = T, N_steps = nb_steps, nbSimulations = nSim)
            save_results = True
            if save_results:
                with open(fr'Analysis ABT2/Baseline/Results_CoCoCross_{q}_scen{s}', 'wb') as f:
                    pickle.dump([bankruptcy_array_small, bankruptcy_array_large, conversion_array_small, conversion_array_large, status_array_small, status_array_large , equity_array_small, equity_array_large], f)

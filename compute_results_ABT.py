import numpy as np
import matplotlib.pyplot as plt
import pickle

## Function
def compute_results_ABT(filename, folder='Baseline', scenarios = [1, 2, 3], q_values = [0, 1/2, 1, 3/2]):
    n_l = 5                       
    n_s = 35                              
    mult_large = 20 
    E_0s = np.ones(n_s) * 10
    E_0l = np.ones(n_l) * 10 * mult_large 
    E_0 = np.hstack((E_0s, E_0l))
    nb_steps = 252

    
    if len(q_values) > 0:
        results_q = []
        for q in q_values:
            result_dict = {}
            for s in scenarios:
                with open(fr'Analysis ABT2/{folder}/Results_{filename}_{q}_scen0', 'rb') as f:
                    bankruptcy_array_small0, bankruptcy_array_large0, conversion_array_small0, conversion_array_large0, status_array_small0, status_array_large0 , equity_array_small0, equity_array_large0 = pickle.load(f)
                with open(fr'Analysis ABT2/{folder}/Results_{filename}_{q}_scen{s}', 'rb') as f:
                    bankruptcy_array_small, bankruptcy_array_large, conversion_array_small, conversion_array_large, status_array_small, status_array_large , equity_array_small, equity_array_large = pickle.load(f)

                nbsmall = np.mean(bankruptcy_array_small) - np.mean(bankruptcy_array_small0)
                nblarge = np.mean(bankruptcy_array_large) - np.mean(bankruptcy_array_large0)
                ncsmall = np.mean(conversion_array_small) - np.mean(conversion_array_small0)
                nclarge = np.mean(conversion_array_large) - np.mean(conversion_array_large0)

                EquityVaR_time_small  = np.quantile(equity_array_small0, q = 0.05, axis=1)
                CoVaR_time_small  = np.quantile(equity_array_small, q = 0.05, axis=1)
                Delta_CoVaR_T_small  =  EquityVaR_time_small[-1] / np.sum(E_0s) - CoVaR_time_small[-1] / np.sum(E_0s)

                EquityVaR_time_large = np.quantile(equity_array_large0, q = 0.05, axis=1)
                CoVaR_time_large = np.quantile(equity_array_large, q = 0.05, axis=1)
                Delta_CoVaR_T_large = EquityVaR_time_large[-1] / np.sum(E_0l) - CoVaR_time_large[-1] / np.sum(E_0l)

                EquityVaR_panel_small  = np.quantile(equity_array_small0, q = 0.05)
                CoVaR_panel_small = np.quantile(equity_array_small, q = 0.05) 
                Delta_CoVaR_panel_small = EquityVaR_panel_small / np.sum(E_0s) - CoVaR_panel_small / np.sum(E_0s) 

                EquityVaR_panel_large = np.quantile(equity_array_large0, q = 0.05)
                CoVaR_panel_large = np.quantile(equity_array_large, q = 0.05) 
                Delta_CoVaR_panel_large = EquityVaR_panel_large / np.sum(E_0l) - CoVaR_panel_large / np.sum(E_0l) 

                EquityVaR_time_total = np.quantile(equity_array_large0 + equity_array_small0, q = 0.05, axis=1)
                CoVaR_time_total = np.quantile(equity_array_large + equity_array_small, q = 0.05, axis=1)
                Delta_CoVaR_T_total = EquityVaR_time_total[-1] / np.sum(E_0) - CoVaR_time_total[-1] / np.sum(E_0)

                EquityVaR_panel_total = np.quantile(equity_array_small0 + equity_array_large0, q = 0.05)
                CoVaR_panel_total = np.quantile(equity_array_small + equity_array_large, q = 0.05) 
                Delta_CoVaR_panel_total = EquityVaR_panel_total / np.sum(E_0) - CoVaR_panel_total / np.sum(E_0) 

                # Delta CoES
                EquityES_time_small = np.zeros((nb_steps+1))
                CoES_time_small = np.zeros((nb_steps+1))
                Delta_CoES_time_small = np.zeros((nb_steps+1))
                EquityES_time_large = np.zeros((nb_steps+1))
                CoES_time_large = np.zeros((nb_steps+1))
                Delta_CoES_time_large = np.zeros((nb_steps+1))
                EquityES_time_total = np.zeros((nb_steps+1))
                CoES_time_total = np.zeros((nb_steps+1))
                Delta_CoES_time_total = np.zeros((nb_steps+1))
                for t in range(nb_steps+1):
                    EQT0_small = equity_array_small0[t]
                    EQT_small = equity_array_small[t]
                    EquityES_time_small[t] = np.mean(EQT0_small[EQT0_small <= EquityVaR_time_small[t]])
                    CoES_time_small[t] = np.mean(EQT_small[EQT_small <= CoVaR_time_small[t]])
                    Delta_CoES_time_small[t]  =  EquityES_time_small[t] / np.sum(E_0s) - CoES_time_small[t] / np.sum(E_0s)

                    EQT0_large = equity_array_large0[t]         
                    EQT_large = equity_array_large[t]
                    EquityES_time_large[t] = np.mean(EQT0_large[EQT0_large <= EquityVaR_time_large[t]])
                    CoES_time_large[t] = np.mean(EQT_large[EQT_large <= CoVaR_time_large[t]])
                    Delta_CoES_time_large[t] = EquityES_time_large[t] / np.sum(E_0l) - CoES_time_large[t] / np.sum(E_0l)

                    EQT0_total = equity_array_small0[t] + equity_array_large0[t]
                    EQT_total = equity_array_small[t] + equity_array_large[t]
                    EquityES_time_total[t] = np.mean(EQT0_total[EQT0_total <= EquityVaR_time_total[t]])
                    CoES_time_total[t] = np.mean(EQT_total[EQT_total <= CoVaR_time_total[t]])
                    Delta_CoES_time_total[t] = EquityES_time_total[t] / np.sum(E_0) - CoES_time_total[t] / np.sum(E_0)
                
                EquityES_panel_small = np.mean(equity_array_small0[equity_array_small0 <= EquityVaR_panel_small])
                CoES_panel_small = np.mean(equity_array_small[equity_array_small <= CoVaR_panel_small])
                Delta_CoES_panel_small =  EquityES_panel_small / np.sum(E_0s) - CoES_panel_small / np.sum(E_0s)

                EquityES_panel_large = np.mean(equity_array_large0[equity_array_large0 <= EquityVaR_panel_large])
                CoES_panel_large = np.mean(equity_array_large[equity_array_large <= CoVaR_panel_large])
                Delta_CoES_panel_large =  EquityES_panel_large / np.sum(E_0l) - CoES_panel_large / np.sum(E_0l)

                total_eq0 = equity_array_small0 + equity_array_large0
                total_eq = equity_array_small + equity_array_large
                EquityES_panel_total = np.mean(total_eq0[total_eq0 <= EquityVaR_panel_total])
                CoES_panel_total = np.mean(total_eq[total_eq <= CoVaR_panel_total])
                Delta_CoES_panel_total = EquityES_panel_total / np.sum(E_0) - CoES_panel_total / np.sum(E_0)

                result_list = [nblarge, nbsmall, nclarge, ncsmall, Delta_CoVaR_panel_large, Delta_CoVaR_panel_small, Delta_CoVaR_panel_total, Delta_CoES_panel_large, Delta_CoES_panel_small, Delta_CoES_panel_total]
                result_dict[s] = result_list
            
            results_q.append(result_dict)
        
        return results_q

    else:
        result_dict = {}
        for s in scenarios:
            with open(fr'Analysis ABT2/{folder}/Results_{filename}_scen0', 'rb') as f:
                bankruptcy_array_small0, bankruptcy_array_large0, conversion_array_small0, conversion_array_large0, status_array_small0, status_array_large0 , equity_array_small0, equity_array_large0 = pickle.load(f)
            with open(fr'Analysis ABT2/{folder}/Results_{filename}_scen{s}', 'rb') as f:
                bankruptcy_array_small, bankruptcy_array_large, conversion_array_small, conversion_array_large, status_array_small, status_array_large , equity_array_small, equity_array_large = pickle.load(f)

            nbsmall = np.mean(bankruptcy_array_small) - np.mean(bankruptcy_array_small0)
            nblarge = np.mean(bankruptcy_array_large) - np.mean(bankruptcy_array_large0)
            ncsmall = np.mean(conversion_array_small) - np.mean(conversion_array_small0)
            nclarge = np.mean(conversion_array_large) - np.mean(conversion_array_large0)

            EquityVaR_time_small  = np.quantile(equity_array_small0, q = 0.05, axis=1)
            CoVaR_time_small  = np.quantile(equity_array_small, q = 0.05, axis=1)
            Delta_CoVaR_T_small  =  EquityVaR_time_small[-1] / np.sum(E_0s) - CoVaR_time_small[-1] / np.sum(E_0s)

            EquityVaR_time_large = np.quantile(equity_array_large0, q = 0.05, axis=1)
            CoVaR_time_large = np.quantile(equity_array_large, q = 0.05, axis=1)
            Delta_CoVaR_T_large = EquityVaR_time_large[-1] / np.sum(E_0l) - CoVaR_time_large[-1] / np.sum(E_0l)

            EquityVaR_panel_small  = np.quantile(equity_array_small0, q = 0.05)
            CoVaR_panel_small = np.quantile(equity_array_small, q = 0.05) 
            Delta_CoVaR_panel_small = EquityVaR_panel_small / np.sum(E_0s) - CoVaR_panel_small / np.sum(E_0s) 

            EquityVaR_panel_large = np.quantile(equity_array_large0, q = 0.05)
            CoVaR_panel_large = np.quantile(equity_array_large, q = 0.05) 
            Delta_CoVaR_panel_large = EquityVaR_panel_large / np.sum(E_0l) - CoVaR_panel_large / np.sum(E_0l) 

            EquityVaR_time_total = np.quantile(equity_array_large0 + equity_array_small0, q = 0.05, axis=1)
            CoVaR_time_total = np.quantile(equity_array_large + equity_array_small, q = 0.05, axis=1)
            Delta_CoVaR_T_total = EquityVaR_time_total[-1] / np.sum(E_0) - CoVaR_time_total[-1] / np.sum(E_0)

            EquityVaR_panel_total = np.quantile(equity_array_small0 + equity_array_large0, q = 0.05)
            CoVaR_panel_total = np.quantile(equity_array_small + equity_array_large, q = 0.05) 
            Delta_CoVaR_panel_total = EquityVaR_panel_total / np.sum(E_0) - CoVaR_panel_total / np.sum(E_0) 

            # Delta CoES
            EquityES_time_small = np.zeros((nb_steps+1))
            CoES_time_small = np.zeros((nb_steps+1))
            Delta_CoES_time_small = np.zeros((nb_steps+1))
            EquityES_time_large = np.zeros((nb_steps+1))
            CoES_time_large = np.zeros((nb_steps+1))
            Delta_CoES_time_large = np.zeros((nb_steps+1))
            EquityES_time_total = np.zeros((nb_steps+1))
            CoES_time_total = np.zeros((nb_steps+1))
            Delta_CoES_time_total = np.zeros((nb_steps+1))
            for t in range(nb_steps+1):
                EQT0_small = equity_array_small0[t]
                EQT_small = equity_array_small[t]
                EquityES_time_small[t] = np.mean(EQT0_small[EQT0_small <= EquityVaR_time_small[t]])
                CoES_time_small[t] = np.mean(EQT_small[EQT_small <= CoVaR_time_small[t]])
                Delta_CoES_time_small[t]  =  EquityES_time_small[t] / np.sum(E_0s) - CoES_time_small[t] / np.sum(E_0s)

                EQT0_large = equity_array_large0[t]         
                EQT_large = equity_array_large[t]
                EquityES_time_large[t] = np.mean(EQT0_large[EQT0_large <= EquityVaR_time_large[t]])
                CoES_time_large[t] = np.mean(EQT_large[EQT_large <= CoVaR_time_large[t]])
                Delta_CoES_time_large[t] = EquityES_time_large[t] / np.sum(E_0l) - CoES_time_large[t] / np.sum(E_0l)

                EQT0_total = equity_array_small0[t] + equity_array_large0[t]
                EQT_total = equity_array_small[t] + equity_array_large[t]
                EquityES_time_total[t] = np.mean(EQT0_total[EQT0_total <= EquityVaR_time_total[t]])
                CoES_time_total[t] = np.mean(EQT_total[EQT_total <= CoVaR_time_total[t]])
                Delta_CoES_time_total[t] = EquityES_time_total[t] / np.sum(E_0) - CoES_time_total[t] / np.sum(E_0)
            
            EquityES_panel_small = np.mean(equity_array_small0[equity_array_small0 <= EquityVaR_panel_small])
            CoES_panel_small = np.mean(equity_array_small[equity_array_small <= CoVaR_panel_small])
            Delta_CoES_panel_small =  EquityES_panel_small / np.sum(E_0s) - CoES_panel_small / np.sum(E_0s)

            EquityES_panel_large = np.mean(equity_array_large0[equity_array_large0 <= EquityVaR_panel_large])
            CoES_panel_large = np.mean(equity_array_large[equity_array_large <= CoVaR_panel_large])
            Delta_CoES_panel_large =  EquityES_panel_large / np.sum(E_0l) - CoES_panel_large / np.sum(E_0l)

            total_eq0 = equity_array_small0 + equity_array_large0
            total_eq = equity_array_small + equity_array_large
            EquityES_panel_total = np.mean(total_eq0[total_eq0 <= EquityVaR_panel_total])
            CoES_panel_total = np.mean(total_eq[total_eq <= CoVaR_panel_total])
            Delta_CoES_panel_total = EquityES_panel_total / np.sum(E_0) - CoES_panel_total / np.sum(E_0)

            result_list = [nblarge, nbsmall, nclarge, ncsmall, Delta_CoVaR_panel_large, Delta_CoVaR_panel_small, Delta_CoVaR_panel_total, Delta_CoES_panel_large, Delta_CoES_panel_small, Delta_CoES_panel_total]
            result_dict[s] = result_list
    
        return result_dict

### PRINT OUT RESULTS ###
resultsNoCoCo = compute_results_ABT(filename='NoCoCo', q_values=[])
resultsCoCoNoCross = compute_results_ABT(filename='CoCoNoCross', q_values=[])
resultsCoCoCross = compute_results_ABT(filename='CoCoCross')

def print_results_latex(scenarios, resultsNoCoCo, resultsCoCoNoCross, resultsCoCoCross):
    for s in scenarios:
        print(f"Bankrupties - scenario {s}: {resultsNoCoCo[s][0]:.3f} & {resultsNoCoCo[s][1]:.3f} & {(resultsNoCoCo[s][0] + resultsNoCoCo[s][1]):.3f} & {resultsCoCoNoCross[s][0]:.3f} & {resultsCoCoNoCross[s][1]:.3f} & {(resultsCoCoNoCross[s][0] + resultsCoCoNoCross[s][1]):.3f} & {resultsCoCoCross[0][s][0]:.3f} & {resultsCoCoCross[0][s][1]:.3f} & {(resultsCoCoCross[0][s][0] + resultsCoCoCross[0][s][1]):.3f} & {resultsCoCoCross[1][s][0]:.3f} & {resultsCoCoCross[1][s][1]:.3f} & {(resultsCoCoCross[1][s][0] + resultsCoCoCross[1][s][1]):.3f} & {resultsCoCoCross[2][s][0]:.3f} & {resultsCoCoCross[2][s][1]:.3f} & {(resultsCoCoCross[2][s][0] + resultsCoCoCross[2][s][1]):.3f} & {resultsCoCoCross[3][s][0]:.3f} & {resultsCoCoCross[3][s][1]:.3f} & {(resultsCoCoCross[3][s][0] + resultsCoCoCross[3][s][1]):.3f} ")
        print(f"Conversions - scenario {s}: {resultsNoCoCo[s][2]:.3f} & {resultsNoCoCo[s][3]:.3f} & {(resultsNoCoCo[s][2] + resultsNoCoCo[s][3]):.3f} & {resultsCoCoNoCross[s][2]:.3f} & {resultsCoCoNoCross[s][3]:.3f} & {(resultsCoCoNoCross[s][2] + resultsCoCoNoCross[s][3]):.3f} & {resultsCoCoCross[0][s][2]:.3f} & {resultsCoCoCross[0][s][3]:.3f} & {(resultsCoCoCross[0][s][2] + resultsCoCoCross[0][s][3]):.3f} & {resultsCoCoCross[1][s][2]:.3f} & {resultsCoCoCross[1][s][3]:.3f} & {(resultsCoCoCross[1][s][2] + resultsCoCoCross[1][s][3]):.3f} & {resultsCoCoCross[2][s][2]:.3f} & {resultsCoCoCross[2][s][3]:.3f} & {(resultsCoCoCross[2][s][2] + resultsCoCoCross[2][s][3]):.3f} & {resultsCoCoCross[3][s][2]:.3f} & {resultsCoCoCross[3][s][3]:.3f} & {(resultsCoCoCross[3][s][2] + resultsCoCoCross[3][s][3]):.3f} ")
        print(f"Delta CoVaR - scenario {s}: {resultsNoCoCo[s][4]:.3f} & {resultsNoCoCo[s][5]:.3f} & {(resultsNoCoCo[s][6]):.3f} & {resultsCoCoNoCross[s][4]:.3f} & {resultsCoCoNoCross[s][5]:.3f} & {(resultsCoCoNoCross[s][6]):.3f} & {resultsCoCoCross[0][s][4]:.3f} & {resultsCoCoCross[0][s][5]:.3f} & {(resultsCoCoCross[0][s][6]):.3f} & {resultsCoCoCross[1][s][4]:.3f} & {resultsCoCoCross[1][s][5]:.3f} & {(resultsCoCoCross[1][s][6]):.3f} & {resultsCoCoCross[2][s][4]:.3f} & {resultsCoCoCross[2][s][5]:.3f} & {(resultsCoCoCross[2][s][6]):.3f} & {resultsCoCoCross[3][s][4]:.3f} & {resultsCoCoCross[3][s][5]:.3f} & {(resultsCoCoCross[3][s][6]):.3f}")
        print(f"Delta CoES - scenario {s}: {resultsNoCoCo[s][7]:.3f} & {resultsNoCoCo[s][8]:.3f} & {(resultsNoCoCo[s][9]):.3f} & {resultsCoCoNoCross[s][7]:.3f} & {resultsCoCoNoCross[s][8]:.3f} & {(resultsCoCoNoCross[s][9]):.3f} & {resultsCoCoCross[0][s][7]:.3f} & {resultsCoCoCross[0][s][8]:.3f} & {(resultsCoCoCross[0][s][9]):.3f} & {resultsCoCoCross[1][s][7]:.3f} & {resultsCoCoCross[1][s][8]:.3f} & {(resultsCoCoCross[1][s][9]):.3f} & {resultsCoCoCross[2][s][7]:.3f} & {resultsCoCoCross[2][s][8]:.3f} & {(resultsCoCoCross[2][s][9]):.3f} & {resultsCoCoCross[3][s][7]:.3f} & {resultsCoCoCross[3][s][8]:.3f} & {(resultsCoCoCross[3][s][9]):.3f}")

resultsNoCoCoTRLOW = compute_results_ABT(filename='NoCoCoRT0.2', folder='Trigger', scenarios = [2], q_values=[])
resultsCoCoNoCrossTRLOW = compute_results_ABT(filename='CoCoNoCrossT0.2', folder='Trigger', scenarios = [2], q_values=[])
resultsCoCoCrossTRLOW = compute_results_ABT(filename='CoCoCrossT0.2', folder='Trigger', scenarios = [2])

resultsNoCoCoTRHIGH = compute_results_ABT(filename='NoCoCoRT0.6', folder='Trigger', scenarios = [2], q_values=[])
resultsCoCoNoCrossTRHIGH = compute_results_ABT(filename='CoCoNoCrossT0.6', folder='Trigger', scenarios = [2], q_values=[])
resultsCoCoCrossTRHIGH = compute_results_ABT(filename='CoCoCrossT0.6', folder='Trigger', scenarios = [2])

# print_results_latex([2], resultsNoCoCoTRLOW, resultsCoCoNoCrossTRLOW, resultsCoCoCrossTRLOW)
# print_results_latex([2], resultsNoCoCo, resultsCoCoNoCross, resultsCoCoCross)
# print_results_latex([2], resultsNoCoCoTRHIGH, resultsCoCoNoCrossTRHIGH, resultsCoCoCrossTRHIGH)

resultsNoCoCoRLOW = compute_results_ABT(filename='NoCoCoR0.2', folder='Recovery rate', scenarios = [2], q_values=[])
resultsCoCoNoCrossRLOW = compute_results_ABT(filename='CoCoNoCrossR0.2', folder='Recovery rate', scenarios = [2], q_values=[])
resultsCoCoCrossRLOW = compute_results_ABT(filename='CoCoCrossR0.2', folder='Recovery rate', scenarios = [2])

resultsNoCoCoRHIGH = compute_results_ABT(filename='NoCoCoR1.0', folder='Recovery rate', scenarios = [2], q_values=[])
resultsCoCoNoCrossRHIGH = compute_results_ABT(filename='CoCoNoCrossR1.0', folder='Recovery rate', scenarios = [2], q_values=[])
resultsCoCoCrossRHIGH = compute_results_ABT(filename='CoCoCrossR1.0', folder='Recovery rate', scenarios = [2])

print_results_latex([2], resultsNoCoCoRLOW, resultsCoCoNoCrossRLOW, resultsCoCoCrossRLOW)
print_results_latex([2], resultsNoCoCo, resultsCoCoNoCross, resultsCoCoCross)
print_results_latex([2], resultsNoCoCoRHIGH, resultsCoCoNoCrossRHIGH, resultsCoCoCrossRHIGH)
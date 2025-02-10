# Explanation of this repository 

- This repository contains the code used to generate the results in my thesis. I explain per Chapter of the thesis how the code should be used. 
- The file 'helper_functions.py' contains multiple helper functions that are called in the files I describe below. 

## Chapter 3
- The files 'static_model_CH3.py' and 'multi_period_model_CH3.py' are used to generate the results in Chapter 3. The files can be used to run the examples described in this Chapter and generate the figures/tables. 

## Chapter 4 
- The files described here are used to analyze systemic risk in the model of Balter et al. (2023). 
- The files 'static_model_contagion_CH4_single.py' and 'static_model_contagion_CH4_multi.py' are used to generate the results in Chapter 4.2. The folder 'Analysis contagion' contains the pickle files with the results of Section 4.2 saved in them. These pickle files can be loaded in the files 'static_model_contagion_CH4_single.py' and 'static_model_contagion_CH4_multi.py' to generate the figures/tables in Section 4.2 
- The file 'static_model_system_CH4.py' is used generate the results in Chapter 4.4. The folder 'Analysis EBA' contains pickle files with the results of Section 4.4 saved in them. These pickle files can be loaded in the file 'static_model_system_CH4.py' to generate the figures/tables in Section 4.4 
    * The folder 'Data EBA' contains the cleaned EBA data set and the liability matrices in csv format generated using the method of Gandy and Veraart (2017). The csv file of the raw EBA data set is not included here in this repo because it is a large file. One can download the EBA data usng the link on p.34 of the thesis.

## Chapter 5
- The file 'generate_scenarios_ABT.py' is used to generate the sample paths of the balance sheets of all banks in each of the three shock scenarios. The file contains the function 'BalanceSheet_scenarios' which is used to generate the sample paths of the stochastic quantities used in the model of Gupta et al. (2021) in each of the three shock scenarios. The output of this function is saved in files. 
- The file 'generate_scenarios_ABT.py' is also used to compute the sample paths of the balance sheets, using the functon 'Compute_evolution_system'. This function outputs the quantities needed to compute the results in Chapter 5.4/5.5, which are saved in pickle files in the folders 'Analysis ABT2/Baseline', 'Analysis ABT2/Recovery rate', and
'Analysis ABT2/Trigger'. To run the functon 'Compute_evolution_system', the following input is required:
    * The files 'Analysis ABT2/W_matrix_CH5_ME.csv' and 'Analysis ABT2/weight_matrix_industry.pkl' (contain the interbank and industry weight matrices used in the code).
    * The files containing the random numbers (outputs of the function 'BalanceSheet_scenarios'). These are not uploaded to GitHub because these files are too large; they are available upon request, or can be generated using the function 'BalanceSheet_scenarios'. 

- The file 'compute_results_ABT.py' loads the results from the above mentioned pickle files, computes the resulting systemic risk metrics, and then prints them in the right format for the tables in Chapter 5.4/5.5.
- The file 'RunEstimation.m' in the folder 'MATLAB_code' is used to estimate the interest rate and spread parameters of the CIR processes in Equations (5.8) and (5.9) in the thesis. We edited the 'RunEstimation.m' from the package of Kladıvko (2007) to fit to our data. To run this file, one needs to download the entire package, which is linked in the thesis on p.49. The folder also contains the 'license.txt' file from the package of Kladıvko (2007). 

## References
* Balter, A., Schweizer, N., and Vera, J. (2023). Contingent capital with stock price triggers in interbank
networks. Mathematics of Operations Research, 48(1):520–543.
* Gandy, A. and Veraart, L. A. M. (2017). A bayesian methodology for systemic risk assessment in
financial networks. Management Science, 63(12):4428–4446.
* Gupta, A., Wang, R., and Lu, Y. (2021). Addressing systemic risk using contingent convertible debt
– a network analysis. European Journal of Operational Research, 290(1):263–277.
* Kladıvko, K. (2007). Maximum likelihood estimation of the cox-ingersoll-ross process: the matlab
implementation. In Technical Computing Prague. working paper. 


    




    






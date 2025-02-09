# Explanation of this repository 

- This repository contains the code used to generate the results in my thesis. I explain per Chapter of the thesis how the code should be used. 
- The 'file helper_functions.py' contains multiple helper functions that are called in the files I describe below. 

## Chapter 3
- The files 'static_model_CH3.py' and 'mult_period_model_CH3.py' are used to generate the results in Chapter 3. The files can be used to run the examples described in this Chapter and generate the figures/tables. 

## Chapter 4 
- The files described here are used to analyze systemic risk in the model of Balter et al. (2023). 
- The files 'static_model_contagion_CH4_single.py' and 'static_model_contagion_CH4_multi.py' are used to generate the results in Chapter 4.2. The folder 'Analysis contagion' contains the pickle files with the results of Section 4.2 saved in them. These pickle files can be loaded in the files 'static_model_contagion_CH4_single.py' and 'static_model_contagion_CH4_multi.py' to generate the figures/tables in Section 4.2 
- The file 'static_model_system_CH4.py' is used generate the results in Chapter 4.4. The folder 'Analysis EBA' contains pickle files with the results of Section 4.4 saved in them. These pickle files can be loaded in the file 'static_model_system_CH4.py' to generate the figures/tables in Section 4.4 
    * The file imports the csv file with the cleaned EBA data set, and the csv files with the generate liability matrices using the method of Gandy and Veraart (2017). The csv files are not uploaded to GitHub, but are available upon request. Alternatively, one can download the EBA data, clean this data set using the file 'clean_EBA_dataset.py', and use the R-code 'generate_liab_matrix.R' in the folder R_code to generate the csv files of the liability matrices.

## Chapter 5
- The file 'generate_scenarios_ABT.py' is used to generate the sample paths of the balance sheets of all banks in each of the three shock scenarios. The file contains the function 'BalanceSheet_scenarios' which is used to generate the sample paths of the stochastic quantities used in the model of Gupta et al. (2021) in each of the three shock scenarios. The output of this function is saved in files and used in the function 'Compute_evolution_system' to compute the sample paths of the balance sheets. The quantities needed to compute the results in Chapter 5.4/5.5 are saved in pickle files in the folders 'Baseline', 'Recovery rate', and 'Trigger'. 
    * The files containing the random numbers (outputs of the function 'BalanceSheet_scenarios') are not uploaded to GitHub because these files are too large; they are available upon request. 
- The file 'compute_results_ABT.py' loads the results from the above mentioned pickle files, computes the resulting systemic risk metrics, and then prints them in the right format for the tables in Chapter 5.4/5.5.


    




    






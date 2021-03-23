#--------------------------------------------
# Arellano-Bond estimation for Note
# Mark van der Plaat
# March 2021
#--------------------------------------------

#--------------------------------------------
# Import Packages
#--------------------------------------------
    
# Data manipulation
import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid', font_scale = 2)

# GMM package
from linearmodels.iv import IVGMM

# Parallelization
from itertools import product
import multiprocessing as mp
from joblib import Parallel, delayed
num_cores = mp.cpu_count()

# Set WD
import os
os.chdir(r'D:\RUG\PhD\Materials_papers\01-Note_on_securitization')

#--------------------------------------------
# Load data and merge
#--------------------------------------------

# Load data
## Securitization data
df_sec = pd.read_csv('Data\df_sec_note.csv', index_col = 0)

## Other data
df_oth = pd.read_csv('Data\df_ri_rc_note.csv')

# Merge data
df = df_sec.merge(df_oth, how = 'inner', on = ['date','IDRSSD'])

# Set multi-index 
df.set_index(keys = ['date','IDRSSD'], inplace = True)

#--------------------------------------------
# Prelims
#--------------------------------------------

# Set variables
y = 'zscore'
sec_vars = df.columns[df.columns.str.contains('cr|hmda')].tolist() + ['cr_cd_gross','cr_cd_net']
x_exo = ['ln_ta','nim','cti','liq_ratio','loan_ratio','gap'] +\
        ['dum_{}'.format(year) for year in range(2016,2019+1)] +\
        ['constant']
x_endo = 'zscore_l1'
instruments = ['zscore_l2','zscore_l3','zscore_l4'] #+\
              #['ln_ta_l2','nim_l2','cti_l2','liq_ratio_l2','loan_ratio_l2','gap_l2'] #+\
              #['ln_ta_l1','nim_l1','cti_l1','liq_ratio_l1','loan_ratio_l1','gap_l1'] 

# Add gross and net credit derivative variables
df['cr_cd_gross'] = df[['cr_cd_sold', 'cr_cd_purchased']].sum(axis = 1)
df['cr_cd_net'] = df['cr_cd_purchased'].subtract(df['cr_cd_sold'])

# Make securitzation variables a ratio
for var in sec_vars:
    if '_cd_' in var:
        df[var] = df[var].divide(df.loan_net)
    else:
        df[var] = df[var].divide(df.loan_gross)

# Lag y var
df['zscore_l1'] = df.groupby(level = 1).zscore.shift(periods = 1)

# Take first differences 
df_fd = df.groupby(level = 1).diff()

# Make instruments
## NOTE: these instruments are not differenced but lagged two periods
df_grouped = df.groupby(level = 1)

for var in sec_vars + x_exo[:6]:
    df_fd['{}_l1'.format(var)] = df_grouped[var].shift(periods = 1)

for var in [y] + sec_vars + x_exo[:6]:
    df_fd['{}_l2'.format(var)] = df_grouped[var].shift(periods = 2)
    
# Lag zscore l3 and l4
df_fd['zscore_l3'] = df_grouped['zscore'].shift(periods = 3)
df_fd['zscore_l4'] = df_grouped['zscore'].shift(periods = 4)

# Add time dummies
time_dummies = pd.get_dummies(df_fd.index.get_level_values(0), prefix = 'dum')
df_fd[time_dummies.columns] = time_dummies.set_index(df_fd.index)

# Add constant
df_fd['constant'] = 1

# Drop NA
df_fd.dropna(inplace = True)

#--------------------------------------------
# Run Arellano Bond estimator
#--------------------------------------------

# Function to save results to excel
def results2Excel(results, filename):
    # Main table
    ## Get table
    main_table = pd.DataFrame(results.summary.tables[1])
    
    ## Set index
    main_table = main_table.set_index(main_table.iloc[:,0])
    
    ## Set index
    main_table.columns = main_table.iloc[0,:]
    
    ## Remove empty cells
    main_table = main_table.iloc[1:,1:]
    
    # Secondary table
    ## Get part of table
    second_table_part1 = pd.DataFrame(results.summary.tables[0]).iloc[:-1,:2]
    second_table_part2 = pd.DataFrame(results.summary.tables[0]).iloc[:-2,2:]
    
    ## Change column names
    for table in [second_table_part1,second_table_part2]:
        table.columns = ['key','value']
        
    ## Concat
    second_table = pd.concat([second_table_part1,second_table_part2],ignore_index = True)
    
    ## Set index
    second_table.index = second_table.iloc[:,0]
    
    ## Remove empty cells
    second_table = second_table.iloc[:,1:]
    
    # Sargan-Hansen J test
    j_test = pd.DataFrame([string.split(':') for string in str(results.j_stat).split('\n')])
    
    # C statistic
    #c_test = pd.DataFrame([string.split(':') for string in str(results.c_stat(variables = [x_endo])).split('\n')])
    
    # Save to excel
    writer = pd.ExcelWriter('Results\GMM_IV\{}.xlsx'.format(filename), engine = 'xlsxwriter')
    
    for sheet, table in zip(['main','secondary','j_test'],[main_table, second_table, j_test]):
        table.to_excel(writer, sheet_name = sheet)
    writer.save()

# Full Sample
#--------------------------------------------

# All variables separate
for sec in sec_vars:
    # Run model
    model = IVGMM(df_fd[y],\
                       df_fd[[sec] + x_exo],\
                       df_fd[x_endo],\
                       df_fd[instruments])
    results = model.fit()
    
    # Save to excel
    results2Excel(results, 'gmmiv_full_sep_{}'.format(sec))

# Model with gross cds
for sec in sec_vars[:6] + sec_vars[8:-2]:
    # Run model
    model = IVGMM(df_fd[y],\
                       df_fd[[sec] + ['cr_cd_gross'] + x_exo],\
                       df_fd[x_endo],\
                       df_fd[instruments])
    results = model.fit()
    
    # Save to excel
    results2Excel(results, 'gmmiv_full_gross_{}'.format(sec))

# Model with net cds
for sec in sec_vars[:6] + sec_vars[8:-2]:
    # Run model
    model = IVGMM(df_fd[y],\
                       df_fd[[sec] + ['cr_cd_net'] + x_exo],\
                       df_fd[x_endo],\
                       df_fd[instruments])
    results = model.fit()
    
    # Save to excel
    results2Excel(results, 'gmmiv_full_net_{}'.format(sec))

# Sec. only
#--------------------------------------------
df_fd_sec = df_fd[(df_fd[sec_vars] > 0).any(axis = 1)]

# All variables separate
for sec in sec_vars:
    # Run model
    model = IVGMM(df_fd_sec[y],\
                       df_fd_sec[[sec] + x_exo],\
                       df_fd_sec[x_endo],\
                       df_fd_sec[instruments + ['{}_l2'.format(sec)]])
    results = model.fit()
    
    # Save to excel
    results2Excel(results, 'gmmiv_sec_sep_{}'.format(sec))

# Model with gross cds
for sec in sec_vars[:6] + sec_vars[8:-2]:
    # Run model
    model = IVGMM(df_fd_sec[y],\
                       df_fd_sec[[sec] + ['cr_cd_gross'] + x_exo],\
                       df_fd_sec[x_endo],\
                       df_fd_sec[instruments])
    results = model.fit()
    
    # Save to excel
    results2Excel(results, 'gmmiv_sec_gross_{}'.format(sec))

# Model with net cds
for sec in sec_vars[:6] + sec_vars[8:-2]:
    # Run model
    model = IVGMM(df_fd_sec[y],\
                       df_fd_sec[[sec] + ['cr_cd_net'] + x_exo],\
                       df_fd_sec[x_endo],\
                       df_fd_sec[instruments])
    results = model.fit()
    
    # Save to excel
    results2Excel(results, 'gmmiv_sec_net_{}'.format(sec))

# Test Bootstrap
#--------------------------------------------

# Function
def BootstrapIVGMM(B, sec):
    # Original model
    model_original = IVGMM(df_fd[y],\
                       df_fd[[sec] + x_exo],\
                       df_fd[x_endo],\
                       df_fd[instruments])
    results_original = model_original.fit()
    
    # Set boostrap prelims
    params_b = []
    std_b = []
    np.random.seed(seed=1)
    
    for b in range(B):
    # Resample 
        df_fd_b = df_fd.sample(frac = 1, replace = True)
        
        # Run model
        model_b = IVGMM(df_fd_b[y],\
                           df_fd_b[[sec] + x_exo],\
                           df_fd_b[x_endo],\
                           df_fd_b[instruments])
        results_b = model_b.fit()
        
        # Save params and std
        params_b.append(results_b.params)
        std_b.append(results_b.std_errors)
    
    return results_original, params_b, std_b
    
# Bootstrap
B = 99

if __name__ == '__main__':
    ori_b, params_b, std_b = zip(*Parallel(n_jobs=num_cores)(delayed(BootstrapIVGMM)(B, var) for var in sec_vars))
    
sns.boxplot(data = [[params_b[j][i][0] for i in range(B)] for j in range(len(sec_vars))])

'''
# Calculate variance and t-stats (for first parameter)
var_mean = np.mean([params_b[i][0] for i in range(B)])
var_b = (1 / (B - 1)) * np.sum([(params_b[i][0] - var_mean)**2 for i in range(B)])
t_b = np.sort([(params_b[i][0] - results_original.params[0]) / var_b for i in range(B)])
alpha = 0.1
i_b = (B + 1) * (1 - alpha/2) 
bias_b = var_mean - results_original.params[0] '''
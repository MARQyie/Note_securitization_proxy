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
        ['dum_{}'.format(year) for year in range(2014,2019+1)] +\
        ['constant']
x_endo = 'zscore_l1'
instruments = ['zscore_l2'] +\
              ['ln_ta_l2','nim_l2','cti_l2','liq_ratio_l2','loan_ratio_l2','gap_l2']

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

for var in [y] + sec_vars + x_exo[:6]:
    df_fd['{}_l2'.format(var)] = df_grouped[var].shift(periods = 2)

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
    c_test = pd.DataFrame([string.split(':') for string in str(results.c_stat()).split('\n')])
    
    # Save to excel
    writer = pd.ExcelWriter('Results\GMM_IV\{}.xlsx'.format(filename), engine = 'xlsxwriter')
    
    for sheet, table in zip(['main','secondary','j_test','c_test'],[main_table, second_table, j_test, c_test]):
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
                       df_fd[['cr_cd_gross'] + [sec] + x_exo],\
                       df_fd[x_endo],\
                       df_fd[instruments])
    results = model.fit()
    
    # Save to excel
    results2Excel(results, 'gmmiv_full_gross_{}'.format(sec))

# Model with net cds
for sec in sec_vars[:6] + sec_vars[8:-2]:
    # Run model
    model = IVGMM(df_fd[y],\
                       df_fd[['cr_cd_net'] + [sec] + x_exo],\
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
                       df_fd_sec[instruments])
    results = model.fit()
    
    # Save to excel
    results2Excel(results, 'gmmiv_sec_sep_{}'.format(sec))

# Model with gross cds
for sec in sec_vars[:6] + sec_vars[8:-2]:
    # Run model
    model = IVGMM(df_fd_sec[y],\
                       df_fd_sec[['cr_cd_gross'] + [sec] + x_exo],\
                       df_fd_sec[x_endo],\
                       df_fd_sec[instruments])
    results = model.fit()
    
    # Save to excel
    results2Excel(results, 'gmmiv_sec_gross_{}'.format(sec))

# Model with net cds
for sec in sec_vars[:6] + sec_vars[8:-2]:
    # Run model
    model = IVGMM(df_fd_sec[y],\
                       df_fd_sec[['cr_cd_net'] + [sec] + x_exo],\
                       df_fd_sec[x_endo],\
                       df_fd_sec[instruments])
    results = model.fit()
    
    # Save to excel
    results2Excel(results, 'gmmiv_sec_net_{}'.format(sec))


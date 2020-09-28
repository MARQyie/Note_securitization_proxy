#--------------------------------------------
# Factor analisys for Note
# Mark van der Plaat
# September 2020
#--------------------------------------------

''' This script performs a factor analysis
    on the proxies of securitization. We remove the SDI variables
    and cr_ta_vie. We also remove the loan sales variables.
    '''
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

# Machine learning packages
from sklearn import preprocessing
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from factor_analyzer import FactorAnalyzer

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

#--------------------------------------------
# Prelims
#--------------------------------------------

# Variable names
## Call Reports
vars_cr = df.columns[df.columns.str.contains('cr')].tolist()

## HMDA
vars_hmda = df.columns[df.columns.str.contains('hmda')].tolist()

## Total
vars_tot = vars_cr + vars_hmda + ['RC2170']
vars_tot.remove('cr_ta_vie')

# standardize data
df_standard = preprocessing.scale(df[vars_tot])

#--------------------------------------------
# Factor Analysis
#--------------------------------------------

'''NOTE: from the previous PCA we know that the first three principle components
    explain 72% of the variance. Hence we base our analysis on three components.
    
    The standard method of the FA package is minimum residual. Also possible:
        MLE, PCA. 
    '''

# Pre-tests
## Bartlett's test. H0: equal variance
bartlett_chi ,bartlett_p = calculate_bartlett_sphericity(df[vars_tot]) # p = 0.0

## Kaiser-Meyer-Olkin (KMO) test. Measures data suitability; should be between 0 and 1, but above 0.6
kmo_all, kmo_model = calculate_kmo(df[vars_tot]) #kmo_model = 0.604

#--------------------------------------------
# Factor Analysis
fa = FactorAnalyzer(rotation = None, n_factors = 3)
fa.fit(df[vars_tot])
ev, v = fa.get_eigenvalues()

fig, ax = plt.subplots(figsize=(20,12)) 
ax.set(ylabel='Variance Explained (%)', xlabel = 'Factor')
ax.bar(range(df_standard.shape[1]), ev / np.sum(ev))
plt.tight_layout()

fig.savefig('Figures/Dimension_reduction/FA_variance_explained.png')

# Get the factor loadings
fa_loadings = pd.DataFrame(fa.loadings_,\
                           columns = range(3),\
                           index = vars_tot)

fa_loadings.to_csv('Results/fa_loadings_norotation_sec.csv')

#--------------------------------------------
# First rotation: varimax (maximizes the sum of the variance of squared loadings)
fa_vm = FactorAnalyzer(rotation = 'varimax', n_factors = 3)
fa_vm.fit(df[vars_tot])

# Get the factor loadings 
fa_vm_loadings = pd.DataFrame(fa_vm.loadings_,\
                           columns = range(3),\
                           index = vars_tot)

fa_vm_loadings.to_csv('Results/fa_loadings_varimax_sec.csv')

# Second rotation: promax (builds upon the varimax rotation, but ultimately allows factors to become correlated.)
fa_pm = FactorAnalyzer(rotation = 'promax', n_factors = 3)
fa_pm.fit(df[vars_tot])

# Get the factor loadings 
fa_pm_loadings = pd.DataFrame(fa_pm.loadings_,\
                           columns = range(3),\
                           index = vars_tot)

fa_pm_loadings.to_csv('Results/fa_loadings_promax_sec.csv')

# Third rotation: quartimax (minimizes the number of factors needed to explain each variable.)
fa_qm = FactorAnalyzer(rotation = 'quartimax', n_factors = 3)
fa_qm.fit(df[vars_tot])

# Get the factor loadings 
fa_qm_loadings = pd.DataFrame(fa_qm.loadings_,\
                           columns = range(3),\
                           index = vars_tot)

fa_qm_loadings.to_csv('Results/fa_loadings_quartimax_sec.csv')

#--------------------------------------------
# Save loadings to excel
#--------------------------------------------

writer = pd.ExcelWriter('Results/fa_loadings_all.xlsx', engine='xlsxwriter')

fa_loadings.to_excel(writer, sheet_name='no_rotation')
fa_vm_loadings.to_excel(writer, sheet_name='varimax')
fa_pm_loadings.to_excel(writer, sheet_name='promax')
fa_qm_loadings.to_excel(writer, sheet_name='quartimax')

writer.save()

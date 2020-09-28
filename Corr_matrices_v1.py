#--------------------------------------------
# Correlation matrices for Note
# Mark van der Plaat
# September 2020
#--------------------------------------------

''' This script makes correlation matrices for the securitization
    note. For all correlation matrices we use the Spearman's rank
    correlation coefficient, which is a nonparametric measure of rank
    correlation.
    
    We make the following matrices:
        1) All variables, no scaling
        2) All variables, scaling with total assets or income
    '''
#--------------------------------------------
# Import Packages
#--------------------------------------------
    
# Data manipulation
import pandas as pd
import numpy as np
from scipy import stats

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid', font_scale = 1.5)

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

## SDI
vars_sdi = df.columns[df.columns.str.contains('sdi')].tolist()

## Total
vars_tot = vars_cr + vars_hmda + vars_sdi + ['RC2170']
vars_tot.remove('cr_ta_vie')

#--------------------------------------------
# Heatmap function
#--------------------------------------------

def heatmap(matrix, file, annot = True):
    
    # Set aesthetics
    dic_aes = {'annot' : annot,
               'vmin' : -1,
               'vmax' : 1,
               'center': 0,
               'cmap': 'coolwarm'}
    
    # Make heatmap
    fig, ax = plt.subplots(figsize=(24,12))  
    sns.heatmap(matrix, **dic_aes)
    plt.tight_layout()
    
    # Save heatmap
    fig.savefig('Figures/Correlation_maps/' + file)

#--------------------------------------------
# Correlation
#--------------------------------------------

# Total, not scaled
## Get correlation matrix
tot_ns_corr = df[vars_tot].corr(method = 'spearman')
_, tot_ns_corr_pval = stats.spearmanr(df[vars_tot])

## PLot
heatmap(tot_ns_corr, 'Corr_tot_noscale.png')

'''NOTE: 
        1) Scaling barely affects the correlation (duh)
        2) SDI variables correlate very strongly with the respective
           Call Report variables. Probably due to the fact they are
           designed to measure exactly the same thing. DROP SDI '''
           
# Call Reports and HMDA, not scaled
## Get correlation matrix
crhmda_ns_corr = df[[var for var in vars_tot if var not in vars_sdi] ].corr(method = 'spearman')
_, crhmda_ns_corr_pval = stats.spearmanr(df[vars_tot]) # Only one thing is insignficant: ABCP vs HMDA priv

## PLot
heatmap(crhmda_ns_corr, 'Corr_crhmda_noscale.png')

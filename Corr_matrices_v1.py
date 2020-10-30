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
sns.set(style = 'whitegrid', font_scale = 1.6)

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
df = df_sec.merge(df_oth, how = 'left', on = ['date','IDRSSD'])

#--------------------------------------------
# Prelims
#--------------------------------------------

# Variable names
## Call Reports
vars_cr = df.columns[df.columns.str.contains('cr')].tolist()

## HMDA
vars_hmda = df.columns[df.columns.str.contains('hmda')].tolist()

## Total
vars_tot = vars_cr + vars_hmda + ['ta']

#--------------------------------------------
# Heatmap function
#--------------------------------------------

def heatmap(matrix, pvals, file, annot = True):
    
    # Set aesthetics
    dic_aes_masked = {'mask' : pvals > 0.05,
                      'annot_kws': {"weight": "bold"},      
                      'annot' : annot,
                      'vmin' : -1,
                      'vmax' : 1,
                      'center': 0,
                      'cmap': 'coolwarm'}
    dic_aes_unmasked = {'mask' : pvals <= 0.05, 
                      'annot' : annot,
                      'vmin' : -1,
                      'vmax' : 1,
                      'center': 0,
                      'cbar': False,
                      'cmap': 'coolwarm'}
    
    # Make heatmap
    fig, ax = plt.subplots(figsize=(24,16))
    sns.heatmap(matrix, **dic_aes_unmasked)
    sns.heatmap(matrix, **dic_aes_masked)
    plt.tight_layout()
    
    # Save heatmap
    fig.savefig('Figures/Correlation_maps/' + file)

#--------------------------------------------
# Correlation
#--------------------------------------------

           
# Call Reports and HMDA, not scaled
## Get correlation matrix
corr = df[vars_tot].corr(method = 'spearman')
corr_pval = stats.spearmanr(df[vars_tot])[1] # Only one thing is insignficant: ABCP vs HMDA priv

## Label columns and index
labels = ['Sec. Income','CD Sold',\
             'CD Purchased',\
             'Assets Sold and Sec.','Asset Sold and Not Sec.',\
             'Cred. Exp. Oth.','TA Sec. Veh.','TA ABCP','TA Oth. VIEs',\
             'HDMA GSE','HMDA Private','HMDA Sec.','TA']

corr.index = labels
corr.columns = labels

## PLot
heatmap(corr, corr_pval, 'Corr.png')

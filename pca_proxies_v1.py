#--------------------------------------------
# Principle component analisys for Note
# Mark van der Plaat
# September 2020
#--------------------------------------------

''' This script performs a principle component analysis
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
from sklearn.decomposition import PCA

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
vars_tot = vars_cr + vars_hmda + ['ta']

# standardize data
df_standard = preprocessing.scale(df[vars_tot])

#--------------------------------------------
# PCA
#--------------------------------------------

# Use sklearn to do the PCA
## Set the method
pca = PCA()

## Get the principle components
princ_comp = pca.fit(df_standard)

# Plot the principle components: variance explained
fig, ax = plt.subplots(figsize=(15,9)) 
ax.set(ylabel='Variance Explained (%)', xlabel = 'Principle Component')
ax.bar(range(1, df_standard.shape[1] + 1), princ_comp.explained_variance_ratio_ * 100)
plt.tight_layout()

fig.savefig('Figures/Dimension_reduction/PCA_variance_explained.png')

# Get the loadings
pc_loadings = pd.DataFrame(pca.components_.T,\
                           columns = range(df_standard.shape[1]),\
                           index = vars_tot)
pc_loadings.to_csv('Results/pc_loadings_sec.csv')

# Analyze the top 3 PCs
# Make plot
w = 0.3
x = np.array([float(i) for i in range(len(vars_tot))])
fig, ax = plt.subplots(figsize=(15,9)) 
ax.set(ylabel='Absolute Loadings', xlabel = 'Variable')
ax.bar(x-w, pc_loadings.iloc[:,0].abs(), width = w, label = 'PC1')
ax.bar(x, pc_loadings.iloc[:,1].abs(), width = w, label = 'PC2')
ax.bar(x+w, pc_loadings.iloc[:,2].abs(), width = w, label = 'PC3')
plt.xticks(x, vars_tot)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
ax.legend()
plt.tight_layout()

fig.savefig('Figures/Dimension_reduction/PCA_loadings.png')





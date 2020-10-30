#--------------------------------------------
# Partial Correlation Matrix for Note
# Between different sec. proxies only
# Mark van der Plaat
# September 2020
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

# Machine learning packages
from scipy import stats, linalg

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

#--------------------------------------------
# Prelims
#--------------------------------------------
''' NOTE:
    Based on the factor analysis we select only the proxies that explain
    one of the three factors.
    
    The control variables are based on a number of studies, see paper.
    '''

# Variable names
## Call Reports
vars_cr = df.columns[df.columns.str.contains('cr')].tolist()

## HMDA
vars_hmda = df.columns[df.columns.str.contains('hmda')].tolist()

## Total
vars_tot = vars_cr + vars_hmda + ['ta']

# Set variables
vars_oth = ['t1_reglev', 't1_regcap', 'cap_ratio', 'dep_ratio', 'loan_ratio',\
            'ra_ratio', 'ci_ratio', 'agri_ratio', 'cons_ratio', 'othl_ratio',\
            'loan_hhi', 'roa', 'liq_ratio', 'cti', 'nii_nor', 'rwata', 'npl',\
            'co_ratio', 'all_ratio', 'prov_ratio']
    
#--------------------------------------------
#  Define function
#--------------------------------------------

def PCorr(C, p):
    '''
    Returns the sample linear partial correlation coefficients between a securitization
    proxy in S and a control variable in C, while controlling for all other control
    variables. Only returns the partial correlations between the proxy and the 
    individual control variables. 

    Parameters
    ----------
    C : array-like, shape (n, q)
        Array with the different control variables. Each column of C is taken as a variable
    p : integer
        Integer that is the size of the number of securitization variables
        
    
    Returns
    -------
    p_corr : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and S[:, j] controlling
        for the remaining variables in C and the proxy.
    p_corr_pval : array-like, shape (p, p)
        P[i, j] contains the p-value of the partial correlation of C[:, i] 
        and S[:, j] controlling for the remaining variables in C and the proxy.
    '''
    
    # Transform matrix to numpy array
    C = np.asarray(C)
    
    # Set dimensions to loop over
    q = C.shape[1]
    
    # Set empty arrays
    p_corr = np.zeros((p, p), dtype=np.float)
    p_corr_pval = np.zeros((p, p), dtype=np.float)
    
    # Add constant to C
    C = np.append(C, np.ones([C.shape[0],1]), axis = 1)
    
    # Loop over the two dimensions
    for i in range(p):
        p_corr[i, i] = 1
        for j in range(i+1, p):
            
            # Remove C[:,j] and C[:,i] from C with boolean selection
            idx = np.ones(q + 1, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            
            # Get the parameter estimates
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]
            
            # Calculate the residuals
            res_j = C[:, j] - C[:, idx].dot(beta_j)
            res_i = C[:, i] - C[:, idx].dot(beta_i)

           # Calculate the Spearman rank correlation 
            corr, pval = stats.spearmanr(res_i, res_j)
            
            # Add to p_corr matrix
            p_corr[j,i] = corr
            p_corr[i,j] = corr
            
            p_corr_pval[j,i] = pval
            p_corr_pval[i,j] = pval
    
    return p_corr, p_corr_pval

#--------------------------------------------
#  Calculate the partial correlation matrix
#--------------------------------------------

# Todo: scale securitization
# Scale the securitization variables.
## NOTE: divide all variables with TA, except TA. Take logs of TA
for var in vars_tot:
    if var == 'ta':
        df[var] = np.log(df[var])
    else:
        df[var] = df[var].divide(df.ta)

# Drop na
df.dropna(inplace = True)

# Get P-corr matrix
p_corr, p_corr_pval = PCorr(df[vars_tot + vars_oth], df[vars_tot].shape[1])

#--------------------------------------------
#  Make pretty correlation table
#--------------------------------------------

# Set labels
labels = ['Sec. Income','CD Sold',\
             'CD Purchased',\
             'Assets Sold and Sec.','Asset Sold and Not Sec.',\
             'Cred. Exp. Oth.','TA Sec. Veh.','TA ABCP','TA Oth. VIEs',\
             'HDMA GSE','HMDA Private','HMDA Sec.','TA']

df_corr = pd.DataFrame(p_corr, index = labels, columns = labels)

# Heatmap function
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

## PLot
heatmap(df_corr, p_corr_pval, 'Partial_correlation_matrix_sec.png')
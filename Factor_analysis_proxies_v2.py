#--------------------------------------------
# Factor analisys for Note
# Mark van der Plaat
# September 2020
#--------------------------------------------

''' This script performs an exploratory factor analysis
    on the proxies of securitization. 
    
    We use the procedure as described in Brown, T. A. (2015). Confirmatory
    Factor Analysis for Applied Research. In T. D. Little (Ed.), 
    (2nd ed., Vol. 53). New York: The Guilford Press.
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
df = pd.read_csv('Data\df_sec_note.csv', index_col = 0)

# set variable names
# Variable names
vars_tot = df.columns[2:].tolist()

# Subset data to only include securitizers
unique_idrssd = df[(df[vars_tot] > 0).any(axis = 1)].IDRSSD.unique()
df_sec = df[df.IDRSSD.isin(unique_idrssd)]

# standardize data
df_standard = pd.DataFrame(preprocessing.scale(df_sec[vars_tot]), columns = vars_tot)

#--------------------------------------------
# Check correlation matrix
#--------------------------------------------

# Set function
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
    
# Get correlation matrix
from scipy.stats import spearmanr
corr = df_standard[vars_tot].corr(method = 'spearman')
corr_pval = spearmanr(df_standard[vars_tot])[1] 

# Label columns and index
labels = ['Serv. Fees','Sec. Income','LS Income','CD Sold',\
             'CD Purchased',\
             'Assets Sold and Sec.','Asset Sold and Not Sec.',\
             'Cred. Exp. Oth.','TA Sec. Veh.','TA ABCP','TA Oth. VIEs',\
             'HDMA GSE','HMDA Private','HMDA Sec.']

corr.index = labels
corr.columns = labels

## PLot
heatmap(corr, corr_pval, 'Corr_standardized_fa.png')

#--------------------------------------------
# Check if multivariate normal distribution 
#--------------------------------------------

# Histograms 
df_sec[vars_tot].hist(bins = 10,figsize=(50,50)) # Not very informative

# QQ plot
from statsmodels.graphics.gofplots import qqplot

for var in vars_tot:
    qqplot(df_sec[var], line='s') 

'''NOTE: Very right-tail heavy'''

# Shapiro-Wilk, D’Agostino’s K^2, and  Anderson-Darling Test
from scipy.stats import shapiro, normaltest, anderson
lst_swtest = np.zeros((len(vars_tot),2))
lst_k2test = np.zeros((len(vars_tot),2))
lst_adtest = np.zeros((len(vars_tot),2))

for i in range(len(vars_tot)):
    lst_swtest[i] = shapiro(df_sec.iloc[:,i+2])
    lst_k2test[i] = normaltest(df_sec.iloc[:,i+2])
    anderson_test = anderson(df_sec.iloc[:,i]+2)
    lst_adtest[i] = anderson_test.statistic, anderson_test.statistic < anderson_test.critical_values[2] # at alpha = 5

'''NOTE: All reject H0 --> Not normal distributed '''

'''Conclusion: Do not use ML procedure '''
#--------------------------------------------
# Pre-tests
#--------------------------------------------

''' We perform two pre-tests
    1) Bartlett's test of sphericity: tests whether the correlation matrix 
        equals an identiy matrix (H0), which means that the variables are 
        unrelated;
    2) Kaiser-Meyer-Olkin test: Is a statistic that indicates the proportion
        of variance that might be caused by underlying factors. Test statistic
        should be over .5'
    '''

# Bartlett's test. H0: equal variance
bartlett_chi, bartlett_p = calculate_bartlett_sphericity(df_standard[vars_tot]) # p = 0.0

# Kaiser-Meyer-Olkin (KMO) test. Measures data suitability; should be between 0 and 1, but above 0.5
kmo_all, kmo_model = calculate_kmo(df_standard[vars_tot]) #kmo_model = 0.72

'''Note: looks good '''

#--------------------------------------------
# Determine number of factors
#--------------------------------------------

'''We use multiple selection criteria:
    1) Scree plot (elbow plot)
    2) Kaiser-Guttman rule
    3) Parallel analysis
    '''
    
# Get factor estimates
fa = FactorAnalyzer(rotation = None)
fa.fit(df_standard[vars_tot])
ev, v = fa.get_eigenvalues()

# Perform a parallel analysis
list_ev_rand = []

np.random.seed(1)
for i in range(100):
    df_rand = pd.DataFrame(np.random.rand(*df_standard[vars_tot].shape))
    fa_rand = FactorAnalyzer(rotation = None).fit(df_rand)
    ev_rand, _ = fa_rand.get_eigenvalues()
    list_ev_rand.append(ev_rand)

fig, ax = plt.subplots(figsize=(15,9)) 
ax.set(ylabel='Eigen Value', xlabel = 'Factor')
ax.plot(range(1, df_standard.shape[1] + 1), ev, marker = 'o', label = 'Factor')
ax.plot(range(1, df_standard.shape[1] + 1), np.mean(list_ev_rand,axis = 0), color = 'black', linestyle = '--', label = 'Parallel Analysis')
ax.legend()
plt.tight_layout()

fig.savefig('Figures/Dimension_reduction/FA_parallel.png')

'''NOTE: First four factors have an eigen value greater than 1.
    Scree plot and parallel analysis also show that four factors are 
    appropriate '''
#--------------------------------------------
# Factor rotation
#--------------------------------------------
    
# Get factor estimates
fa = FactorAnalyzer(rotation = None, n_factors = 4)
fa.fit(df_standard[vars_tot])

# Get the non-rotated factor loadings
fa_loadings = pd.DataFrame(fa.loadings_,\
                           columns = range(4),\
                           index = vars_tot)

fa_loadings.to_csv('Results/fa_loadings_norotation_sec.csv')

#--------------------------------------------
# Orthogonal rotations

# Varimax (maximizes the sum of the variance of squared loadings)
fa_vm = FactorAnalyzer(rotation = 'varimax', n_factors = 4)
fa_vm.fit(df_standard[vars_tot])

## Get the factor loadings 
fa_vm_loadings = pd.DataFrame(fa_vm.loadings_,\
                           columns = range(4),\
                           index = vars_tot)

fa_vm_loadings.to_csv('Results/fa_loadings_varimax_sec.csv')

# Quartimax (minimizes the number of factors needed to explain each variable.)
fa_qm = FactorAnalyzer(rotation = 'quartimax', n_factors = 4)
fa_qm.fit(df_standard[vars_tot])

## Get the factor loadings 
fa_qm_loadings = pd.DataFrame(fa_qm.loadings_,\
                           columns = range(4),\
                           index = vars_tot)

fa_qm_loadings.to_csv('Results/fa_loadings_quartimax_sec.csv')

#--------------------------------------------
# Oblique rotations

# Promax 
fa_pm = FactorAnalyzer(rotation = 'promax', n_factors = 4)
fa_pm.fit(df_standard[vars_tot])

## Get the factor loadings 
fa_pm_loadings = pd.DataFrame(fa_pm.loadings_,\
                           columns = range(4),\
                           index = vars_tot)

fa_pm_loadings.to_csv('Results/fa_loadings_promax_sec.csv')

# Oblimax
fa_om = FactorAnalyzer(rotation = 'oblimax', n_factors = 4)
fa_om.fit(df_standard[vars_tot])

## Get the factor loadings 
fa_om_loadings = pd.DataFrame(fa_om.loadings_,\
                           columns = range(4),\
                           index = vars_tot)

fa_om_loadings.to_csv('Results/fa_loadings_oblimax_sec.csv')

'''NOTE Based on promax we find badly behaving indicators: 'cr_sec_income', 
    'cr_ce_sec', 'cr_ta_vie_other','cr_ta_secveh','cr_ta_abcp'. We do not find poorly defined factors. 
    
    Preliminary interpretation:
        1) CDO/ABCP securitization
        2) Asset sales (sec and non-sec)
        3) General securitization
        4) Loan sales
    '''

#--------------------------------------------
# Rerun fa
#--------------------------------------------

# Get factor estimates
fa = FactorAnalyzer(rotation = None, n_factors = 3)
fa.fit(df_standard[[value for value in vars_tot if value not in ('cr_sec_income', 'cr_ce_sec', 'cr_ta_vie_other','cr_ta_secveh','cr_ta_abcp')]])

# Get eigenvalues
ev, v = fa.get_eigenvalues() # Note: Now only three factors are necessary

# Get the non-rotated factor loadings
fa_loadings = pd.DataFrame(fa.loadings_,\
                           columns = range(3),\
                           index = [value for value in vars_tot if value not in  ('cr_sec_income', 'cr_ce_sec', 'cr_ta_vie_other','cr_ta_secveh','cr_ta_abcp')])

fa_loadings.to_csv('Results/fa_loadings_norotation_sec_rerun.csv')

# quartimax
fa_qm = FactorAnalyzer(rotation = 'quartimax', n_factors = 3)
fa_qm.fit(df_standard[[value for value in vars_tot if value not in  ('cr_sec_income', 'cr_ce_sec', 'cr_ta_vie_other','cr_ta_secveh','cr_ta_abcp')]])

## Get the factor loadings 
fa_qm_loadings = pd.DataFrame(fa_qm.loadings_,\
                           columns = range(3),\
                           index = [value for value in vars_tot if value not in  ('cr_sec_income', 'cr_ce_sec', 'cr_ta_vie_other','cr_ta_secveh','cr_ta_abcp')])

fa_qm_loadings.to_csv('Results/fa_loadings_quartimax_sec_rerun.csv')

# promax
fa_pm = FactorAnalyzer(rotation = 'promax', n_factors = 3)
fa_pm.fit(df_standard[[value for value in vars_tot if value not in  ('cr_sec_income', 'cr_ce_sec', 'cr_ta_vie_other','cr_ta_secveh','cr_ta_abcp')]])

## Get the factor loadings 
fa_pm_loadings = pd.DataFrame(fa_pm.loadings_,\
                           columns = range(3),\
                           index = [value for value in vars_tot if value not in  ('cr_sec_income', 'cr_ce_sec', 'cr_ta_vie_other','cr_ta_secveh','cr_ta_abcp')])

fa_pm_loadings.to_csv('Results/fa_loadings_promax_sec_rerun.csv')


#--------------------------------------------
# Save loadings to excel
#--------------------------------------------

writer = pd.ExcelWriter('Results/fa_loadings_all.xlsx', engine='xlsxwriter')

fa_loadings.to_excel(writer, sheet_name='no_rotation')
fa_vm_loadings.to_excel(writer, sheet_name='varimax')
fa_pm_loadings.to_excel(writer, sheet_name='promax')
fa_qm_loadings.to_excel(writer, sheet_name='quartimax')
fa_om_loadings.to_excel(writer, sheet_name='oblimax')

writer.save()

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

# Make net cd variable
#df['cr_cd_net'] = df.cr_cd_purchased - df.cr_cd_sold

# set variable names
vars_tot = [var for var in df.columns if var not in ['IDRSSD','date','cr_cd_purchased','cr_cd_sold', 'cr_cds_sold', 'cr_trs_sold', 'cr_co_sold', 'cr_cdoth_sold', 'cr_ta_vie_other', 'cr_as_sec', 'cr_ce_sec', 'cr_serv_fees', 'cr_sec_income', 'cr_ls_income','ta']]
vars_tot_nols = [var for var in vars_tot if var not in ['hmda_gse_amount', 'hmda_priv_amount', 'cr_as_nonsec', 'cr_as_sbo']]
# Subset data to only needed variables
df_sub = df[vars_tot]

# First log the data (to reduce skewness in all but the income variables)
df_log = np.log(df_sub - df_sub.min() + 1) 

# standardize data
df_standard = pd.DataFrame(preprocessing.scale(df_log), columns = vars_tot)

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
    fig, ax = plt.subplots(figsize=(36,24))
    sns.heatmap(matrix, **dic_aes_unmasked)
    sns.heatmap(matrix, **dic_aes_masked)
    plt.tight_layout()
    
    # Save heatmap
    fig.savefig('Figures/Correlation_maps/' + file)
    
# Get correlation matrix
from scipy.stats import spearmanr
corr = df_log[vars_tot].corr(method = 'pearson')
corr_pval = spearmanr(df_log[vars_tot])[1] 

corr_nols = df_log[vars_tot_nols].corr(method = 'pearson')
corr_nols_pval = spearmanr(df_log[vars_tot_nols])[1] 

# Label columns and index
labels = ['HDMA GSE','HMDA Private','HMDA Sec.',\
          'CDS Pur.', 'TRS Pur.', 'CO Pur.',\
          'Other CDs Pur.', 'Assets Sold and Sec. (RMBS)',\
          'Assets Sold and Sec. (ABS)', 'Asset Sold and Not Sec.',\
          'Cred. Exp. Oth. (RMBS)', 'Cred. Exp. Oth. (ABS)', 'Asset Sold (SBO)',\
          'Cred. Exp. ABCP', 'Unused Comm. ABCP', 'TA Sec. Vehicles', \
          'TA ABCP Conduits', 'Comm. Paper ABCP']
labels_nols = ['HMDA Sec.',\
          'CDS Pur.', 'TRS Pur.', 'CO Pur.',\
          'Other CDs Pur.', 'Assets Sold and Sec. (RMBS)',\
          'Assets Sold and Sec. (ABS)',
          'Cred. Exp. Oth. (RMBS)', 'Cred. Exp. Oth. (ABS)', 
          'Cred. Exp. ABCP', 'Unused Comm. ABCP', 'TA Sec. Vehicles', \
          'TA ABCP Conduits', 'Comm. Paper ABCP']

corr.index = labels
corr.columns = labels

corr_nols.index = labels_nols
corr_nols.columns = labels_nols

## PLot
heatmap(corr, corr_pval, 'Corr_fa.png')
heatmap(corr_nols, corr_nols_pval, 'Corr_nols_fa.png')

#--------------------------------------------
# Check if multivariate normal distribution 
#--------------------------------------------

# Shapiro-Wilk, D’Agostino’s K^2, and  Anderson-Darling Test
from scipy.stats import shapiro, normaltest, anderson
lst_swtest = np.zeros((len(vars_tot),2))
lst_k2test = np.zeros((len(vars_tot),2))
lst_adtest = np.zeros((len(vars_tot),2))

for i in range(len(vars_tot)):
    lst_swtest[i] = shapiro(df_log.iloc[:,i])
    lst_k2test[i] = normaltest(df_log.iloc[:,i])
    anderson_test = anderson(df_log.iloc[:,i])
    lst_adtest[i] = anderson_test.statistic, anderson_test.statistic < anderson_test.critical_values[2] # at alpha = 5

'''NOTE: All reject H0 --> Not normal distributed '''

'''Conclusion: Use principle factors method '''
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
bartlett_chi_nols, bartlett_p_nols = calculate_bartlett_sphericity(df_standard[vars_tot_nols]) # p = 0.0


# Kaiser-Meyer-Olkin (KMO) test. Measures data suitability; should be between 0 and 1, but above 0.5
kmo_all, kmo_model = calculate_kmo(df_standard[vars_tot]) #kmo_model = 0.87
kmo_all_nols, kmo_model_nols = calculate_kmo(df_standard[vars_tot_nols]) #kmo_model = 0.88

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
fa = FactorAnalyzer(rotation = None, method = 'principal')
fa.fit(df_standard[vars_tot])
ev, v = fa.get_eigenvalues()

fa_nols = FactorAnalyzer(rotation = None, method = 'principal')
fa_nols.fit(df_standard[vars_tot_nols])
ev_nols, v_nols = fa_nols.get_eigenvalues()

# Perform a parallel analysis
list_ev_rand = []
list_ev_rand_nols = []

np.random.seed(1)
for i in range(100):
    df_rand = pd.DataFrame(np.random.rand(*df_standard[vars_tot].shape))
    df_rand_nols = pd.DataFrame(np.random.rand(*df_standard[vars_tot_nols].shape))
    
    fa_rand = FactorAnalyzer(rotation = None, method = 'principal').fit(df_rand)
    fa_rand_nols = FactorAnalyzer(rotation = None, method = 'principal').fit(df_rand_nols)
    
    ev_rand, _ = fa_rand.get_eigenvalues()
    ev_rand_nols, _ = fa_rand_nols.get_eigenvalues()
    
    list_ev_rand.append(ev_rand)
    list_ev_rand_nols.append(ev_rand_nols)

fig, ax = plt.subplots(figsize=(15,9)) 
ax.set(ylabel='Eigenvalue', xlabel = 'Factor')
ax.plot(range(1, df_standard.shape[1] + 1), ev, marker = 'o', label = 'Factor')
ax.plot(range(1, df_standard.shape[1] + 1), np.mean(list_ev_rand,axis = 0), color = 'black', linestyle = '--', label = 'Parallel Analysis')
ax.legend()
plt.tight_layout()

fig.savefig('Figures/Dimension_reduction/FA_parallel.png')

fig, ax = plt.subplots(figsize=(15,9)) 
ax.set(ylabel='Eigenvalue', xlabel = 'Factor')
ax.plot(range(1, df_standard[vars_tot_nols].shape[1] + 1), ev_nols, marker = 'o', label = 'Factor')
ax.plot(range(1, df_standard[vars_tot_nols].shape[1] + 1), np.mean(list_ev_rand_nols,axis = 0), color = 'black', linestyle = '--', label = 'Parallel Analysis')
ax.legend()
plt.tight_layout()

fig.savefig('Figures/Dimension_reduction/FA_parallel_nols.png')

'''NOTE: With LS: First 5 factors have an eigen value greater than 1.
    Parallel analysis: ev_parallel is slightly above ev_efa: 
        N factors is 4
    Scree plot: 5 factors
    
    Conclusion: 5 factors is appropriate
    
    Without LS: 3 factors are appropriate'''
#--------------------------------------------
# Factor rotation all variables
# Note: No orthogonal rotations, since the orth. assumption is unrealistic
# See Brown (2015)
#--------------------------------------------
    
# Get factor estimates
fa = FactorAnalyzer(rotation = None, n_factors = 5, method = 'principal')
fa.fit(df_standard[vars_tot])

# Get the non-rotated factor loadings
fa_loadings = pd.DataFrame(fa.loadings_,\
                           columns = range(5),\
                           index = vars_tot)

fa_loadings.to_csv('Results/fa_loadings_norotation.csv')

#--------------------------------------------
# Oblique rotations

# Promax 
fa_pm = FactorAnalyzer(rotation = 'promax', n_factors = 5, method = 'principal')
fa_pm.fit(df_standard[vars_tot])

## Get the factor loadings 
fa_pm_loadings = pd.DataFrame(fa_pm.loadings_,\
                           columns = range(5),\
                           index = vars_tot)

fa_pm_loadings.to_csv('Results/fa_loadings_promax.csv')

# Oblimin
fa_om = FactorAnalyzer(rotation = 'oblimin', n_factors = 5, method = 'principal')
fa_om.fit(df_standard[vars_tot])

## Get the factor loadings 
fa_om_loadings = pd.DataFrame(fa_om.loadings_,\
                           columns = range(5),\
                           index = vars_tot)

fa_om_loadings.to_csv('Results/fa_loadings_oblimin.csv')

# Quartimin
fa_qmo = FactorAnalyzer(rotation = 'quartimin', n_factors = 5, method = 'principal')
fa_qmo.fit(df_standard[vars_tot])

## Get the factor loadings 
fa_qmo_loadings = pd.DataFrame(fa_qmo.loadings_,\
                           columns = range(5),\
                           index = vars_tot)

fa_qmo_loadings.to_csv('Results/fa_loadings_quartimin.csv')

'''NOTE 
    Cutoff: x > .4 (cf. Brown, 2015: CFA for Applied Research p. 27).
    We base ourselfs on the promax rotation, other oblique rotations might
    lead to other results.
        
    Badly behaving indicators (low loadings)            : hmda_sec_amount, cr_secveh_ta
    Badly behaving indicators (multiple high loadings)  : NONE
    Badly identified factors (only 1--2 salient loading) : F3, F4
        
    Interpretation Factors:
        F0: CDO/ABCP Securitization
        F1: ABS Securitization
        F2: Loan Sales (Stock)
        F3: Loan Sales Income
        F4: Small Business Obligation Transfers
    
    Conclusion: Interpretation of the factors is straight forward. Three
    factors is probably most appropriate based on the loadings. cr_as_sbo
    probably not needed in follow-up CFA. Other variables to keep in mind
    when doing CFA: cr_ls_income, hmda_sec_amount, cr_secveh_ta.
    
    WARNING: The data contains many zeros. The PAF method might not be
    robust enough.
    '''

#--------------------------------------------
# Save loadings to excel


writer = pd.ExcelWriter('Results/fa_loadings_all.xlsx', engine='xlsxwriter')

fa_loadings.to_excel(writer, sheet_name='no_rotation')
fa_pm_loadings.to_excel(writer, sheet_name='promax')
fa_om_loadings.to_excel(writer, sheet_name='oblimin')
fa_qmo_loadings.to_excel(writer, sheet_name='quartimin')

writer.save()

#--------------------------------------------
# Factor rotation No loan sales
# Note: No orthogonal rotations, since the orth. assumption is unrealistic
# See Brown (2015)
#--------------------------------------------
    
# Get factor estimates
fa = FactorAnalyzer(rotation = None, n_factors = 3, method = 'principal')
fa.fit(df_standard[vars_tot_nols])

# Get the non-rotated factor loadings
fa_loadings = pd.DataFrame(fa.loadings_,\
                           columns = range(3),\
                           index = vars_tot_nols)

fa_loadings.to_csv('Results/fa_loadings_norotation_nols.csv')

#--------------------------------------------
# Oblique rotations

# Promax 
fa_pm = FactorAnalyzer(rotation = 'promax', n_factors = 3, method = 'principal')
fa_pm.fit(df_standard[vars_tot_nols])

## Get the factor loadings 
fa_pm_loadings = pd.DataFrame(fa_pm.loadings_,\
                           columns = range(3),\
                           index = vars_tot_nols)

fa_pm_loadings.to_csv('Results/fa_loadings_promax_nols.csv')

# Oblimin
fa_om = FactorAnalyzer(rotation = 'oblimin', n_factors = 3, method = 'principal')
fa_om.fit(df_standard[vars_tot_nols])

## Get the factor loadings 
fa_om_loadings = pd.DataFrame(fa_om.loadings_,\
                           columns = range(3),\
                           index = vars_tot_nols)

fa_om_loadings.to_csv('Results/fa_loadings_oblimin_nols.csv')

# Quartimin
fa_qmo = FactorAnalyzer(rotation = 'quartimin', n_factors = 3, method = 'principal')
fa_qmo.fit(df_standard[vars_tot_nols])

## Get the factor loadings 
fa_qmo_loadings = pd.DataFrame(fa_qmo.loadings_,\
                           columns = range(3),\
                           index = vars_tot_nols)

fa_qmo_loadings.to_csv('Results/fa_loadings_quartimin_nols.csv')

'''NOTE 
    Cutoff: x > .4 (cf. Brown, 2015: CFA for Applied Research p. 27).
    We base ourselfs on the promax rotation, other oblique rotations might
    lead to other results.
        
    Badly behaving indicators (low loadings)            : cr_secveh_ta
    Badly behaving indicators (multiple high loadings)  : cr_cds_purchased
    Badly identified factors (only 1--2 salient loading) : F2
        
    Interpretation Factors:
        F0: CDO/ABCP securitization
        F1: ABS securitization
        F2: HMDA RMBS securitization 
    '''

#--------------------------------------------
# Save loadings to excel

writer = pd.ExcelWriter('Results/fa_loadings_all_nols.xlsx', engine='xlsxwriter')

fa_loadings.to_excel(writer, sheet_name='no_rotation')
fa_pm_loadings.to_excel(writer, sheet_name='promax')
fa_om_loadings.to_excel(writer, sheet_name='oblimin')
fa_qmo_loadings.to_excel(writer, sheet_name='quartimin')

writer.save()


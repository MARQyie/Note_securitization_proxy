#--------------------------------------------
# Factor analisys for Note
# Mark van der Plaat
# July 2021 -- Update: November 2021
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
from matplotlib.ticker import MaxNLocator
import seaborn as sns
sns.set(style = 'whitegrid', font_scale = 3)

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
var_names = ['cr_as_sbo','cr_as_rmbs','cr_as_ccr','cr_as_auto',\
             'cr_as_abs_oth','hmda_sec_amount',\
             'cr_sec_income','cr_serv_fees',\
             'cr_cds_purchased',\
             'cr_abcp_uc_own','cr_abcp_ce_own',\
             'cr_abcp_uc_oth','cr_abcp_ce_oth']
var_labels = ['SBO Sold','Res. Assets Sold, Sec.',\
              'CC Assets Sold, Sec.','Auto Assets Sold, Sec.',\
              'Other Assets Sold, Sec.','Securitized (HMDA)',\
              'Sec. Income', 'Net Servicing Fees',\
              'CDSs Purchased','Unused Com. ABCP (Own)',\
              'Credit Exp. ABCP (Own)','Unused Com. ABCP (Others)',\
              'Credit Exp. ABCP (Others)']
# Subset data to only needed variables
df_sub = df[var_names]

# First log the data (to reduce skewness in all but the income variables)
df_log = np.log(df_sub - df_sub.min() + 1) 

# standardize data
df_standard = pd.DataFrame(preprocessing.scale(df_log), columns = var_names)

#--------------------------------------------
# Check if multivariate normal distribution 
#--------------------------------------------

# Shapiro-Wilk, D’Agostino’s K^2, and  Anderson-Darling Test
from scipy.stats import shapiro, normaltest, anderson
lst_swtest = np.zeros((len(var_names),2))
lst_k2test = np.zeros((len(var_names),2))
lst_adtest = np.zeros((len(var_names),2))

for i in range(len(var_names)):
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
bartlett_chi, bartlett_p = calculate_bartlett_sphericity(df_standard) # p = 0.0

# Kaiser-Meyer-Olkin (KMO) test. Measures data suitability; should be between 0 and 1, but above 0.5
kmo_all, kmo_model = calculate_kmo(df_standard) #kmo_model = 0.8173515125363751

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
fa.fit(df_standard)
ev, v = fa.get_eigenvalues()

# Perform a parallel analysis
list_ev_rand = []

np.random.seed(1)
for i in range(100):
    df_rand = pd.DataFrame(np.random.rand(*df_standard.shape))
    fa_rand = FactorAnalyzer(rotation = None, method = 'principal').fit(df_rand)
    ev_rand, _ = fa_rand.get_eigenvalues()
    list_ev_rand.append(ev_rand)

# Get intersect of two lists
ev_rand = np.mean(list_ev_rand,axis = 0)
int_intersect = sum(ev > ev_rand)
ev_greater_one = sum(ev > 1)
    
# Plot
fig, ax = plt.subplots(figsize=(15,9)) 
ax.set(ylabel='Eigenvalue', xlabel = 'Factor')
ax.plot(range(1, df_standard.shape[1] + 1), ev, marker = 'o', label = 'Factor')
ax.plot(range(1, df_standard.shape[1] + 1), ev_rand, color = 'black', linestyle = '--', marker = "x", label = 'Parallel Analysis')
ax.axvline(ev_greater_one, color = 'red', alpha = 0.75, label = 'Factor > 1')
ax.axvline(int_intersect, color = 'red', linestyle = '--', alpha = 0.75,  label = 'Factor > PA')
ax.legend()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()

fig.savefig('Figures/Dimension_reduction/FA_parallel.png')

'''NOTE: First 4 factors have an eigen value greater than 1.
    Parallel analysis: ev_parallel is slightly above ev_efa: 
        N factors is 3
    
    Conclusion: 3 factors is appropriate, which is a more conservative number'''
#--------------------------------------------
# Factor rotation all variables
# Note: No orthogonal rotations, since the orth. assumption is unrealistic
# See Brown (2015)
#--------------------------------------------
    
# Get factor estimates
fa = FactorAnalyzer(rotation = None, n_factors = 3, method = 'principal')
fa.fit(df_standard)

# Get the non-rotated factor loadings
fa_loadings = pd.DataFrame(fa.loadings_,\
                           columns = range(3),\
                           index = var_names)

fa_loadings.to_csv('Results/fa_loadings_norotation.csv')

#--------------------------------------------
# Oblique rotations

# Promax 
fa_pm = FactorAnalyzer(rotation = 'promax', n_factors = 3, method = 'principal')
fa_pm.fit(df_standard)

## Get the factor loadings 
fa_pm_loadings = pd.DataFrame(fa_pm.loadings_,\
                           columns = range(3),\
                           index = var_names)

fa_pm_loadings.to_csv('Results/fa_loadings_promax.csv')

# Oblimin
fa_om = FactorAnalyzer(rotation = 'oblimin', n_factors = 3, method = 'principal')
fa_om.fit(df_standard)

## Get the factor loadings 
fa_om_loadings = pd.DataFrame(fa_om.loadings_,\
                           columns = range(3),\
                           index = var_names)

fa_om_loadings.to_csv('Results/fa_loadings_oblimin.csv')

# Quartimin
fa_qmo = FactorAnalyzer(rotation = 'quartimin', n_factors = 3, method = 'principal')
fa_qmo.fit(df_standard)

## Get the factor loadings 
fa_qmo_loadings = pd.DataFrame(fa_qmo.loadings_,\
                           columns = range(3),\
                           index = var_names)

fa_qmo_loadings.to_csv('Results/fa_loadings_quartimin.csv')

'''NOTE 
    Cutoff: x > .4 (cf. Brown, 2015: CFA for Applied Research p. 27).
    We base ourselfs on the promax rotation, other oblique rotations might
    lead to other results.
        
    Badly behaving indicators (low loadings)            : cr_ls_income, hmda_sec_amount
    Badly behaving indicators (multiple high loadings)  : cr_as_abs
    Badly identified factors (only 1--2 salient loading) : F4
        
    Interpretation Factors:
        F0: CDO/ABCP Securitization
        F1: Loan Sales
        F2: Credit Exposure Others
        F3: Non-residential asset sales
        F4: UNCLEAR
    
    Conclusion: Interpretation of the factors is straight forward. Two
    factors is probably most appropriate based on the loadings. cr_as_sbo
    probably not needed in follow-up CFA, as are the variables for credit exposure
    
    WARNING: The data contains many zeros. The PAF method might not be
    robust enough.
    '''
#--------------------------------------------
# Do promax with five
fa_pm5 = FactorAnalyzer(rotation = 'promax', n_factors = 4, method = 'principal')
fa_pm5.fit(df_standard)

## Get the factor loadings 
fa_pm5_loadings = pd.DataFrame(fa_pm5.loadings_,\
                           columns = range(4),\
                           index = var_names)
    
'''OLD
#--------------------------------------------
# Do promax with three
# First run with three and all variables, then delete badly behaving indicators
fa_pm2 = FactorAnalyzer(rotation = 'promax', n_factors = 2, method = 'principal')
fa_pm2.fit(df_standard)

## Get the factor loadings 
fa_pm2_loadings = pd.DataFrame(fa_pm2.loadings_,\
                           columns = range(2),\
                           index = var_names)

# Get list with only good behaving indicators
var_names_good = [var for var in var_names if var not in ['cr_serv_fees', 'hmda_sec_amount',\
                                                          'cr_ls_income', \
                                                          'cr_sec_income','cr_as_sbo']]
# Promax
fa_pm2 = FactorAnalyzer(rotation = 'promax', n_factors = 2)
fa_pm2.fit(df_standard[var_names_good])

## Get the factor loadings 
fa_pm2_loadings = pd.DataFrame(fa_pm2.loadings_,\
                           columns = range(2),\
                           index = var_names_good)
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
# Save Promax tables to Latex
#--------------------------------------------

# Rename columns
fa_pm_loadings.columns = pd.MultiIndex.from_product([['Three Factor Model',],['F1','F2','F3']])
fa_pm5_loadings.columns = pd.MultiIndex.from_product([['Four Factor Model',],['F1','F2','F3','F4']])

# Combine 4 and 2 factor models
fa_table = pd.concat([fa_pm_loadings,fa_pm5_loadings], axis = 1)

# rename index
fa_table.index = var_labels

# Round numbers to 4 decimals and make boldface if > .4
fa_table = fa_table.round(4)
fa_table = fa_table.applymap(lambda x: '\\textbf{' + str(x) + '}' if x > .4 else '\\textit{' + str(x) + '}' if .3 < x <= .4 else str(x))

# Replace nan with ''
fa_table = fa_table.replace('nan', '')

# Prelims
column_format = 'p{4cm}' + 'p{1cm}' * fa_table.shape[1]
caption = ('Results Explanatory Factor Analysis: Three and Four Factor Models')
notes = '\\multicolumn{8}{p{14cm}}{\\textit{Notes.} Factor loadings based on the oblique promax rotation. The explanatory factor models are estimated by a principal factor algorithm implemented by FactorAnalyzer in Python. The principal factor algorithm is more robust to deviations from normal than a maximum likelihood algorithm. The first three columns contain the factor loadings from a three factor model with all proxies. The next four columns present the factor loadings of a four factor model. The number of original factors are determined by a parallel analysis and the rule of thumb eigenvalue $>0$. Factor loadings are in boldface if they are greater than 0.4, and in italics if they are between 0.3 and 0.4.} \n'
label = 'tab:efa_table'
position = 'th'
string_size = '\\scriptsize\n'

# To latex
fa_table_latex = fa_table.to_latex(column_format = column_format,\
                                   longtable = False, caption = caption,\
                                   label = label, position = position,\
                                   multicolumn = True, multicolumn_format = 'c',\
                                   escape = False)
    
# add notes to bottom of the table
location_mid = fa_table_latex.find('\end{tabular}')
fa_table_latex = fa_table_latex[:location_mid] + notes + fa_table_latex[location_mid:]

# adjust sting size
location_size = fa_table_latex.find('\centering\n')
fa_table_latex = fa_table_latex[:location_size + len('\centering\n')] + string_size + fa_table_latex[location_size + len('\centering\n'):]

# Save
text_fa_table_latex_latex = open('Results/EFA_4factors.tex', 'w')
text_fa_table_latex_latex.write(fa_table_latex)
text_fa_table_latex_latex.close()

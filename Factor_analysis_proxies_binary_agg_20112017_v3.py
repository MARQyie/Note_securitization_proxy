#--------------------------------------------
# Exploratory Factor analisys for Note
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
df = pd.read_csv('Data\df_sec_note_binary_agg_20112017.csv', index_col = 0)

# Make net cd variable
#df['cr_cd_net'] = df.cr_cd_purchased - df.cr_cd_sold

# set variable names
var_names = ['cr_as_sbo',
             'cr_as_rmbs',
             'cr_as_abs',
             'hmda_sec_amount',
             'cr_secveh_ta',
             'cr_cds_purchased',
             'cr_abcp_ta',
             'cr_abcp_uc_own',
             'cr_abcp_ce_own',
             'cr_abcp_uc_oth']
var_labels = ['Small Bus. Obl. Transf.',
              'Sec. Residential Loans',
              'Sec. Other Assets',
              'Sec. Residential Mortgages',
              'TA Sec. Vehicles' ,
              'CDSs Purchased',
              'TA ABCP Conduits',
              'Un. Com. Own ABCP Conduits',
              'Credit Exp. Own ABCP Conduits',
              'Un. Com. Other ABCP Conduits']
# Subset data to only needed variables
df_sub = df[var_names]

#--------------------------------------------
# Determine number of factors
#--------------------------------------------

'''We use multiple selection criteria:
    1) Scree plot (elbow plot)
    2) Kaiser-Guttman rule
    3) Parallel analysis
    '''
    
# Get factor estimates
fa = FactorAnalyzer(rotation = None,
                    method = 'principal')
fa.fit(df_sub)
ev, v = fa.get_eigenvalues()

# Perform a parallel analysis
list_ev_rand = []

np.random.seed(1)
for i in range(100):
    df_rand = pd.DataFrame(np.random.normal(size = df_sub.shape))
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
ax.plot(range(1, df_sub.shape[1] + 1), ev, marker = 'o', label = 'Factor')
ax.plot(range(1, df_sub.shape[1] + 1), ev_rand, color = 'black', linestyle = '--', marker = "x", label = 'Parallel Analysis')
ax.axvline(ev_greater_one, color = 'red', alpha = 0.75, label = 'Eigenvalue > 1')
ax.axvline(int_intersect, color = 'red', linestyle = '--', alpha = 0.75,  label = 'Cut-off Par. Analysis')
ax.legend()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()

fig.savefig('Figures/Dimension_reduction/FA_parallel_binary_agg_20112017.png')

'''NOTE: First 4 factors have an eigen value greater than 1.
    Parallel analysis: ev_parallel is slightly above ev_efa: 
        N factors is 3
    
    Conclusion: 3 factors is appropriate, which is a more conservative number'''
#--------------------------------------------
# Factor rotation all variables
# Note: No orthogonal rotations, since the orth. assumption is unrealistic
# See Brown (2015)
#--------------------------------------------
    
#--------------------------------------------
# Oblique rotations

# Promax 
fa_pm = FactorAnalyzer(rotation = 'promax', n_factors = 1, method = 'principal')
fa_pm.fit(df_sub)

## Get the factor loadings 
fa_pm_loadings = pd.DataFrame(fa_pm.loadings_,
                           columns = range(1),
                           index = var_names)

fa_pm_loadings.to_csv('Results/fa_binary_agg_loadings_promax_20112017.csv')


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
fa_pm5 = FactorAnalyzer(rotation = 'promax', n_factors = 3, method = 'principal')
fa_pm5.fit(df_sub)

## Get the factor loadings 
fa_pm5_loadings = pd.DataFrame(fa_pm5.loadings_,
                           columns = range(3),
                           index = var_names)
    
#--------------------------------------------
# Save Promax tables to Latex
#--------------------------------------------

# Rename columns
fa_pm_loadings.columns = pd.MultiIndex.from_product([['One Factor Model',],['F1']])
fa_pm5_loadings.columns = pd.MultiIndex.from_product([['Three Factor Model',],['F2','F1','F3']])

# Combine 4 and 2 factor models
fa_table = pd.concat([fa_pm_loadings['One Factor Model'][['F1']],\
                      fa_pm5_loadings['Three Factor Model'][['F1','F2','F3']]], axis = 1)

# rename index
fa_table.index = var_labels

# Round numbers to 4 decimals and make boldface if > .4
fa_table = fa_table.applymap('{:,.4f}'.format)
fa_table = fa_table.applymap(lambda x: '\\textbf{' + x + '}' if float(x) > .4 else '\\textit{' + x + '}' if .3 < float(x) <= .4 else x)

# Replace nan with ''
fa_table = fa_table.replace('nan', '')

# Prelims
column_format = 'p{4.75cm}' + 'p{1cm}' * fa_table.shape[1]
caption = ('Results Explanatory Factor Analysis: One and Three Factor Models')
notes = '\\multicolumn{6}{p{11.75cm}}{\\textit{Notes.} Factor loadings based on the oblique promax rotation. The explanatory factor models are estimated by a principal factor algorithm implemented by FactorAnalyzer in Python. The principal factor algorithm is more robust to deviations from normal than a maximum likelihood algorithm. The first two columns contain the factor loadings from a two factor model with all proxies. The next three columns present the factor loadings of a three factor model. The number of original factors are determined by a parallel analysis and the rule of thumb eigenvalue $>0$. Factor loadings are in boldface if they are greater than 0.4, and in italics if they are between 0.3 and 0.4.} \n'
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
text_fa_table_latex_latex = open('Results/EFA_binary_agg_20112017.tex', 'w')
text_fa_table_latex_latex.write(fa_table_latex)
text_fa_table_latex_latex.close()

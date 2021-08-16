#--------------------------------------------
# Summary Statistics for Note
# Mark van der Plaat
# September 2020 -- Update: March 2021
#--------------------------------------------

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
sns.set(style = 'whitegrid', font_scale = 2)

# Set WD
import os
os.chdir(r'D:\RUG\PhD\Materials_papers\01-Note_on_securitization')

#--------------------------------------------
# Load data 
#--------------------------------------------

df = pd.read_csv('Data\df_sec_note.csv', index_col = 0)

#--------------------------------------------
# Prelims
#--------------------------------------------

# Variable names
var_names = ['cr_as_nsres','cr_as_nsoth','hmda_gse_amount','hmda_priv_amount',\
             'cr_ls_income','cr_as_rmbs','cr_as_abs','hmda_sec_amount',\
             'cr_sec_income','cr_as_sbo',\
             'cr_cds_purchased','cr_trs_purchased','cr_co_purchased',\
             'cr_abcp_uc_own','cr_abcp_ce_own','cr_abcp_uc_oth','cr_abcp_ce_oth',\
                 'cr_serv_fees']
var_labels = ['Res. Assets Sold, Not Sec.','Other Assets Sold, Not Sec.',\
              'Sold To GSE (HMDA)','Sold to Private (HMDA)',\
              'Loan Sales Income','Res. Assets Sold, Sec.',\
              'Other Assets Sold, Sec.','Securitized (HMDA)',\
              'Sec. Income','SBO Sold',\
              'CDSs Purchased','TRSs Purchased',\
              'COs Purchased','Unused Com. ABCP (Own)',\
              'Credit Exp. ABCP (Own)','Unused Com. ABCP (Others)',\
              'Credit Exp. ABCP (Others)', 'Net Servicing Fees']
#--------------------------------------------
# Input Correlation map
#--------------------------------------------
# Set heatmap function 
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
    
# Get correlation matrix and p-vals
corr = df[var_names].corr()
corr_pval = stats.spearmanr(df[var_names])[1] 

# Set nice names
corr.index = var_labels
corr.columns = var_labels

heatmap(corr, corr_pval, 'Corr_input_pearson.png')

#--------------------------------------------
# Summary Statistics (Mean, SDs, quantiles)
#--------------------------------------------

# Get mean, SDs and quartiles
sumstats = df[var_names].agg(['mean','std']).T
sumstats[['.5','.75','.95','.99']] = df[var_names].quantile([.5,.75,.95,.99]).T

# rename columns and index
sumstats.columns = ['Mean','SD','Median','75\%','95\%','99\%']
sumstats.index = var_labels

# Round to one decimal
sumstats = sumstats.round(1).astype(str)

#--------------------------------------------
# Count table securitization variables
#--------------------------------------------

# Make count table
num_sec = df[['date'] + var_names].groupby('date').apply(lambda x: np.sum(abs(x)>0)).T

# Change column  and index labels
num_sec.columns = num_sec.columns.tolist()
num_sec.index = ['N'] + var_labels

#--------------------------------------------
# Tables to latex
#--------------------------------------------

# Set function
def resultsToLatex(results, caption = '', label = '', size_string = '\\scriptsize \n',  note_string = None, sidewaystable = False):
    # Prelim
    if results.shape == (18,6):
        function_parameters = dict(na_rep = '',
                               index_names = True,
                               column_format = 'p{4cm}' + 'p{1.5cm}' * results.shape[1],
                               escape = False,
                               multicolumn = True,
                               multicolumn_format = 'c',
                               caption = caption,
                               label = label)
  
    else:
        function_parameters = dict(na_rep = '',
                               index_names = True,
                               column_format = 'p{4cm}' + 'p{.6cm}' * results.shape[1],
                               escape = False,
                               multicolumn = True,
                               multicolumn_format = 'c',
                               caption = caption,
                               label = label)
  
    # To Latex
    latex_table = results.to_latex(**function_parameters)
       
    # Add string size
    location_size = latex_table.find('\centering\n')
    latex_table = latex_table[:location_size + len('\centering\n')] + size_string + latex_table[location_size + len('\centering\n'):]
    
    # Add note to the table
    if note_string is not None:
        location_note = latex_table.find('\end{tabular}\n')
        latex_table = latex_table[:location_note + len('\end{tabular}\n')]\
            + '\\begin{tablenotes}\n\\scriptsize\n\\item ' + note_string + '\\end{tablenotes}\n' + latex_table[location_note + len('\end{tabular}\n'):]
            
    # Add midrule above 'Observations'
    # if latex_table.find('N                     &') >= 0:
    #     size_midrule = '\\midrule '
    #     location_mid = latex_table.find('N                     &')
    #     latex_table = latex_table[:location_mid] + size_midrule + latex_table[location_mid:]
           
    # Makes sidewaystable
    if sidewaystable:
        latex_table = latex_table.replace('{table}','{sidewaystable}',2)
        
    # Make threeparttable
    location_centering = latex_table.find('\centering\n')
    latex_table = latex_table[:location_centering + len('\centering\n')] + '\\begin{threeparttable}\n' + latex_table[location_centering + len('\centering\n'):]
    
    location_endtable = latex_table.find('\\end{tablenotes}\n')
    latex_table = latex_table[:location_endtable + len('\\end{tablenotes}\n')] + '\\end{threeparttable}\n' + latex_table[location_endtable + len('\\end{tablenotes}\n'):]
        
    return latex_table

# Set function parameters
## Sum stats
caption_sumstats = 'Summary Statistics Securitization Proxies'
label_sumstats  = 'tab:summary_statistics_proxies'
size_string_sumstats = '\\scriptsize \n'
note_sumstats = "\\textit{Notes.} Summary statistics of the securitization proxies. All numbers are in thousands USD. 75\%, 95\% and 99\% are the 75th, 95th and 99th percentile, respectively."

## Num Sec
caption_numsec = 'Number of Securitizers Per Proxy'
label_numsec  = 'tab:number_securitizers'
size_string_numsec = '\\scriptsize \n'
note_numsec = "\\textit{Notes.} The number of securitizers per proxy per year."

# Make latex tables
sumstats_latex = resultsToLatex(sumstats, caption_sumstats, label_sumstats,\
                                 size_string = size_string_sumstats, note_string = note_sumstats,\
                                 sidewaystable = False)

numsec_latex = resultsToLatex(num_sec, caption_numsec, label_numsec,\
                                 size_string = size_string_numsec, note_string = note_numsec,\
                                 sidewaystable = False)

# Save to latex and excel
## Sum stats
sumstats.to_excel('Tables/Summary_statistics.xlsx')

text_ss_latex = open('Tables/Summary_statistics.tex', 'w')
text_ss_latex.write(sumstats_latex)
text_ss_latex.close()

## Num sec
num_sec.to_excel('Tables/Number_securitizers.xlsx')

text_num_sec_latex = open('Tables/Number_securitizers.tex', 'w')
text_num_sec_latex.write(numsec_latex)
text_num_sec_latex.close()

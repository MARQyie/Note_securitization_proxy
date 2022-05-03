#--------------------------------------------
# Summary Statistics for Note
# Mark van der Plaat
# September 2020 -- Update: March 2022
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
import matplotlib
import seaborn as sns

sns.set(style = 'whitegrid', font_scale = 2.75)

# Set WD
import os
os.chdir(r'D:\RUG\PhD\Materials_papers\01-Note_on_securitization')

#--------------------------------------------
# Load data 
#--------------------------------------------

df = pd.read_csv('Data\df_sec_note_binary_20112017.csv', index_col = 0)

#--------------------------------------------
# Prelims
#--------------------------------------------

# Variable names
var_names = ['cr_as_sbo','cr_as_rmbs',\
             'cr_as_abs','hmda_sec_amount',\
            'cr_secveh_ta',\
             'cr_cds_purchased','cr_abcp_ta',\
             'cr_abcp_uc_own','cr_abcp_ce_own',\
             'cr_abcp_uc_oth','cr_abcp_ce_oth']
var_labels = ['Small Bus. Obl. Transf.','Sec. Residential Loans',\
              'Sec. Other Assets','Sec. Residential Mortgages',\
              'TA Sec. Vehicles' ,\
              'CDSs Purchased','TA ABCP Conduits',\
              'Un. Com. Own ABCP Conduits',\
              'Credit Exp. Own ABCP Conduits','Un. Com. Other ABCP Conduits',\
              'Credit Exp. Other ABCP Conduits']
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
                      'cmap': 'coolwarm',
                      'fmt':'.3f'}
    dic_aes_unmasked = {'mask' : pvals <= 0.05, 
                      'annot' : annot,
                      'vmin' : -1,
                      'vmax' : 1,
                      'center': 0,
                      'cbar': False,
                      'cmap': 'coolwarm',
                      'fmt':'.3f'}
    
    # Make heatmap
    fig, ax = plt.subplots(figsize=(36,24))
    sns.heatmap(matrix, **dic_aes_unmasked)
    sns.heatmap(matrix, **dic_aes_masked)
    plt.tight_layout()
    
    # Save heatmap
    fig.savefig('Figures/Correlation_maps/' + file)
    
# Get correlation matrix and p-vals
corr = df[var_names[:-1]].corr(method = 'pearson')
corr_pval = stats.spearmanr(df[var_names[:-1]])[1]

# Set nice names
corr.index = var_labels[:-1]
corr.columns = var_labels[:-1]

heatmap(corr, corr_pval, 'Corr_binary_input_pearson.png')

#--------------------------------------------
# Summary Statistics (Mean, SDs, quantiles)
#--------------------------------------------

# Get mean, SDs and quartiles
sumstats = df[var_names].agg(['mean','std']).T

# rename columns and index
sumstats.columns = ['Mean','SD']
sumstats.index = var_labels

# Round to 4 decimal with decimal separator
sumstats = sumstats.applymap('{:,.4f}'.format)

#--------------------------------------------
# Count table securitization variables
#--------------------------------------------

# Make count table
num_sec = df[['date'] + var_names].groupby('date').apply(lambda x: np.sum(abs(x)>0)).T

# Change column  and index labels
num_sec.columns = num_sec.columns.tolist()
num_sec.index = ['N'] + var_labels

#--------------------------------------------
# Recurring securitization
#--------------------------------------------

# Get data
data_rec = df.groupby(['IDRSSD','date'])[var_names].sum().sum(axis = 1)
rec_counts = data_rec.value_counts()

# Plot
x = range(len(rec_counts[1:]))
fig, ax = plt.subplots(figsize = (12,10))
ax.set(ylabel = 'Number of bank-years', xlabel = 'Number of securitization variables reported')
plt.bar(x,rec_counts[1:])
ax.set_xticks(x)
plt.tight_layout()
plt.savefig('Figures/Plot_number_securitizationtypes.png')

#--------------------------------------------
# Tables to latex
#--------------------------------------------

# Set function
def resultsToLatex(results, caption = '', label = '', size_string = '\\scriptsize \n',  note_string = None, sidewaystable = False):
    # Prelim
    if results.shape == (11,2):
        function_parameters = dict(na_rep = '',
                               index_names = True,
                               column_format = 'p{4.75cm}' + 'p{1.5cm}' * results.shape[1],
                               escape = False,
                               multicolumn = True,
                               multicolumn_format = 'c',
                               caption = caption,
                               label = label)
  
    else:
        function_parameters = dict(na_rep = '',
                               index_names = True,
                               column_format = 'p{4.75cm}' + 'p{.5cm}' * results.shape[1],
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
note_sumstats = "\\textit{Notes.} Summary statistics of the securitization proxies. 75\%, 95\% and 99\% are the 75th, 95th and 99th percentile, respectively."

## Num Sec
caption_numsec = 'Number of Securitizers Per Proxy'
label_numsec  = 'tab:number_securitizers'
size_string_numsec = '\\scriptsize \n'
note_numsec = "\\textit{Notes.} This table shows the number of securitizers per proxy per year"

# Make latex tables
sumstats_latex = resultsToLatex(sumstats, caption_sumstats, label_sumstats,
                                 size_string = size_string_sumstats, note_string = note_sumstats,
                                 sidewaystable = False)

numsec_latex = resultsToLatex(num_sec, caption_numsec, label_numsec,
                                 size_string = size_string_numsec, note_string = note_numsec,
                                 sidewaystable = False)

# Save to latex and excel
## Sum stats
text_ss_latex = open('Tables/Summary_statistics_binary.tex', 'w')
text_ss_latex.write(sumstats_latex)
text_ss_latex.close()

## Num sec
text_num_sec_latex = open('Tables/Number_securitizers.tex', 'w')
text_num_sec_latex.write(numsec_latex)
text_num_sec_latex.close()
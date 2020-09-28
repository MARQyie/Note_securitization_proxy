#--------------------------------------------
# Summary Statistics for Note
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
vars_tot = vars_cr + vars_hmda + ['RC2170']
vars_tot.remove('cr_ta_vie')

#--------------------------------------------
# Summary Statistics
#--------------------------------------------

# Get Summary Statistics
ss = df[vars_tot].describe().T[['mean','std']]

## round and add thousand seperators
ss.iloc[:,0] = ss.iloc[:,0].round(2).apply(lambda x : "{:,}".format(x))
ss.iloc[:,1] = ss.iloc[:,1].round(2).apply(lambda x : "{:,}".format(x))

# Change column names
ss.columns = ['Mean','SD']

# Change row names
row_names = ['Securitization Income','Loans Pending Securitization','Credit Derivatives Sold',\
             'Credit Derivatives Purchased','On-Balance Sheet Exposure','On-Balance Sheet Exposure',\
             'Assets Sold and Securitized','Asset Sold and Not Securitized',\
             'Credit Exposure Other Parties','Total Asset Sec. Vehicles','Total Assets ABCP Conduits',\
             'Total Assets Other','HDMA Sold To GSE','HMDA Sold to Private',\
             'HMDA Securitized','Total Assets']
ss.index = row_names

#------------------------------------------------------------
# To Latex
#------------------------------------------------------------

# Set function
def resultsToLatex(results, caption = '', label = '', size_string = '\\scriptsize \n',  note_string = None, sidewaystable = False):
    # Prelim
    function_parameters = dict(na_rep = '',
                               index_names = True,
                               column_format = 'p{4cm}' + 'p{1cm}' * results.shape[1],
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
    if latex_table.find('N                     &') >= 0:
        size_midrule = '\\midrule '
        location_mid = latex_table.find('N                     &')
        latex_table = latex_table[:location_mid] + size_midrule + latex_table[location_mid:]
           
    # Makes sidewaystable
    if sidewaystable:
        latex_table = latex_table.replace('{table}','{sidewaystable}',2)
        
    # Make threeparttable
    location_centering = latex_table.find('\centering\n')
    latex_table = latex_table[:location_centering + len('\centering\n')] + '\\begin{threeparttable}\n' + latex_table[location_centering + len('\centering\n'):]
    
    location_endtable = latex_table.find('\\end{tablenotes}\n')
    latex_table = latex_table[:location_endtable + len('\\end{tablenotes}\n')] + '\\end{threeparttable}\n' + latex_table[location_endtable + len('\\end{tablenotes}\n'):]
        
    return latex_table

# Call function
caption = 'Summary Statistics'
label = 'tab:summary_statistics'
size_string = '\\tiny \n'
note = "\\textit{Notes.} Summary statistics of the securitization proxies."

ss_latex = resultsToLatex(ss, caption, label,\
                                 size_string = size_string, note_string = note,\
                                 sidewaystable = False)

#------------------------------------------------------------
# Save
#------------------------------------------------------------

ss.to_excel('Tables/Summary_statistics.xlsx')

text_ss_tot_latex = open('Tables/Summary_statistics.tex', 'w')
text_ss_tot_latex.write(ss_latex)
text_ss_tot_latex.close()

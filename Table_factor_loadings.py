#--------------------------------------------
# Table Factor analisys for Note
# Mark van der Plaat
# September 2020
#--------------------------------------------

''' Loads the Quartimax and Varimax rotation tables and returns a 
    LaTeX table
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
df_q = pd.read_csv('Results/fa_loadings_quartimax_sec.csv', index_col = 0)
df_p = pd.read_csv('Results/fa_loadings_promax_sec.csv', index_col = 0)

#--------------------------------------------
# Make tables nice
#--------------------------------------------

# Set row labels and column labels
# TODO: change row order
row_names = ['Net Servicing Fees','Securitization Income','Loan Sales Income','Credit Derivatives Sold',\
             'Credit Derivatives Purchased',\
             'Assets Sold and Securitized','Asset Sold and Not Securitized',\
             'Credit Exposure Other Parties','Total Asset Securitization Vehicles','Total Assets ABCP Conduits',\
             'Total Assets Other VIEs','HDMA Sold To GSE','HMDA Sold to Private',\
             'HMDA Securitized']

col_names = [('Quartimax Rotation','F1'), ('Quartimax Rotation','F2'), ('Quartimax Rotation','F3'),\
             ('Quartimax Rotation','F4'), ('Promax Rotation','F1'), ('Promax Rotation','F2'),\
             ('Promax Rotation','F3'), ('Promax Rotation','F4')]

# Make new table
df = pd.concat([df_q,df_p], axis = 1)

## add index and columns
df.index = row_names
df.columns = pd.MultiIndex.from_tuples(col_names)

## Round to 4 decimals
df = df.round(4)

# Give boldface to loadings >0.5
def boldLoading(loading, cutoff = 0.5):
    if abs(loading) > cutoff:
        return '\textbf{' + str(loading) + '}'
    else:
        return str(loading)
    
df_bold = df.applymap(boldLoading)

#------------------------------------------------------------
# To Latex
#------------------------------------------------------------

# Set function
def resultsToLatex(results, caption = '', label = '', size_string = '\\scriptsize \n',  note_string = None, sidewaystable = False):
    # Prelim
    function_parameters = dict(na_rep = '',
                               index_names = True,
                               column_format = 'p{4.5cm}' + 'p{1cm}' * results.shape[1],
                               escape = False,
                               multicolumn = True,
                               multicolumn_format = 'c',
                               caption = caption,
                               label = label)
  
    # To Latex
    latex_table = results.to_latex(**function_parameters)
    
    # Add table placement
    location_place = latex_table.find('\begin{table}\n')
    latex_table = latex_table[:location_place + len('\begin{table}\n') + 1] + '[h]' + latex_table[location_place + len('\begin{table}\n') + 1:]
       
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
caption = 'Factor Loadings Quartimax and Promax Rotations'
label = 'tab:factor_loadings'
size_string = '\\scriptsize \n'
note = "\\textit{Notes.} Factor loadings of the first Four factors. "

ss_latex = resultsToLatex(df_bold, caption, label,\
                                 size_string = size_string, note_string = note,\
                                 sidewaystable = False)

#------------------------------------------------------------
# Save
#------------------------------------------------------------

df.to_excel('Tables/Factor_loadings.xlsx')

text_ss_tot_latex = open('Tables/Factor_loadings.tex', 'w')
text_ss_tot_latex.write(ss_latex)
text_ss_tot_latex.close()

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
vars_tot = vars_cr + vars_hmda + ['ta']

#--------------------------------------------
# Summary Statistics Securitization
#--------------------------------------------

# Get Summary Statistics
ss = df[vars_tot].describe().T[['mean','std']]

## round and add thousand seperators
ss.iloc[:,0] = ss.iloc[:,0].round(2).apply(lambda x : "{:,}".format(x))
ss.iloc[:,1] = ss.iloc[:,1].round(2).apply(lambda x : "{:,}".format(x))

# Change column names
ss.columns = ['Mean','SD']

# Change row names
row_names = ['Securitization Income','Credit Derivatives Sold',\
             'Credit Derivatives Purchased',\
             'Assets Sold and Securitized','Asset Sold and Not Securitized',\
             'Credit Exposure Other Parties','Total Asset Securitization Vehicles','Total Assets ABCP Conduits',\
             'Total Assets Other VIEs','HDMA Sold To GSE','HMDA Sold to Private',\
             'HMDA Securitized','Total Assets']
ss.index = row_names

#--------------------------------------------
# Structure of the securitization data
#--------------------------------------------

# Count number of securitizers per year
num_sec = df[['date'] + vars_tot].groupby(df.date).apply(lambda x: np.sum(x>0))

# Get percentages
perc_sec = num_sec.apply(lambda x: round(x[1:] / x[0] * 100, 2), axis = 1)

# Combine number and percentage
numperc_sec = pd.DataFrame(num_sec.iloc[:,0])

for col in range(perc_sec.shape[1]):
    numperc_sec.insert(col + 1, num_sec.iloc[:,col + 1].name, num_sec.iloc[:,col + 1].astype(str) + ' (' + perc_sec.iloc[:,col].astype(str) + '\%)', True)

# Change column labels
num_sec.columns = ['N'] + row_names
numperc_sec.columns = ['N'] + row_names

# remove index name
num_sec.index.name = None
numperc_sec.index.name = None

#--------------------------------------------
# Summary Statistics Other variables
#--------------------------------------------

# Set Variables needed
vars_oth = df_oth.columns[2:].tolist()
vars_oth.remove('ta')

# Get Summary Statistics
ss_oth = df[vars_oth].describe().T[['mean','std']]

## round and add thousand seperators
ss_oth.iloc[:,0] = ss_oth.iloc[:,0].round(4)
ss_oth.iloc[:,1] = ss_oth.iloc[:,1].round(4)

ss_oth.iloc[0,0] = "{:,}".format(ss_oth.iloc[0,0].round(2))
ss_oth.iloc[0,1] = "{:,}".format(ss_oth.iloc[0,1].round(2))

# Change column names
ss_oth.columns = ['Mean','SD']

# Change row names
row_names_oth = ['Operational income', 'Liquidity Ratio',\
            'Trading asset ratio', 'Loan Ratio',\
            'Loans-to-deposits', 'Deposit ratio',\
            'Capital ratio', 'Real estate loan ratio',\
            'Commercial and industrial loan ratio', 'Agricultural loan ratio',\
            'Consumer loan ratio', 'Other loan ratio',\
            'loan HHI', 'Tier 1 leverage ratio',\
            'Tier 1 capital ratio', 'Total regulatory capital ratio',\
            'RWA/TA', 'NPL Ratio',\
            'Charge-off ratio', 'Allowance ratio',\
            'Provision ratio', 'Interest expense / Total liabilities',\
            'Interest expense deposits / Total deposits',\
            'Return on equity', 'Return on assets',\
            'Net interest margin', 'Cost to income',\
            'Revenue HHI', 'Non-insterest income / net operating revenue']
ss_oth.index = row_names_oth

#------------------------------------------------------------
# To Latex
#------------------------------------------------------------

# Set function
def resultsToLatex(results, caption = '', label = '', size_string = '\\scriptsize \n',  note_string = None, sidewaystable = False):
    # Prelim
    if results.shape == (14,9):
        function_parameters = dict(na_rep = '',
                               index_names = True,
                               column_format = 'p{4.5cm}' + 'p{.65cm}' * results.shape[1],
                               escape = False,
                               multicolumn = True,
                               multicolumn_format = 'c',
                               caption = caption,
                               label = label)
  
    else:
        function_parameters = dict(na_rep = '',
                               index_names = True,
                               column_format = 'p{5cm}' + 'p{2cm}' * results.shape[1],
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
## Securitization summary statistics
caption = 'Summary Statistics Securitization Proxies'
label = 'tab:summary_statistics_proxies'
size_string = '\\footnotesize \n'
note = "\\textit{Notes.} Summary statistics of the securitization proxies. All numbers are in thousands USD."

ss_latex = resultsToLatex(ss, caption, label,\
                                 size_string = size_string, note_string = note,\
                                 sidewaystable = False)

# Number and percentage securitization
caption_num_sec = 'Number of Securitizers Per Proxy'
label_num_sec = 'tab:number_securitizers'
size_string_num_sec = '\\scriptsize \n'
note_num_sec = "\\textit{Notes.} Total is the number of banks in the sample per year."

num_sec_latex = resultsToLatex(num_sec.T, caption_num_sec, label_num_sec,\
                                 size_string = size_string_num_sec, note_string = note_num_sec,\
                                 sidewaystable = False)

caption_numperc_sec = 'Number and Percentage of Securitizers Per Proxy'
label_numperc_sec = 'tab:number_Percentage_securitizers'
size_string_numperc_sec = '\\scriptsize \n'
note_numperc_sec = "\\textit{Notes.} Total is the number of banks in the sample per year."
numperc_sec_latex = resultsToLatex(numperc_sec.T, caption_numperc_sec, label_numperc_sec,\
                                 size_string = size_string_numperc_sec, note_string = note_numperc_sec,\
                                 sidewaystable = True)
    
## Control Variables summary statistics
caption_oth = 'Summary Statistics Control Variables'
label_oth = 'tab:summary_control'
size_string_oth = '\\footnotesize \n'
note_oth = "\\textit{Notes.} All variables besides Operational income are in percentages. Operational income is in thousands USD."

ss_oth_latex = resultsToLatex(ss_oth, caption_oth, label_oth,\
                                 size_string = size_string_oth, note_string = note_oth,\
                                 sidewaystable = False)

#------------------------------------------------------------
# Save
#------------------------------------------------------------

# Securitization summary statistics
ss.to_excel('Tables/Summary_statistics.xlsx')

text_ss_latex = open('Tables/Summary_statistics.tex', 'w')
text_ss_latex.write(ss_latex)
text_ss_latex.close()

# Number and percentage securitization
num_sec.to_excel('Tables/Number_securitizers.xlsx')

text_num_sec_latex = open('Tables/Number_securitizers.tex', 'w')
text_num_sec_latex.write(num_sec_latex)
text_num_sec_latex.close()

numperc_sec.to_excel('Tables/Number_percentage_securitizers.xlsx')

text_numperc_sec_latex = open('Tables/Number_percentage_securitizers.tex', 'w')
text_numperc_sec_latex.write(numperc_sec_latex)
text_numperc_sec_latex.close()

# Control Variables summary statistics
ss_oth.to_excel('Tables/Summary_statistics_control.xlsx')

text_ss_oth_latex = open('Tables/Summary_statistics_control.tex', 'w')
text_ss_oth_latex.write(ss_oth_latex)
text_ss_oth_latex.close()

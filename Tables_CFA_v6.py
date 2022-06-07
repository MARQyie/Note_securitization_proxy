# --------------------------------------------
# TABLES CFA
# Mark van der Plaat
# December 2021
# --------------------------------------------
''' This script makes tables for the CFA securitization paper.

    All data is generated by Lavaan in R. See CFA_v1.R for more details.
'''

# --------------------------------------------
# Import Packages
# --------------------------------------------

# Data manipulation
import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid', font_scale=3)

import os

os.chdir(r'D:\RUG\PhD\Materials_papers\01-Note_on_securitization')

# Load table functions
from Functions_tables_CFA import *

# --------------------------------------------
# Prelims
# --------------------------------------------
dict_vars = getVarDict()

# Add Factor names
dict_vars.update({'LS': 'Loan Sales',
                  'ABS': 'ABS Sec.',
                  'CDO': 'CDO Sec.',
                  'ABCP': 'ABCP Sec.',
                  'SEC': 'Securitization'})

# --------------------------------------------
# Load data 
# --------------------------------------------

# Get all file names with binary and 1f|SBO|NoSBO
# NOTE: Do not load DWLS, WLSMV is better for ordinal data
files = [i for i in os.listdir('Results') if 'binary' in i
         and ('1f' in i or 'SBO' in i or 'NoSBO' in i)
         and 'WLSMV' in i
         and not 'agg' in i]
files.sort() # order: 1f, NoSBO, SBO

# Load files in list
## fit indices
dfs_fit = [pd.read_csv('Results/' + file, index_col=0)
           for file in files if 'fitmeasures' in file]

## Params
dfs_params = [pd.read_csv('Results/' + file, index_col=0)
           for file in files if 'params' in file]

## Reliability measures
# TODO Check which ones to take with ordinal data
dfs_reli = [pd.read_csv('Results/' + file, index_col=0)
           for file in files if 'reliability' in file]

## Modification Indices
dfs_mi = [pd.read_csv('Results/' + file, index_col=0)
           for file in files if 'modindices' in file]

## Residual covariance matrix
dfs_rescov = [pd.read_csv('Results/' + file, index_col=0)
           for file in files if 'rescov_standard' in file]

## Polychoric correlation Matrix
dfs_poly = [pd.read_csv('Results/' + file, index_col=0)
           for file in files if 'polychoriccorr' in file]

## Communality
dfs_r2 = [pd.read_csv('Results/' + file, index_col=0)
           for file in files if 'r2' in file]

# --------------------------------------------
# fit indices 
# --------------------------------------------

# One securization factor
## Get tidy table
df_fi_1f_tidy = tidyFitInd(dfs_fit[0])

## To latex
### Prelims
dict_options = {'column_format': 'p{3cm}' + 'p{1.5cm}' * df_fi_1f_tidy.shape[1],
                'caption': ('Fit Indices: One-Factor Model'),
                'label': 'tab:cfa_fit_1f',
                'position': 'th'}
notes = '\\multicolumn{2}{p{4.5cm}}{\\textit{Notes.} This table contains the fit indices for the one-factor model.} \n'
string_size = '\\scriptsize\n'

### Get latex table and save
latex_fi_onesec = table2Latex(df_fi_1f_tidy, dict_options, notes, string_size)
text_latex_fi_onesec = open('Results/CFA_fi_onesec.tex', 'w')
text_latex_fi_onesec.write(latex_fi_onesec)
text_latex_fi_onesec.close()

# All two factor models
## Get tidy table
df_fi_2f = pd.concat([dfs_fit[i] for i in range(len(dfs_fit) - 1, 0, -1)], axis=1)
df_fi_2f_tidy = tidyFitInd(df_fi_2f)

## To latex
### Prelims
dict_options = {'column_format': 'p{3cm}' + 'p{1.5cm}' * df_fi_2f_tidy.shape[1],
                'caption': ('Fit Indices: Two-factor Models'),
                'label': 'tab:cfa_fit_twofactor_models',
                'position': 'th'}
notes = '\\multicolumn{4}{p{9cm}}{\\textit{Notes.} The first column of this table contains the fit indices for the two-factor model with small business obligations transferred, and  the second column for the two-factor model without small business obligations transferred.} \n'
string_size = '\\scriptsize\n'

### Get latex table and save
latex_fi_2f = table2Latex(df_fi_2f_tidy, dict_options, notes, string_size)
text_latex_latex_fi_2f = open('Results/CFA_fi_2f.tex', 'w')
text_latex_latex_fi_2f.write(latex_fi_2f)
text_latex_latex_fi_2f.close()

# --------------------------------------------
# Parameter estimates 
# --------------------------------------------

# One securitization factor
## Get tidy table
df_params_1f_tidy = tidyParamEst(dfs_params[0])

## To latex
### Prelims
dict_options = {'column_format': 'p{4.75cm}p{4.75cm}' + 'p{1cm}' * df_params_1f_tidy.shape[1],
                'caption': ('Estimated Factor Loadings and (Co-)Variances and Thresholds: One-Factor Model'),
                'label': 'tab:cfa_table_1f',
                'position': 'th'}
notes = '\\multicolumn{6}{p{16cm}}{\\textit{Notes.}  For the factor loadings and (co-)variances, this table shows their estimated values, standard deviations and $p$-values, as well as completely standardized estimates. The table also presents the thresholds for each proxy.} \n'
string_size = '\\scriptsize\n'

### Get latex table and save
latex_params_onesec = table2Latex(df_params_1f_tidy, dict_options, notes, string_size)
text_latex_params_onesec = open('Results/CFA_params_1f.tex', 'w')
text_latex_params_onesec.write(latex_params_onesec)
text_latex_params_onesec.close()

# Two factors: Without SBO
df_params_nosbo_tidy = tidyParamEst(dfs_params[1])

## To latex
### Prelims
dict_options = {'column_format': 'p{4.75cm}p{4.75cm}' + 'p{1cm}' * df_params_nosbo_tidy.shape[1],
                'caption': ('Estimated Factor Loadings and (Co-)Variances and Thresholds: Two-Factor Model without Small Business Obligations Transferred'),
                'label': 'tab:cfa_table_2fnosbo',
                'position': 'th'}
notes = '\\multicolumn{6}{p{16cm}}{\\textit{Notes.}  For the factor loadings and (co-)variances, this table shows their estimated values, standard deviations and $p$-values, as well as completely standardized estimates. The table also presents the thresholds for each proxy.} \n'
string_size = '\\scriptsize\n'

### Get latex table and save
latex_params_nosbo = table2Latex(df_params_nosbo_tidy, dict_options, notes, string_size)
text_latex_params_nosbo = open('Results/CFA_params_nosbo.tex', 'w')
text_latex_params_nosbo.write(latex_params_nosbo)
text_latex_params_nosbo.close()

# Two factors: With SBO
df_params_theory_tidy = tidyParamEst(dfs_params[2])

## To latex
### Prelims
dict_options = {'column_format': 'p{4.75cm}p{4.75cm}' + 'p{1cm}' * df_params_theory_tidy.shape[1],
                'caption': ('Estimated Factor Loadings and (Co-)Variances and Thresholds: Two-Factor Model with Small Business Obligations Transferred'),
                'label': 'tab:cfa_table_2fsbo',
                'position': 'th'}
notes = '\\multicolumn{6}{p{16cm}}{\\textit{Notes.} For the factor loadings and (co-)variances, this table shows their estimated values, standard deviations and $p$-values, as well as completely standardized estimates. The table also presents the thresholds for each proxy.} \n'
string_size = '\\scriptsize\n'

### Get latex table and save
latex_params_theory = table2Latex(df_params_theory_tidy, dict_options, notes, string_size)
text_latex_params_theory = open('Results/CFA_params_sbo.tex', 'w')
text_latex_params_theory.write(latex_params_theory)
text_latex_params_theory.close()

# --------------------------------------------
#  Reliability
# --------------------------------------------

# Two-factor models
df_reliability = pd.concat([dfs_reli[i].loc[['alpha.ord','omega'],:]
                            for i in range(len(dfs_reli) - 1, -1, -1)], axis=1)

df_reliability.columns = pd.MultiIndex.from_tuples([('Two-Factor With SBO','F1'),
                              ('Two-Factor With SBO','F2'),
                              ('Two-Factor Without SBO','F1'),
                              ('Two-Factor Without SBO','F2'),
                              ('One-Factor','')],
                              names = ['',''])
df_reliability.index = ['Ordinal Alpha','Omega']

## To latex
### Prelims
dict_options = {'column_format': 'p{4cm}' + 'p{1cm}' * df_reliability.shape[1],
                'caption': ('Scale Reliability One- and Two-factor models'),
                'label': 'tab:cfa_reliability_2f',
                'position': 'th'}
notes = '\\multicolumn{6}{p{6cm}}{\\textit{Notes.} } \n'
string_size = '\\scriptsize\n'

### Get latex table and save
latex_params_reliability = table2Latex(df_reliability, dict_options, notes, string_size)
text_latex_params_reliability = open('Results/CFA_scale_reliability.tex', 'w')
text_latex_params_reliability.write(latex_params_reliability)
text_latex_params_reliability.close()

# --------------------------------------------
# Standardized Residual covariance matrix
# --------------------------------------------

# Set file name for rescov
files_rescov = ['Rescov_std_heatmap_1f.png',
                'Rescov_std_heatmap_nosbo.png',
                'Rescov_std_heatmap_sbo.png']

for matrix,name in zip(dfs_rescov, files_rescov):
    heatmap(matrix.iloc[1:,:-1].rename(columns=dict_vars, index=dict_vars), name)

# --------------------------------------------
# Polychoric covariance matrix
# --------------------------------------------

# Set file name for rescov
files_poly = ['Cov_polychoric_1f.png',
              'Cov_polychoric_nosbo.png',
              'Cov_polychoric_sbo.png']

for matrix,name in zip(dfs_poly, files_poly):
    heatmap(matrix.rename(columns=dict_vars, index=dict_vars), name)

# --------------------------------------------
# Modification indices 
# --------------------------------------------

# One factor model (the 1f model does not have factor loading MIs/EPCs)
# --------------------------------------------
## Get tidy tables
df_miepc_onesec_load, df_mi_cov_onesec, df_epc_cov_onesec = tidyModInd(dfs_mi[0])

## To latex
### MI Covariances
dict_options = {'column_format': 'p{4.75cm}' + 'p{1cm}' * df_mi_cov_onesec.shape[1],
                'caption': ('Modification Indices for the Error Covariances: One-Factor Model'),
                'label': 'tab:cfa_mi_cov_1f',
                'position': 'th'}
notes = '\\multicolumn{9}{p{20.5cm}}{\\textit{Notes.} This table shows the modification indices for the error covariances in the one-factor model.} \n'
string_size = '\\scriptsize\n'

latex_miepc = table2Latex(df_mi_cov_onesec, dict_options, notes, string_size)
latex_miepc = latex_miepc.replace('{table}','{sidewaystable}')
text_latex_miepc = open('Results/CFA_mi_1f_cov.tex', 'w')
text_latex_miepc.write(latex_miepc)
text_latex_miepc.close()

### EPC Covariances
dict_options = {'column_format': 'p{4.75cm}' + 'p{1cm}' * df_epc_cov_onesec.shape[1],
                'caption': ('EPCs of the Covariances: One-factor Model'),
                'label': 'tab:cfa_epc_cov_1f',
                'position': 'th'}
notes = '\\multicolumn{9}{p{20.5cm}}{\\textit{Notes.} This table shows the completely standardized expected parameter changes (EPCs) for the error covariances in the one-factor model.} \n'
string_size = '\\scriptsize\n'

latex_miepc = table2Latex(df_epc_cov_onesec, dict_options, notes, string_size)
latex_miepc = latex_miepc.replace('{table}','{sidewaystable}')
text_latex_miepc = open('Results/CFA_epc_1f_cov.tex', 'w')
text_latex_miepc.write(latex_miepc)
text_latex_miepc.close()

#SBO (Theory-based)
# --------------------------------------------
## Get tidy tables
df_miepc_theory_load, df_mi_cov_theory, df_epc_cov_theory = tidyModInd(dfs_mi[2])

## To latex
### MI EPC loadings
dict_options = {'column_format': 'p{4.75cm}' + 'p{1.5cm}' * df_miepc_theory_load.shape[1],
                'caption': ('Modification Indices and EPCs for the Factor Loadings: Two-Factor Model With Small Business Obligations Transferred'),
                'label': 'tab:cfa_mipec_load_sbo',
                'position': 'th'}
notes = '\\multicolumn{5}{p{13cm}}{\\textit{Notes.} This table provides modification indices and completely standardized expected parameter changes (EPCs) for the factor loadings in the two-factor  model with small business obligations transferred. This table does not provide any modification indices and EPCs  for the proxy \textit{Credit Default Swaps Purchased ($x_8$)}, because it already loads on both factors.} \n'
string_size = '\\scriptsize\n'

latex_miepc = table2Latex(df_miepc_theory_load, dict_options, notes, string_size)
text_latex_miepc = open('Results/CFA_miepc_sbo_load.tex', 'w')
text_latex_miepc.write(latex_miepc)
text_latex_miepc.close()

### MI Covariances
dict_options = {'column_format': 'p{4.75cm}' + 'p{1cm}' * df_mi_cov_theory.shape[1],
                'caption': ('Modification Indices for the Error Covariances:  Two-Factor Model With Small Business Obligations Transferred'),
                'label': 'tab:cfa_mi_cov_sbo',
                'position': 'th'}
notes = '\\multicolumn{10}{p{20.5cm}}{\\textit{Notes.} This table shows the modification indices for the error covariances in the two-factor model with small business obligations transferred.} \n'
string_size = '\\scriptsize\n'

latex_miepc = table2Latex(df_mi_cov_theory, dict_options, notes, string_size)
latex_miepc = latex_miepc.replace('{table}','{sidewaystable}')
text_latex_miepc = open('Results/CFA_mi_sbo_cov.tex', 'w')
text_latex_miepc.write(latex_miepc)
text_latex_miepc.close()

### EPC Covariances
dict_options = {'column_format': 'p{4.75cm}' + 'p{1cm}' * df_epc_cov_theory.shape[1],
                'caption': ('EPCs for the Error Covariances: Two-Factor Model With Small Business Obligations Transferred'),
                'label': 'tab:cfa_epc_cov_sbo',
                'position': 'th'}
notes = '\\multicolumn{10}{p{20.5cm}}{\\textit{Notes.} This tables shows the completely standardized expected parameter changes (EPCs) for the error covariances in the two-factor model with small business obligations transferred.} \n'
string_size = '\\scriptsize\n'

latex_miepc = table2Latex(df_epc_cov_theory, dict_options, notes, string_size)
latex_miepc = latex_miepc.replace('{table}','{sidewaystable}')
text_latex_miepc = open('Results/CFA_epc_sbo_cov.tex', 'w')
text_latex_miepc.write(latex_miepc)
text_latex_miepc.close()

# SBO
# --------------------------------------------
## Get tidy tables
df_miepc_theory_load, df_mi_cov_theory, df_epc_cov_theory = tidyModInd(dfs_mi[1])

## To latex
### MI EPC loadings
dict_options = {'column_format': 'p{4.75cm}' + 'p{1.5cm}' * df_miepc_theory_load.shape[1],
                'caption': ('Modification Indices and EPCs for the Factor Loadings: Two-Factor Model Without Small Business Obligations Transferred'),
                'label': 'tab:cfa_mipec_load_nosbo',
                'position': 'th'}
notes = '\\multicolumn{5}{p{13cm}}{\\textit{Notes.} This table provides modification indices and completely standardized expected parameter changes (EPCs) for the factor loadings in the two-factor model without small business obligations transferred.. This table does not provide any modification indices and EPCs  for the proxy \textit{Credit Default Swaps Purchased ($x_8$)}, because it already loads on both factors.} \n'
string_size = '\\scriptsize\n'

latex_miepc = table2Latex(df_miepc_theory_load, dict_options, notes, string_size)
text_latex_miepc = open('Results/CFA_miepc_nosbo_load.tex', 'w')
text_latex_miepc.write(latex_miepc)
text_latex_miepc.close()

### MI Covariances
dict_options = {'column_format': 'p{4.75cm}' + 'p{1cm}' * df_mi_cov_theory.shape[1],
                'caption': ('Modification Indices for the Error Covariances:  Two-Factor Model Without Small Business Obligations Transferred'),
                'label': 'tab:cfa_mi_cov_nosbo',
                'position': 'th'}
notes = '\\multicolumn{9}{p{20.5cm}}{\\textit{Notes.} This table shows the modification indices for the error covariances in the two-factor model without small business obligations transferred.} \n'
string_size = '\\scriptsize\n'

latex_miepc = table2Latex(df_mi_cov_theory, dict_options, notes, string_size)
latex_miepc = latex_miepc.replace('{table}','{sidewaystable}')
text_latex_miepc = open('Results/CFA_mi_nosbo_cov.tex', 'w')
text_latex_miepc.write(latex_miepc)
text_latex_miepc.close()

### EPC Covariances
dict_options = {'column_format': 'p{4.75cm}' + 'p{1cm}' * df_epc_cov_theory.shape[1],
                'caption': ('EPCs for the Error Covariances: Two-Factor Model Without Small Business Obligations Transferred'),
                'label': 'tab:cfa_epc_cov_nosbo',
                'position': 'th'}
notes = '\\multicolumn{9}{p{20.5cm}}{\\textit{Notes.} This tables shows the completely standardized expected parameter changes (EPCs) for the error covariances in the two-factor model without small business obligations transferred.} \n'
string_size = '\\scriptsize\n'

latex_miepc = table2Latex(df_epc_cov_theory, dict_options, notes, string_size)
latex_miepc = latex_miepc.replace('{table}','{sidewaystable}')
text_latex_miepc = open('Results/CFA_epc_nosbo_cov.tex', 'w')
text_latex_miepc.write(latex_miepc)
text_latex_miepc.close()

# --------------------------------------------
# Communalities
# --------------------------------------------

# Clean the tables
dfs_r2_tidy = [data.rename(columns={'x': 'Communality'}, index=dict_vars) for data in dfs_r2]

# Add unique variance columns
for i in range(3):
    dfs_r2_tidy[i]['Unique Variance'] = 1 - dfs_r2_tidy[i].Communality

# To latex
# One factor
## Prelims
dict_options = {'column_format': 'p{5cm}' + 'p{1.5cm}' * dfs_r2_tidy[0].shape[1],
                'caption': ('Communalities and Unique Variances: One-factor Model'),
                'label': 'tab:cfa_r2_1f',
                'position': 'th'}
notes = '\\multicolumn{3}{p{9cm}}{\\textit{Notes.} This table shows the communalities and unique variances for all proxy variables. Both are expressed as a fraction of the total variance.} \n'
string_size = '\\scriptsize\n'

### Get latex table and save
latex_r2_theory = table2Latex(dfs_r2_tidy[0], dict_options, notes, string_size)
text_latex_r2_theory = open('Results/CFA_r2_1f.tex', 'w')
text_latex_r2_theory.write(latex_r2_theory)
text_latex_r2_theory.close()

# Two factor NO SBO
## Prelims
dict_options = {'column_format': 'p{5cm}' + 'p{1.5cm}' * dfs_r2_tidy[1].shape[1],
                'caption': ('Communalities and Unique Variances: Two-Factor Model Without Small Business Obligations Transferred'),
                'label': 'tab:cfa_r2_nosbo',
                'position': 'th'}
notes = '\\multicolumn{3}{p{9cm}}{\\textit{Notes.} This table shows the communalities and unique variances for all proxy variables. Both are expressed as a fraction of the total variance.} \n'
string_size = '\\scriptsize\n'

### Get latex table and save
latex_r2_theory = table2Latex(dfs_r2_tidy[1], dict_options, notes, string_size)
text_latex_r2_theory = open('Results/CFA_r2_nosbo.tex', 'w')
text_latex_r2_theory.write(latex_r2_theory)
text_latex_r2_theory.close()

# Two factor SBO
## Prelims
dict_options = {'column_format': 'p{5cm}' + 'p{1.5cm}' * dfs_r2_tidy[1].shape[1],
                'caption': ('Communalities and Unique Variances: Two-Factor Model With Small Business Obligations Transferred'),
                'label': 'tab:cfa_r2_sbo',
                'position': 'th'}
notes = '\\multicolumn{3}{p{9cm}}{\\textit{Notes.} This table shows the communalities and unique variances for all proxy variables. Both are expressed as a fraction of the total variance.} \n'
string_size = '\\scriptsize\n'

### Get latex table and save
latex_r2_theory = table2Latex(dfs_r2_tidy[2], dict_options, notes, string_size)
text_latex_r2_theory = open('Results/CFA_r2_sbo.tex', 'w')
text_latex_r2_theory.write(latex_r2_theory)
text_latex_r2_theory.close()
#--------------------------------------------
# TABLES CFA
# Mark van der Plaat
# August 2021
#--------------------------------------------
''' This script makes tables for the CFA securitization paper.

    All data is generated by Lavaan in R. See CFA_v1.R for more details.
'''

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

import os
os.chdir(r'D:\RUG\PhD\Materials_papers\01-Note_on_securitization')

#--------------------------------------------
# Functions
#--------------------------------------------
# Set heatmap function 
def heatmap(matrix, file, annot = True):
    
    # Set mask
    mask = np.triu(np.ones_like(matrix, dtype=bool), 1)
      
    # Set aesthetics
    dic_aes = {'mask':mask,
               'annot' : annot,
               'center': 0,
               'cmap': 'coolwarm'}
    
    # Make heatmap
    fig, ax = plt.subplots(figsize=(36,24))
    sns.heatmap(matrix, **dic_aes)  
    plt.tight_layout()
    
    # Save heatmap
    fig.savefig('Figures/CFA_covariance_maps/' + file)
    
# Function to make the parameter estimate table pretty
def tidyParamEst(data):
    
    # Select columns
    data_clean = data.iloc[:,[3,4,5,6,9,10]]
    
    # Set column names
    data_clean.columns = ['Estimates', 'SD', 'Z-value', 'P-value', 'Estimates (std.)', 'Estimates (compl. std.)']
    
    # Set index names
    ## Prelims
    index_lst = list()
    lst_factors = ('SEC','LS','ABS','CDO','ABCP','ABSCDO','CD')
    var_names = ['cr_as_nsres','cr_as_nsoth','hmda_gse_amount','hmda_priv_amount',\
             'cr_ls_income','cr_as_rmbs','cr_as_abs','hmda_sec_amount',\
             'cr_sec_income','cr_as_sbo',\
             'cr_cds_purchased','cr_trs_purchased','cr_co_purchased',\
             'cr_abcp_uc_own','cr_abcp_ce_own','cr_abcp_uc_oth',\
                 'cr_serv_fees']
    var_labels = ['Res. Assets Sold, Not Sec.','Oth. Assets Sold, Not Sec.',\
                  'Sold To GSE (HMDA)','Sold to Private (HMDA)',\
                  'Loan Sales Income','Res. Assets Sold, Sec.',\
                  'Other Assets Sold, Sec.','Securitized (HMDA)',\
                  'Sec. Income','SBO Sold',\
                  'CDSs Purchased','TRSs Purchased',\
                  'COs Purchased','Un. Com. ABCP (Own)',\
                  'Credit Exp. ABCP (Own)','Un. Com. ABCP (Others)',\
                  'Net Servicing Fees']
    var_dict = dict(zip(var_names,var_labels))
    
    ## Loop over rows to get the correct parameter name
    for index, row  in data.iloc[:,:3].iterrows(): 
        if row['lhs'] in lst_factors:
            if row['rhs'] in lst_factors:
                index_lst.append(['Variance ($\phi$)', str(row['lhs']), str(row['rhs'])])
                #index_lst.append('$\phi_{' + str(row['lhs']) + ',' +  str(row['rhs']) +'}$')
            else:
                index_lst.append(['Loading ($\lambda$)', str(row['lhs']), str(var_dict[row['rhs']])])
                #index_lst.append('$\lambda_{' + str(row['lhs']) + ',' +  str(var_dict[row['rhs']]) +'}$')
        else:
            index_lst.append(['Variance ($\delta$)', str(var_dict[row['lhs']]), str(var_dict[row['rhs']])])
            #index_lst.append('$\delta_{' + str(var_dict[row['lhs']]) + ',' +  str(var_dict[row['rhs']]) +'}$')
    
    ## Change Index
    data_index = list(map(list, zip(*index_lst)))
    data_clean = data_clean.assign(Operator = data_index[0],
                                   LHS = data_index[1],
                                   RHS = data_index[2])

    data_clean.set_index(['Operator','LHS','RHS'], inplace = True)
    
    return data_clean

# Function to make latex table from pandas dataframe
def table2Latex(data, options, notes, string_size):
    
    # Get latex table
    latex_table = data.to_latex(na_rep = '', float_format = '{:0.4f}'.format,\
                                                longtable = False, multicolumn = False,\
                                                multicolumn_format = 'c', escape = False,\
                                                **options)
    
    # add notes to bottom of the table
    location_mid = latex_table.find('\end{tabular}')
    latex_table = latex_table[:location_mid] + notes + latex_table[location_mid:]
    
    # adjust sting size
    location_size = latex_table.find('\centering\n')
    latex_table = latex_table[:location_size + len('\centering\n')] + string_size + latex_table[location_size + len('\centering\n'):]
    
    return latex_table
    
def tidyFitInd(data):
    
    # Get indices
    lst_fi = ['npar','df','chisq','pvalue','chisq.scaled','pvalue.scaled','chisq.scaling.factor',\
              'baseline.df','baseline.chisq', 'baseline.pvalue', 'baseline.chisq.scaled',\
              'baseline.pvalue.scaled', 'baseline.chisq.scaling.factor',\
              'gfi','agfi','nfi','tli.robust','cfi.robust','rmsea.robust',\
              'rmsea.ci.lower.robust', 'rmsea.ci.upper.robust','srmr','ifi','rni.robust',\
              'aic','bic']
    data_clean = data.loc[lst_fi,:]
    
    # Rename index
    lst_fi_labels = ['No. Params','DoF','$\chi^2$','p-val $\chi^2$','$\chi^2$ (scaled)',\
                     'p-val $\chi^2$ (scaled)','$\chi^2$ scaling factor',\
                     'DoF baseline','$\chi^2$ baseline', 'p-val $\chi^2$ baseline',\
                     '$\chi^2$ baseline (scaled)', 'p-val $\chi^2$ baseline (scaled)', '$\chi^2$ baseline scaling factor',\
                     'GFI','AGFI','NFI','TLI (robust)','CFI (robust)','RMSEA (robust)',\
                     'RMSEA lower bound (robust)', 'RMSEA upper bound (robust)',\
                     'SRMR','IFI','RNI (robust)','AIC','BIC']
    data_clean.index = lst_fi_labels
    
    # Rename columns
    if data.shape[1] == 2:
        data_clean.columns = ['First-Order','Second-Order']
    else:
        data_clean.rename(columns = {'Unnamed: 1':'Index'}, inplace = True)
    
    return data_clean

# Function to make the modification table table pretty
def tidyModInd(data):
    
    # Select columns
    data_clean = data.iloc[:,3:-1]
    
    # Set column names
    data_clean.columns = ['Modification Indices','EPC','EPC (std.)','EPC (compl. std.)']
    
    # Set index names
    ## Prelims
    index_lst = list()
    lst_factors = ('SEC','LS','ABS','CDO','ABCP','ABSCDO','CD')
    var_names = ['cr_as_nsres','cr_as_nsoth','hmda_gse_amount','hmda_priv_amount',\
             'cr_ls_income','cr_as_rmbs','cr_as_abs','hmda_sec_amount',\
             'cr_sec_income','cr_as_sbo',\
             'cr_cds_purchased','cr_trs_purchased','cr_co_purchased',\
             'cr_abcp_uc_own','cr_abcp_ce_own','cr_abcp_uc_oth',\
                 'cr_serv_fees']
    var_labels = ['Res. Assets Sold, Not Sec.','Oth. Assets Sold, Not Sec.',\
                  'Sold To GSE (HMDA)','Sold to Private (HMDA)',\
                  'Loan Sales Income','Res. Assets Sold, Sec.',\
                  'Other Assets Sold, Sec.','Securitized (HMDA)',\
                  'Sec. Income','SBO Sold',\
                  'CDSs Purchased','TRSs Purchased',\
                  'COs Purchased','Un. Com. ABCP (Own)',\
                  'Credit Exp. ABCP (Own)','Un. Com. ABCP (Others)',\
                  'Net Servicing Fees']
    var_dict = dict(zip(var_names,var_labels))
    
    ## Loop over rows to get the correct parameter name
    for index, row  in data.iloc[:,:3].iterrows(): 
        if row['lhs'] in lst_factors:
            if row['rhs'] in lst_factors:
                index_lst.append(['Variance ($\phi$)', str(row['lhs']), str(row['rhs'])])
                #index_lst.append('$\phi_{' + str(row['lhs']) + ',' +  str(row['rhs']) +'}$')
            else:
                index_lst.append(['Loading ($\lambda$)', str(row['lhs']), str(var_dict[row['rhs']])])
                #index_lst.append('$\lambda_{' + str(row['lhs']) + ',' +  str(var_dict[row['rhs']]) +'}$')
        else:
            index_lst.append(['Variance ($\delta$)', str(var_dict[row['lhs']]), str(var_dict[row['rhs']])])
            #index_lst.append('$\delta_{' + str(var_dict[row['lhs']]) + ',' +  str(var_dict[row['rhs']]) +'}$')
    
    ## Change Index
    data_index = list(map(list, zip(*index_lst)))
    data_clean = data_clean.assign(Operator = data_index[0],
                                   LHS = data_index[1],
                                   RHS = data_index[2])

    data_clean.set_index(['Operator','LHS','RHS'], inplace = True)
    
    return data_clean

#--------------------------------------------
# Prelims
#--------------------------------------------
var_names = ['cr_as_nsres','cr_as_nsoth','hmda_gse_amount','hmda_priv_amount',\
             'cr_ls_income','cr_as_rmbs','cr_as_abs','hmda_sec_amount',\
             'cr_sec_income','cr_as_sbo',\
             'cr_cds_purchased','cr_trs_purchased','cr_co_purchased',\
             'cr_abcp_uc_own','cr_abcp_ce_own','cr_abcp_uc_oth',\
                 'cr_serv_fees']
var_labels = ['Res. Assets Sold, Not Sec.','Other Assets Sold, Not Sec.',\
              'Sold To GSE (HMDA)','Sold to Private (HMDA)',\
              'Loan Sales Income','Res. Assets Sold, Sec.',\
              'Other Assets Sold, Sec.','Securitized (HMDA)',\
              'Sec. Income','SBO Sold',\
              'CDSs Purchased','TRSs Purchased',\
              'COs Purchased','Unused Com. ABCP (Own)',\
              'Credit Exp. ABCP (Own)','Unused Com. ABCP (Others)',\
              'Net Servicing Fees']
dict_vars = dict(zip(var_names,var_labels))

# Add Factor names
dict_vars.update({'LS':'Loan Sales',
                  'ABS':'ABS Sec.',
                  'CDO':'CDO Sec.',
                  'ABCP':'ABCP Sec.',
                  'SEC':'Securitization'})

#--------------------------------------------
# Load data 
#--------------------------------------------

# Parameter estimates
df_params_nest = pd.read_csv('Results/CFA_params_impr_nest.csv', index_col = 0)
df_params_impr = pd.read_csv('Results/CFA_params_impr.csv', index_col = 0)
df_params_onesec = pd.read_csv('Results/CFA_params_onesec.csv', index_col = 0)

# Model implied covariance matrix
df_micm_nest = pd.read_csv('Results/CFA_modimplied_cov_impr_nest.csv', index_col = 0)
df_micm_impr = pd.read_csv('Results/CFA_modimplied_cov_impr.csv', index_col = 0)
df_micm_onesec = pd.read_csv('Results/CFA_modimplied_cov_onesec.csv', index_col = 0)

# Residual covariance matrix
df_rcm_nest = pd.read_csv('Results/CFA_rescov_impr_nest.csv', index_col = 0)
df_srcm_nest = pd.read_csv('Results/CFA_rescov_standard_impr_nest.csv', index_col = 0)

df_rcm_impr = pd.read_csv('Results/CFA_rescov_impr.csv', index_col = 0)
df_srcm_impr = pd.read_csv('Results/CFA_rescov_standard_impr.csv', index_col = 0)

df_rcm_onesec = pd.read_csv('Results/CFA_rescov_onesec.csv', index_col = 0)
df_srcm_onesec = pd.read_csv('Results/CFA_rescov_standard_onesec.csv', index_col = 0)

# fit indices 
df_fi_nest = pd.read_csv('Results/CFA_fitmeasures_impr_nest.csv', index_col = 0)
df_fi_impr = pd.read_csv('Results/CFA_fitmeasures_impr.csv', index_col = 0)
df_fi_onesec = pd.read_csv('Results/CFA_fitmeasures_onesec.csv', index_col = 0)

# Modification indices
df_mi_nest = pd.read_csv('Results/CFA_modindices_impr_nest.csv', index_col = 0)
df_mi_impr = pd.read_csv('Results/CFA_modindices_impr.csv', index_col = 0)
df_mi_onesec = pd.read_csv('Results/CFA_modindices_onesec.csv', index_col = 0)

# Communalities
df_r2_impr = pd.read_csv('Results/CFA_r2_impr.csv', index_col = 0)

#--------------------------------------------
# Parameter estimates 
#--------------------------------------------

# Nested model
## Get tidy table
df_params_nest_tidy = tidyParamEst(df_params_nest)

## To latex
### Prelims
dict_options = {'column_format':'p{1.45cm}p{3cm}p{3cm}' + 'p{.95cm}' * df_params_nest_tidy.shape[1],
                'caption':('Parameter Estimates: Hypothesized Second-Order Model'),
                'label':'tab:cfa_table_nest',
                'position':'th'}
notes = '\\multicolumn{9}{p{16.75cm}}{\\textit{Notes.} Factor loadings, factor variances and indicator variances of the hypothesized factor model with a nested securitization structure. The factors ABS, CDO, ABCP are nested under a factor Securitization, which captures the general variance of securitization and equals the hypothesized factor model.The second to last column contains the standardized parameter estimates, where the factor variances are fixed to one. The last column presents the completely standardized parameter estimates, where the factor variances are fixed to one and all other parameters are standardized.} \n'
string_size = '\\tiny\n'

### Get latex table and save
latex_params_nest = table2Latex(df_params_nest_tidy,dict_options,notes,string_size)
text_latex_params_nest = open('Results/CFA_params_nest.tex', 'w')
text_latex_params_nest.write(latex_params_nest)
text_latex_params_nest.close()

# Improved model
## Get tidy table
df_params_impr_tidy = tidyParamEst(df_params_impr)

## To latex
### Prelims
dict_options = {'column_format':'p{1.45cm}p{3cm}p{3cm}' + 'p{.95cm}' * df_params_impr_tidy.shape[1],
                'caption':('Parameter Estimates: Hypothesized First-Order Model'),
                'label':'tab:cfa_table_impr',
                'position':'th'}
notes = '\\multicolumn{9}{p{16.75cm}}{\\textit{Notes.} Parameter estimates, factor variances and indicator variances of the hypothesized factor model without a nested securitization structure. The improved model is similar to the non-nested model but without the CDO factor and without servicing fees. The second to last column contains the standardized parameter estimates, where the factor variances are fixed to one. The last column presents the completely standardized parameter estimates, where the factor variances are fixed to one and all other parameters are standardized.} \n'
string_size = '\\tiny\n'

### Get latex table and save
latex_params_impr = table2Latex(df_params_impr_tidy,dict_options,notes,string_size)
text_latex_params_impr = open('Results/CFA_params_impr.tex', 'w')
text_latex_params_impr.write(latex_params_impr)
text_latex_params_impr.close()

# One Sec
## Get tidy table
df_params_onesec_tidy = tidyParamEst(df_params_onesec)

## To latex
### Prelims
dict_options = {'column_format':'p{1.45cm}p{3cm}p{3cm}' + 'p{.95cm}' * df_params_onesec_tidy.shape[1],
                'caption':('Parameter Estimates: Single Securitization Factor Model'),
                'label':'tab:cfa_table_onesec',
                'position':'th'}
notes = '\\multicolumn{9}{p{16.75cm}}{\\textit{Notes.} Parameter estimates, factor variances and indicator variances of the factor model with a one securitization factor. The second to last column contains the standardized parameter estimates, where the factor variances are fixed to one. The last column presents the completely standardized parameter estimates, where the factor variances are fixed to one and all other parameters are standardized.} \n'
string_size = '\\tiny\n'

### Get latex table and save
latex_params_onesec = table2Latex(df_params_onesec_tidy,dict_options,notes,string_size)
text_latex_params_onesec = open('Results/CFA_params_onesec.tex', 'w')
text_latex_params_onesec.write(latex_params_onesec)
text_latex_params_onesec.close()

#--------------------------------------------
#  Model implied covariance matrix 
#--------------------------------------------

# Nested model
heatmap(df_micm_nest.rename(columns = dict_vars, index = dict_vars), 'Modelimplied_cov_heatmap_nest.png')

# Improved model
heatmap(df_micm_impr.rename(columns = dict_vars, index = dict_vars), 'Modelimplied_cov_heatmap_impr.png')

# One Sec
heatmap(df_micm_onesec.rename(columns = dict_vars, index = dict_vars), 'Modelimplied_cov_heatmap_onesec.png')

#--------------------------------------------
# Residual covariance matrix
#--------------------------------------------

# Nested model
heatmap(df_rcm_nest.rename(columns = dict_vars, index = dict_vars), 'Residual_cov_heatmap_nest.png')
heatmap(df_srcm_nest.rename(columns = dict_vars, index = dict_vars), 'Residual_cov_std_heatmap_nest.png')

# Non-nested model
heatmap(df_rcm_impr.rename(columns = dict_vars, index = dict_vars), 'Residual_cov_heatmap_impr.png')
heatmap(df_srcm_impr.rename(columns = dict_vars, index = dict_vars), 'Residual_cov_std_heatmap_impr.png')

# One Sec
heatmap(df_rcm_onesec.rename(columns = dict_vars, index = dict_vars), 'Residual_cov_heatmap_onesec.png')
heatmap(df_srcm_onesec.rename(columns = dict_vars, index = dict_vars), 'Residual_cov_std_heatmap_onesec.png')

#--------------------------------------------
# fit indices 
#--------------------------------------------

# Respecified model
df_fi_resp = pd.concat([df_fi_impr, df_fi_nest], axis = 1)

## Get tidy table
df_fi_resp_tidy = tidyFitInd(df_fi_resp)

## To latex
### Prelims
dict_options = {'column_format':'p{4cm}' + 'p{1.5cm}' * df_fi_resp_tidy.shape[1],
                'caption':('Fit Indices: Hypothesized Models'),
                'label':'tab:cfa_fit_resp',
                'position':'th'}
notes = '\\multicolumn{3}{p{7.5cm}}{\\textit{Notes.} Fit indices of the hypothesized first- and second-order model.} \n'
string_size = '\\scriptsize\n'

### Get latex table and save
latex_fi_nest = table2Latex(df_fi_resp_tidy,dict_options,notes,string_size)
text_latex_fi_nest = open('Results/CFA_fi_resp.tex', 'w')
text_latex_fi_nest.write(latex_fi_nest)
text_latex_fi_nest.close()

# Nested model
## Get tidy table
df_fi_nest_tidy = tidyFitInd(df_fi_nest)

## To latex
### Prelims
dict_options = {'column_format':'p{4cm}' + 'p{1.5cm}' * df_fi_nest_tidy.shape[1],
                'caption':('Fit Indices: Hypothesized Second-Order Model Model'),
                'label':'tab:cfa_fit_nest',
                'position':'th'}
notes = '\\multicolumn{2}{p{5.5cm}}{\\textit{Notes.} Fit indices of the hypothesized second-order model.} \n'
string_size = '\\scriptsize\n'

### Get latex table and save
latex_fi_nest = table2Latex(df_fi_nest_tidy,dict_options,notes,string_size)
text_latex_fi_nest = open('Results/CFA_fi_nest.tex', 'w')
text_latex_fi_nest.write(latex_fi_nest)
text_latex_fi_nest.close()

# Improved model
## Get tidy table
df_fi_impr_tidy = tidyFitInd(df_fi_impr)

## To latex
### Prelims
dict_options = {'column_format':'p{4cm}' + 'p{1.5cm}' * df_fi_impr_tidy.shape[1],
                'caption':('Fit Indices: Hypothesized First-Order Model'),
                'label':'tab:cfa_fit_impr',
                'position':'th'}
notes = '\\multicolumn{2}{p{5.5cm}}{\\textit{Notes.} Fit indices of the hypothesized first-order model.} \n'
string_size = '\\scriptsize\n'

### Get latex table and save
latex_fi_impr = table2Latex(df_fi_impr_tidy,dict_options,notes,string_size)
text_latex_fi_impr = open('Results/CFA_fi_impr.tex', 'w')
text_latex_fi_impr.write(latex_fi_impr)
text_latex_fi_impr.close()

# One Sec
## Get tidy table
df_fi_onesec_tidy = tidyFitInd(df_fi_onesec)

## To latex
### Prelims
dict_options = {'column_format':'p{4cm}' + 'p{1.5cm}' * df_fi_onesec_tidy.shape[1],
                'caption':('Fit Indices: One Securitization Factor Model'),
                'label':'tab:cfa_fit_onesec',
                'position':'th'}
notes = '\\multicolumn{2}{p{5.5cm}}{\\textit{Notes.} Fit indices of the model with one securitization factor.} \n'
string_size = '\\scriptsize\n'

### Get latex table and save
latex_fi_onesec = table2Latex(df_fi_onesec_tidy,dict_options,notes,string_size)
text_latex_fi_onesec = open('Results/CFA_fi_onesec.tex', 'w')
text_latex_fi_onesec.write(latex_fi_onesec)
text_latex_fi_onesec.close()

#--------------------------------------------
# Modification indices 
#--------------------------------------------

# Nested model
## Get tidy table
df_mi_nest_tidy = tidyModInd(df_mi_nest)

## To latex
### Prelims
dict_options = {'column_format':'p{1.45cm}p{3cm}p{3cm}' + 'p{1.35cm}' * df_mi_nest_tidy.shape[1],
                'caption':('Modification Indices: Hypothesized Second-Order Model'),
                'label':'tab:cfa_mi_nest',
                'position':'th'}
notes = '\\multicolumn{7}{p{15cm}}{\\textit{Notes.} Modification indices of the hypothesized second-order model. The second to last column contains the standardized modification indices, where the factor variances are fixed to one. The last column presents the completely standardized modification indices, where the factor variances are fixed to one and all other parameters are standardized.} \n'
string_size = '\\tiny\n'

### Get latex table and save
latex_mi_nest = table2Latex(df_mi_nest_tidy,dict_options,notes,string_size)
text_latex_mi_nest = open('Results/CFA_mi_nest.tex', 'w')
text_latex_mi_nest.write(latex_mi_nest)
text_latex_mi_nest.close()

# Improved model
## Get tidy table
df_mi_impr_tidy = tidyModInd(df_mi_impr)

## To latex
### Prelims
dict_options = {'column_format':'p{1.45cm}p{3cm}p{3cm}' + 'p{1.35cm}' * df_mi_impr_tidy.shape[1],
                'caption':('Modification Indices: Respecified First-Order Model'),
                'label':'tab:cfa_mi_impr',
                'position':'th'}
notes = '\\multicolumn{7}{p{15cm}}{\\textit{Notes.} Modification indices of the hypothesized first-order model. The second to last column contains the standardized modification indices, where the factor variances are fixed to one. The last column presents the completely standardized modification indices, where the factor variances are fixed to one and all other parameters are standardized.} \n'
string_size = '\\tiny\n'

### Get latex table and save
latex_mi_impr = table2Latex(df_mi_impr_tidy,dict_options,notes,string_size)
text_latex_mi_impr = open('Results/CFA_mi_impr.tex', 'w')
text_latex_mi_impr.write(latex_mi_impr)
text_latex_mi_impr.close()

# One Sec
## Get tidy table
df_mi_onesec_tidy = tidyModInd(df_mi_onesec)

## To latex
### Prelims
dict_options = {'column_format':'p{1.45cm}p{3cm}p{3cm}' + 'p{1.35cm}' * df_mi_onesec_tidy.shape[1],
                'caption':('Modification Indices: One Securitization Factor Model'),
                'label':'tab:cfa_mi_onesec',
                'position':'th'}
notes = '\\multicolumn{7}{p{15cm}}{\\textit{Notes.} Modification indices of the model with one securitization factor. The second to last column contains the standardized modification indices, where the factor variances are fixed to one. The last column presents the completely standardized modification indices, where the factor variances are fixed to one and all other parameters are standardized.} \n'
string_size = '\\tiny\n'

### Get latex table and save
latex_mi_onesec = table2Latex(df_mi_impr_tidy,dict_options,notes,string_size)
text_latex_mi_onesec = open('Results/CFA_mi_onesec.tex', 'w')
text_latex_mi_onesec.write(latex_mi_onesec)
text_latex_mi_onesec.close()

#--------------------------------------------
# Communalities
#--------------------------------------------

# Improved model
## Clean the table
df_r2_impr_tidy = df_r2_impr.rename(columns = {'x':'Communality'}, index = dict_vars)

## To latex
### Prelims
dict_options = {'column_format':'p{5cm}' + 'p{1.5cm}' * df_r2_impr_tidy.shape[1],
                'caption':('Communalities: Respecified First-Order Model'),
                'label':'tab:cfa_r2_impr',
                'position':'th'}
notes = '\\multicolumn{2}{p{6.5cm}}{\\textit{Notes.} Communalities of the respecified first-order model.} \n'
string_size = '\\scriptsize\n'

### Get latex table and save
latex_r2_impr = table2Latex(df_r2_impr_tidy,dict_options,notes,string_size)
text_latex_r2_impr = open('Results/CFA_r2_impr.tex', 'w')
text_latex_r2_impr.write(latex_r2_impr)
text_latex_r2_impr.close()
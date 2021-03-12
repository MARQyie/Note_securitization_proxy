#--------------------------------------------
# Tables of the Arellano-Bond estimation for Note
# Mark van der Plaat
# March 2021
#--------------------------------------------

#--------------------------------------------
# Import Packages
#--------------------------------------------
    
# Data manipulation
import numpy as np
import pandas as pd

# Plot packages
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
sns.set(style = 'ticks', font_scale = 1.5, palette = 'Greys_d')

# Set WD
import glob
import os
os.chdir(r'D:\RUG\PhD\Materials_papers\01-Note_on_securitization')

#------------------------------------------------------------
# Make functions
#------------------------------------------------------------

def estimationTable(df, show = 'pval', stars = False, col_label = 'Est. Results'):
    ''' This function takes a df with estimation results and returns 
        a formatted column. 
        
        Input:  df = pandas dataframe estimation results
                show = Str indicating what to show (std, tval, or pval)
                stars = Boolean, show stars based on pval yes/no
                col_label = Str, label for the resulting pandas dataframe
        
        Output: Pandas dataframe
        '''
    # Prelims
    ## Set dictionary for index and columns
    dict_other = {'ln_ta':'Size',
                  'nim':'Net Interest Margin',
                  'cti':'Costs-to-Income',
                  'liq_ratio':'Liquidity Ratio',
                  'loan_ratio':'Loan ratio',
                  'gap':'GAP',
                  'dum_2014':'Year 2014',
                  'dum_2015':'Year 2015',
                  'dum_2016':'Year 2016',
                  'dum_2017':'Year 2017',
                  'dum_2018':'Year 2018',
                  'dum_2019':'Year 2019',
                  'zscore_l1':'Z-score$_{t-1}$',
                  'Parameter':'Parameter',
                  'Std. Err.':'Standard Deviation',
                  'T-stat':'$t$-value',
                  'P-value':'$p$-value',
                  'Lower CI':'Lower CI',
                  'Upper CI':'Upper CI',
                  'nobs':'Observations',
                  'adj_rsquared':'Adj. $R^2$',
                  'j_test':'J-test',
                  'c_test':'C-test',
                  'constant':'Intercept'}
    dict_sec = dict(zip(['hmda_gse_amount', 'hmda_priv_amount','hmda_sec_amount',\
                         'cr_serv_fees','cr_sec_income','cr_ls_income',\
                         'cr_cd_sold','cr_cd_purchased','cr_as_sec','cr_as_nonsec',\
                         'cr_ce_sec','cr_ta_secveh','cr_ta_abcp','cr_ta_vie_other',\
                         'cr_cd_gross', 'cr_cd_net'],\
            ['HDMA GSE','HMDA Private','HMDA Sec.',\
             'Serv. Fees','Sec. Income','LS Income','CD Sold',\
             'CD Purchased','Assets Sold and Sec.','Asset Sold and Not Sec.',\
             'Cred. Exp. Oth.','TA Sec. Veh.','TA ABCP','TA Oth. VIEs',\
             'Gross CD','Net CD']))
    dictionary = dict(dict_other, **dict_sec)
    
    # Get parameter column and secondary columns (std, tval, pval)
    params = df.Parameter.round(4)
    
    if show == 'std':
        secondary = df.std
    elif show == 'tval':
        secondary = df['T-stat']
    else:
        secondary = df['P-value']

    # Transform secondary column 
    # If stars, then add stars to the parameter list
    if stars:
        stars_count = ['*' * i for i in sum([df.p <0.10, df.p <0.05, df.p <0.01])]
        params = ['{:.4f}{}'.format(val, stars) for val, stars in zip(params,stars_count)]
    secondary_formatted = ['({:.4f})'.format(val) for val in secondary]
    
    # Zip lists to make one list
    results = [val for sublist in list(zip(params, secondary_formatted)) for val in sublist]
    
    # Make pandas dataframe
    ## Make index col (make list of lists and zip)
    lol_params = list(zip([dictionary[val] for val in params.index],\
                          ['{} {}'.format(show, val) for val in [dictionary[val] for val in params.index]]))
    index_row = [val for sublist in lol_params for val in sublist]
    
    # Make df
    results_df = pd.DataFrame(results, index = index_row, columns = [col_label])    
    
    # append N, lenders, MSAs, adj. R2, Depvar, and FEs
    ## Make stats lists and maken index labels pretty
    stats = df[['nobs', 'adj_rsquared', 'j_test']].iloc[0,:].apply(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
    stats.index = [dictionary[val] for val in stats.index]
    
    ### Make df from stats
    stats_df = pd.DataFrame(stats)
    stats_df.columns = [col_label]
    
    ## Append to results_df
    results_df = results_df.append(stats_df)

    return results_df  

def resultsToLatex(results, caption = '', label = ''):
    # Prelim
    function_parameters = dict(na_rep = '',
                               index_names = False,
                               column_format = 'p{3cm}' + 'p{1cm}' * results.shape[1],
                               escape = False,
                               multicolumn = True,
                               multicolumn_format = 'c',
                               caption = caption,
                               label = label)
  
    # To Latex
    return results.to_latex(**function_parameters)


def concatResults(path_list, show = 'pval', stars = False, col_label = None, caption = '', label = ''):
    '''Calls estimationTable and returns a concatenated table '''
    
    list_of_results = []
    for df_path, lab in zip(path_list, col_label):
        # Read df
        df = pd.read_excel(df_path, sheet_name = 'main', index_col = 0)
        df_sec = pd.read_excel(df_path, sheet_name = 'secondary', index_col = 0, dtype={'No. Observations:':'str'})
        df_j = pd.read_excel(df_path, sheet_name = 'j_test', index_col = 0)
        #df_c = pd.read_excel(df_path, sheet_name = 'c_test', index_col = 0)
        
        ## Add stats to df
        df['nobs'] = df_sec.loc['No. Observations:','value']
        df['adj_rsquared'] = df_sec.loc['  Adj. R-squared:    ','value']
        df['j_test'] = df_j.iloc[2,1]
        #df['c_test'] = df_c.iloc[3,1]
        
        # Call estimationTable and append to list
        list_of_results.append(estimationTable(df, show = 'pval', stars = False,\
                                               col_label = lab))

    # Concat all list of dfs to a single df
    results = pd.concat(list_of_results, axis = 1)
    
    # Order results
    ## Get column indexes that are not in fist column and insert in index column 0
    missing_cols = [var for i in range(0,len(list_of_results)-1,1) for var in list_of_results[i+1].index if var not in list_of_results[0].index]
    target_cols = list_of_results[0].index.tolist()
    for i in range(len(missing_cols)):
        target_cols.insert(i + 2, missing_cols[i])
    
    ## order results    
    results = results.loc[target_cols,:]

    # Rename index
    results.index = [result if not show in result else '' for result in results.index]
        
    # Rename columns if multicolumn
    if '|' in results.columns:
        col_names = np.array([string.split('|') for string in results.columns])
        results.columns = pd.MultiIndex.from_arrays([col_names[:,0], col_names[:,1]], names = ['Method','Number'])
    
    # To latex
    results_latex = resultsToLatex(results, caption, label)
    
    ## Add table placement
    location = results_latex.find('\begin{table}\n')
    results_latex = results_latex[:location + len('\begin{table}\n') + 1] + '[th!]' + results_latex[location + len('\begin{table}\n') + 1:]
    
    ## Make the font size of the table footnotesize
    size_string = '\\tiny \n'
    location = results_latex.find('\centering\n')
    results_latex = results_latex[:location + len('\centering\n')] + size_string + results_latex[location + len('\centering\n'):]
    
    # Add midrule above 'Observations'
    size_midrule = '\\midrule'
    location = results_latex.find('\nObservations')
    results_latex = results_latex[:location] + size_midrule + results_latex[location:]
    
    ## Add note to the table
    # TODO: Add std, tval and stars option
    note_string = '\justify\n\\scriptsize{\\textit{Notes.} Estimation results of the Arellano-Bond estimator. The model is estimated in first differences and includes time dummies to obsorb onobserved time effects. We use HAC standard errors. The J-test is the Sargan-Hansen J-test, which tests for over-identifying restrictions. We only report the p-value.}\n'
    location = results_latex.find('\end{tabular}\n')
    results_latex = results_latex[:location + len('\end{tabular}\n')] + note_string + results_latex[location + len('\end{tabular}\n'):]
    
    return results_latex

#------------------------------------------------------------
# Call concatResults
#------------------------------------------------------------

# Set path lists and split
#------------------------------------------------------------
'''NOTE: Since we have many regressions we split the regression tables into 2.
    this results in 2 times 8 for all separate regressions and 2 times 6
    for all regressions with credit derivatives'''

# Full sample, separate 
lst_full_sep = []

for file in glob.glob('Results\GMM_IV\gmmiv_full_sep*.xlsx'):
    lst_full_sep.append(file)
    
lst_full_sep1 = lst_full_sep[:len(lst_full_sep)//2]
lst_full_sep2 = lst_full_sep[len(lst_full_sep)//2:]
    
# Sec sample, separate 
lst_sec_sep = []

for file in glob.glob('Results\GMM_IV\gmmiv_sec_sep*.xlsx'):
    lst_sec_sep.append(file)

lst_sec_sep1 = lst_sec_sep[:len(lst_sec_sep)//2]
lst_sec_sep2 = lst_sec_sep[len(lst_sec_sep)//2:]
    
# Full sample, gross 
lst_full_gross = []

for file in glob.glob('Results\GMM_IV\gmmiv_full_gross*.xlsx'):
    lst_full_gross.append(file)

lst_full_gross1 = lst_full_gross[:len(lst_full_gross)//2]
lst_full_gross2 = lst_full_gross[len(lst_full_gross)//2:]
    
# Sec sample, gross 
lst_sec_gross = []

for file in glob.glob('Results\GMM_IV\gmmiv_sec_gross*.xlsx'):
    lst_sec_gross.append(file)
    
lst_sec_gross1 = lst_sec_gross[:len(lst_sec_gross)//2]
lst_sec_gross2 = lst_sec_gross[len(lst_sec_gross)//2:]
    
# Full sample, net 
lst_full_net = []

for file in glob.glob('Results\GMM_IV\gmmiv_full_net*.xlsx'):
    lst_full_net.append(file)
    
lst_full_net1 = lst_full_net[:len(lst_full_net)//2]
lst_full_net2 = lst_full_net[len(lst_full_net)//2:]
    
# Sec sample, separate 
lst_sec_net = []

for file in glob.glob('Results\GMM_IV\gmmiv_sec_net*.xlsx'):
    lst_sec_net.append(file)
    
lst_sec_net1 = lst_sec_net[:len(lst_sec_net)//2]
lst_sec_net2 = lst_sec_net[len(lst_sec_net)//2:]
    
# Set label lists
#------------------------------------------------------------

lst_labels8 = ['({})'.format(i) for i in range(1,9)]
lst_labels6 = ['({})'.format(i) for i in range(1,7)]

# Set titles and labels
#------------------------------------------------------------

caption_full_sep1 = 'Estimation Results Arellano-Bond Estimator -- Full Sample, Separate Proxies (1)'
label_full_sep1 = 'tab:results_full_sep1'
caption_full_sep2 = 'Estimation Results Arellano-Bond Estimator -- Full Sample, Separate Proxies (2)'
label_full_sep2 = 'tab:results_full_sep2'

caption_sec_sep1 = 'Estimation Results Arellano-Bond Estimator -- Securitizers Only, Separate Proxies (1)'
label_sec_sep1 = 'tab:results_sec_sep1'
caption_sec_sep2 = 'Estimation Results Arellano-Bond Estimator -- Securitizers Only, Separate Proxies (2)'
label_sec_sep2 = 'tab:results_sec_sep2'

caption_full_gross1 = 'Estimation Results Arellano-Bond Estimator -- Full Sample, Gross CD (1)'
label_full_gross1 = 'tab:results_full_gross1'
caption_full_gross2 = 'Estimation Results Arellano-Bond Estimator -- Full Sample, Gross CD (2)'
label_full_gross2 = 'tab:results_full_gross2'

caption_sec_gross1 = 'Estimation Results Arellano-Bond Estimator -- Securitizers Only, Gross CD (1)'
label_sec_gross1 = 'tab:results_sec_gross1'
caption_sec_gross2 = 'Estimation Results Arellano-Bond Estimator -- Securitizers Only, Gross CD (2)'
label_sec_gross2 = 'tab:results_sec_gross2'

caption_full_net1 = 'Estimation Results Arellano-Bond Estimator -- Full Sample, Net CD (1)'
label_full_net1 = 'tab:results_full_net1'
caption_full_net2 = 'Estimation Results Arellano-Bond Estimator -- Full Sample, Net CD (2)'
label_full_net2 = 'tab:results_full_net2'

caption_sec_net1 = 'Estimation Results Arellano-Bond Estimator -- Securitizers Only, Net CD (1)'
label_sec_net1 = 'tab:results_sec_net1'
caption_sec_net2 = 'Estimation Results Arellano-Bond Estimator -- Securitizers Only, Net CD (2)'
label_sec_net2 = 'tab:results_sec_net2'

# Call function
#------------------------------------------------------------

# Separate variables
lst_path_sep = [lst_full_sep1, lst_full_sep2, lst_sec_sep1, lst_sec_sep2]
lst_caption_sep = [caption_full_sep1, caption_full_sep2, caption_sec_sep1, caption_sec_sep2]
lst_label_sep = [label_full_sep1, label_full_sep2, label_sec_sep1, label_sec_sep2]

lst_latex_sep = []
for path, cap, lab in zip(lst_path_sep, lst_caption_sep, lst_label_sep):
    lst_latex_sep.append(concatResults(path, col_label = lst_labels8,\
                                                  caption = cap, label = lab))

# Gross Derivatives
lst_path_gross = [lst_full_gross1, lst_full_gross2, lst_sec_gross1, lst_sec_gross2]
lst_caption_gross = [caption_full_gross1, caption_full_gross2, caption_sec_gross1, caption_sec_gross2]
lst_label_gross = [label_full_gross1, label_full_gross2, label_sec_gross1, label_sec_gross2]

lst_latex_gross = []
for path, cap, lab in zip(lst_path_gross, lst_caption_gross, lst_label_gross):
    lst_latex_gross.append(concatResults(path, col_label = lst_labels6,\
                                                  caption = cap, label = lab))

#  Net Derivative
lst_path_net = [lst_full_net1, lst_full_net2, lst_sec_net1, lst_sec_net2]
lst_caption_net = [caption_full_net1, caption_full_net2, caption_sec_net1, caption_sec_net2]
lst_label_net = [label_full_net1, label_full_net2, label_sec_net1, label_sec_net2]

lst_latex_net = []
for path, cap, lab in zip(lst_path_net, lst_caption_net, lst_label_net):
    lst_latex_net.append(concatResults(path, col_label = lst_labels6,\
                                                  caption = cap, label = lab))

#------------------------------------------------------------
# Save
#------------------------------------------------------------

lst_filenames = ['full_sep1','full_sep2','sec_sep1','sec_sep2',\
                 'full_gross1','full_gross2','sec_gross1','sec_gross2',\
                 'full_net1','full_net2','sec_net1','sec_net2']

for name, latex in zip(lst_filenames, lst_latex_sep + lst_latex_gross + lst_latex_net):
    latex_results = open('Results/GMM_IV/table_gmmiv_{}.tex'.format(name), 'w')
    latex_results.write(latex)
    latex_results.close()
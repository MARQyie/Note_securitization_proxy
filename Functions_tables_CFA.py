# --------------------------------------------
# Functions
# --------------------------------------------

# Data manipulation
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', font_scale=3)

# Function that sets a dictionary of variable names
def getVarDict():
    var_names = ['cr_as_sbo', 'cr_as_rmbs',
                 'cr_as_abs', 'hmda_sec_amount',
                 'cr_secveh_ta',
                 'cr_sec_income', 'cr_serv_fees',
                 'cr_cds_purchased', 'cr_abcp_ta',
                 'cr_abcp_uc_own', 'cr_abcp_ce_own',
                 'cr_abcp_uc_oth', 'cr_abcp_ce_oth']
    var_labels = ['Small Bus. Obl. Transf.', 'Sec. Residential Loans',
                  'Sec. Other Assets', 'Sec. Residential Mortgages',
                  'TA Sec. Vehicles',
                  'Sec. Income', 'Servicing Fees',
                  'CDSs Purchased', 'TA ABCP Conduits',
                  'Un. Com. Own ABCP Conduits',
                  'Credit Exp. Own ABCP Conduits', 'Un. Com. Other ABCP Conduits']
    var_dict = dict(zip(var_names, var_labels))

    return var_dict

# Set heatmap function
def heatmap(matrix, file, annot=True):
    # Set mask
    mask = np.triu(np.ones_like(matrix, dtype=bool), 1)

    # Set aesthetics
    dic_aes = {'mask': mask,
               'annot': annot,
               'center': 0,
               'cmap': 'coolwarm',
               'fmt': '.3f'}

    # Make heatmap
    fig, ax = plt.subplots(figsize=(36, 24))
    sns.heatmap(matrix, **dic_aes)
    plt.tight_layout()

    # Save heatmap
    fig.savefig('Figures/CFA_covariance_maps/' + file)


# Function to make the parameter estimate table pretty
def tidyParamEst(data):
    # Select columns
    data_clean = data.iloc[:, [3, 4, 6, 10]]

    # Set column names
    data_clean.columns = ['Estimates', 'SD', 'P-value', 'Estimates (compl. std.)']

    # Set index names
    ## Prelims
    index_lst = list()
    lst_factors = ('SEC', 'ABCP', 'ABSCDO','LS')
    factor_labels = ('F1', 'F2', 'F1','F3')
    factor_dict = dict(zip(lst_factors, factor_labels))

    var_dict = getVarDict()

    ## Loop over rows to get the correct parameter name
    for index, row in data.iloc[:, :3].iterrows():
        if row['lhs'] in lst_factors:
            if row['rhs'] in lst_factors:
                if row['lhs'] == row['rhs']:
                    index_lst.append(['Variance ($\phi$)', str(factor_dict[row['lhs']]), np.nan])
                else:
                    index_lst.append(
                        ['Covariance ($\phi$)', str(factor_dict[row['lhs']]), str(factor_dict[row['rhs']])])
                # index_lst.append('$\phi_{' + str(row['lhs']) + ',' +  str(row['rhs']) +'}$')
            else:
                index_lst.append(['Loading ($\lambda$)', str(factor_dict[row['lhs']]), str(var_dict[row['rhs']])])
                # index_lst.append('$\lambda_{' + str(row['lhs']) + ',' +  str(var_dict[row['rhs']]) +'}$')
        else:
            if row['lhs'] == row['rhs']:
                index_lst.append(['Variance ($\delta$)', str(var_dict[row['lhs']]), np.nan])
            else:
                index_lst.append(['Covariance ($\delta$)', str(var_dict[row['lhs']]), str(var_dict[row['rhs']])])
            # index_lst.append('$\delta_{' + str(var_dict[row['lhs']]) + ',' +  str(var_dict[row['rhs']]) +'}$')

    ## Change Index
    data_index = list(map(list, zip(*index_lst)))
    data_clean = data_clean.assign(LHS=data_index[1],
                                   RHS=data_index[2])

    data_clean.set_index(['LHS', 'RHS'], inplace=True)

    ## Remove Index names
    data_clean.index.names = [None, None]

    return data_clean


# Function to make latex table from pandas dataframe
def table2Latex(data, options, notes, string_size):
    # Get latex table and
    latex_table = data.to_latex(na_rep='', float_format='{:0.4f}'.format,
                                longtable=False, multicolumn=True,
                                multicolumn_format='c', escape=False,
                                **options)

    # add notes to bottom of the table
    location_mid = latex_table.find('\end{tabular}')
    latex_table = latex_table[:location_mid] + notes + latex_table[location_mid:]

    # adjust sting size
    location_size = latex_table.find('\centering\n')
    latex_table = latex_table[:location_size + len('\centering\n')] + string_size + latex_table[location_size + len(
        '\centering\n'):]

    # Insert multicolumns for the fit measures
    if data.index[0] == 'No. Params':
        # Absolute fit
        if 'DLS' in data.columns or 'ULS' in data.columns:
            abs_fit = '& \\\\ \n \\multicolumn{2}{l}{\\textbf{Absolute fit}} \\\\ \n'
            location = latex_table.find('$\chi^2$')
            latex_table = latex_table[:location] + abs_fit + latex_table[location:]
        else:
            abs_fit = '& \\\\ \n \\multicolumn{2}{l}{\\textbf{Absolute fit}} \\\\ \n'
            location = latex_table.find('Yuan--Bentler $\chi^2$')
            latex_table = latex_table[:location] + abs_fit + latex_table[location:]

        # Parsimonious fit
        pars_fit = '& \\\\ \n \\multicolumn{2}{l}{\\textbf{Parsimonious fit}} \\\\ \n'
        location = latex_table.find('RMSEA')
        latex_table = latex_table[:location] + pars_fit + latex_table[location:]

        # Comparative and relative fit
        cr_fit = '& \\\\ \n \\multicolumn{2}{l}{\\textbf{Comparative and relative fit}} \\\\ \n'
        location = latex_table.find('CFI')
        latex_table = latex_table[:location] + cr_fit + latex_table[location:]

        # Other
        location = latex_table.find('AIC')
        if location != -1:
            oth_fit = '& \\\\ \n \\multicolumn{2}{l}{\\textbf{Other}} \\\\ \n'
            latex_table = latex_table[:location] + oth_fit + latex_table[location:]

    # Insert multicolumns for the parameter estimates for two-factor model
    if (data.index[0][0] == 'F1') and (data.index[-1][0] != 'F3'):
        # Factor loadings
        factor_loadings = ' \n \\multicolumn{6}{l}{\\textbf{Factor loadings}} \\\\ \n'
        location = latex_table.find('F1')
        latex_table = latex_table[:location] + factor_loadings + latex_table[location:]

        # Error covariances (unique covariances)
        error_cov = '&&&&& \\\\ \n \\multicolumn{6}{l}{\\textbf{Error covariances}} \\\\ \n'
        location = latex_table.find('Un. Com. Own ABCP Conduits & Un. Com. Other ABCP Conduits')
        latex_table = latex_table[:location] + error_cov + latex_table[location:]

        # Unique variances
        if data.index.get_level_values(0).str.contains('F3').any():
            unique_var = '\n&&&&& \\\\ \n \\multicolumn{6}{l}{\\textbf{Unique variances}} \\\\'
            location = latex_table.find('\nSec. Residential Loans &')
            latex_table = latex_table[:location] + unique_var + latex_table[location:]
        else:
            unique_var = '\n&&&&& \\\\ \n \\multicolumn{6}{l}{\\textbf{Unique variances}} \\\\'
            location = latex_table.find('\nSmall Bus. Obl. Transf. &')
            latex_table = latex_table[:location] + unique_var + latex_table[location:]

        # Factor variances
        fac_var = '&&&&& \\\\ \n \\multicolumn{6}{l}{\\textbf{Factor variances}} \\\\ \n'
        location = latex_table.replace('F1', 'XX', 1).find('F1')  # trick to find the second ABSCDO
        latex_table = latex_table[:location] + fac_var + latex_table[location:]

        # Factor covariances
        fac_cov = '&&&&& \\\\ \n \\multicolumn{6}{l}{\\textbf{Factor covariances}} \\\\ \n'
        location = latex_table.find('F1 & F2')
        latex_table = latex_table[:location] + fac_cov + latex_table[location:]

    # Insert multicolumns for the parameter estimates for one-factor model
    if data.index[-1][0] == 'F3':
        # Factor loadings
        factor_loadings = ' \n \\multicolumn{6}{l}{\\textbf{Factor loadings}} \\\\ \n'
        location = latex_table.find('F1')
        latex_table = latex_table[:location] + factor_loadings + latex_table[location:]

        # Error covariances (unique covariances)
        error_cov = '&&&&& \\\\ \n \\multicolumn{6}{l}{\\textbf{Error covariances}} \\\\ \n'
        location = latex_table.find('\nUn. Com. Own ABCP Conduits & Un. Com. Other ABCP Conduits')
        latex_table = latex_table[:location] + error_cov + latex_table[location:]

        # Unique variances
        unique_var = '\n&&&&& \\\\ \n \\multicolumn{6}{l}{\\textbf{Unique variances}} \\\\'
        location = latex_table.find('\nSmall Bus. Obl. Transf. &')
        latex_table = latex_table[:location] + unique_var + latex_table[location:]

        # Factor variances
        fac_var = '&&&&& \\\\ \n \\multicolumn{6}{l}{\\textbf{Factor variances}} \\\\ \n'
        location = latex_table.replace('F1', 'XX', 2).find('F1')
        latex_table = latex_table[:location] + fac_var + latex_table[location:]

        # Insert multi columns in the communality table
    if data.columns[-1] == 'Source':
        # ABS sec
        col = ' \\multicolumn{3}{l}{\\textbf{Loading only on F1}} \\\\ \n'
        location = latex_table.find('Small Bus. Obl. Transf.')
        latex_table = latex_table[:location] + col + latex_table[location:]

        # Both factors
        col = '&&&\\\\ \n \\multicolumn{3}{l}{\\textbf{Loading on both F1 and F2}} \\\\ \n'
        location = latex_table.find('Sec. Income')
        latex_table = latex_table[:location] + col + latex_table[location:]

        # ABCP factors
        col = '&&&\\\\ \n \\multicolumn{3}{l}{\\textbf{Loading only on F2}} \\\\ \n'
        location = latex_table.find('TA ABCP Conduits')
        latex_table = latex_table[:location] + col + latex_table[location:]

    return latex_table


def tidyFitInd(data):
    # Get indices
    lst_fi = ['npar', 'df', 'baseline.df', 'chisq.scaled',
              'pvalue.scaled','srmr', 'rmsea.robust',
              'rni.robust', 'cfi.robust', 'tli.robust', 'aic', 'bic']

    if not('aic' in data.index and 'bic' in data.index):
        lst_fi = lst_fi[:-2]
    data_clean = data.loc[lst_fi, :]

    # Rename index
    lst_fi_labels = ['No. Params', 'DoF', 'DoF baseline','Yuan--Bentler $\chi^2$',
                     'p-val Yuan--Bentler $\chi^2$', 'SRMR', 'RMSEA',
                     'RNI', 'CFI', 'TLI',
                     'AIC', 'BIC']

    if not('aic' in data.index and 'bic' in data.index):
        lst_fi_labels = lst_fi_labels[:-2]

    data_clean.index = lst_fi_labels

    # Rename columns
    if data.shape[1] == 3:
        data_clean.columns = ['Theory', 'EFA', 'Combined']
    if data.shape[1] == 2:
        data_clean.columns = ['(1)', '(2)']
    else:
        data_clean.rename(columns={'Unnamed: 1': 'Index'}, inplace=True)

    # Format values
    data_clean = data_clean.applymap(lambda x: '{:,.4f}'.format(x) if not 24 <= x <= 66 else '{:,.0f}'.format(x))

    return data_clean


# Function to make the modification table table pretty
def pivotTableMI(data, value = 'mi'):
    # Pivot table
    data_pivot = data.pivot(index='rhs', columns='lhs', values=value)

    # Split table into two: 1) Loadings, 2) Covariances
    # Only for two factor models
    var_names = list(getVarDict().keys())

    if data_pivot.columns.str.contains('ABSCDO|ABCP').any() and not data_pivot.columns.str.contains('LS').any():
        data_load = data_pivot.iloc[:,:2]
        data_cov = data_pivot.iloc[:,2:]

        # Sort index/columns tables
        if data_cov.shape[0] == 12:
            try:
                data_load = data_load.loc[var_names, :]
                data_cov = data_cov.loc[var_names[1:], var_names[:-1]]
            except:
                var_names = [name for name in var_names if
                         name not in ['cr_cds_purchased']] + ['cr_cds_purchased']
                data_load = data_load.loc[var_names, :]
                data_cov = data_cov.loc[var_names[1:], var_names[:-1]]
        else:
            var_names = [name for name in var_names if
                         name not in ['cr_as_sbo', 'hmda_sec_amount', 'cr_cds_purchased']] + ['cr_cds_purchased']
            data_load = data_load.loc[var_names, :]
            data_cov = data_cov.loc[var_names[1:], var_names[:-1]]

        # Remove index and column names
        data_load.index.name = None
        data_load.columns.name = None

        data_cov.index.name = None
        data_cov.columns.name = None
    elif data_pivot.columns.str.contains('LS').any():
        data_load = data_pivot.iloc[:, :3]
        data_cov = data_pivot.iloc[:, 3:]

        # Sort index/columns tables
        data_load = data_load.loc[var_names, :]
        var_names = [name for name in var_names if
                     name not in ['cr_as_sbo']] + ['cr_as_sbo']
        data_cov = data_cov.loc[var_names[1:], var_names[:-1]]

        # Remove index and column names
        data_load.index.name = None
        data_load.columns.name = None

        data_cov.index.name = None
        data_cov.columns.name = None
    else:
        data_load = None
        data_cov = data_pivot

        # Sort index/columns
        data_cov = data_cov.loc[var_names[1:], var_names[:-1]]

        # Remove index and column names
        data_cov.index.name = None
        data_cov.columns.name = None

    # return tables
    return data_load, data_cov

def tidyModInd(data):
    # Get tables for MI and EPC
    data_mi_load, data_mi_cov = pivotTableMI(data, 'mi')
    data_epc_load, data_epc_cov = pivotTableMI(data, 'sepc.all')

    # Set column and index names
    var_dict = getVarDict()

    if isinstance(data_mi_load, pd.DataFrame):
        if data_mi_load.shape[1] == 2:
            data_mi_load.columns = [('Mod. Indices','F2'),('Mod. Indices','F1')]
            data_mi_load.rename(var_dict, axis=0, inplace = True)

            data_epc_load.columns = [('EPC (Compl. Std.)', 'F2'), ('EPC (Compl. Std.)', 'F1')]
            data_epc_load.rename(var_dict, axis=0, inplace=True)
        else:
            data_mi_load.columns = [('Mod. Indices', 'F2'), ('Mod. Indices', 'F1'), ('Mod. Indices', 'F3')]
            data_mi_load.rename(var_dict, axis=0, inplace=True)

            data_epc_load.columns = [('EPC (Compl. Std.)', 'F2'), ('EPC (Compl. Std.)', 'F1'), ('EPC (Compl. Std.)', 'F3')]
            data_epc_load.rename(var_dict, axis=0, inplace=True)

        # Concat loadings table and sort columns
        data_load = pd.concat([data_mi_load, data_epc_load], axis=1)
        data_load = data_load.reindex(sorted(data_load.columns), axis=1)

        # Remove nan-columns in loadings table
        data_load = data_load.loc[~data_load.isnull().all(axis=1), :]

        # Make multicolumn for loadings table
        data_load.columns = pd.MultiIndex.from_tuples(data_load.columns,
                                                      names=['', ''])
    else:
        data_load = None

    data_mi_cov.rename(var_dict, axis=0, inplace=True)
    data_mi_cov.rename(var_dict, axis=1, inplace=True)

    data_epc_cov.rename(var_dict, axis=0, inplace=True)
    data_epc_cov.rename(var_dict, axis=1, inplace=True)

    # Return data
    return data_load, data_mi_cov, data_epc_cov

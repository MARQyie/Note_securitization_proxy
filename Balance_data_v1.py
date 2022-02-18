# -----------------------------------
# Balance the 2011--2017 dataset
# Mark van der Plaat 01-02-2022
# -----------------------------------

# This file loads the original unbalanced
# dataset and balances

# -----------------------------------
# Load Packages
# -----------------------------------

# Set Path
import os
os.chdir(r'D:\RUG\PhD\Materials_papers\01-Note_on_securitization')

# Data manipulation
import pandas as pd
import numpy as np

# -----------------------------------
# Load DataFrame
# -----------------------------------

df = pd.read_csv('Data/df_sec_note_20112017.csv', index_col=0)

# -----------------------------------
# Balance DataFrame
# -----------------------------------

# Get number of years from DataFrame
years = df.date.nunique()

# Get IDRSSDs where count equals years
idrssds = df.IDRSSD.value_counts()[df.IDRSSD.value_counts() == years].index.tolist()

# Balance the DataFrame
df_balanced = df[df.IDRSSD.isin(idrssds)]

# -----------------------------------
# Add percentiles column
# -----------------------------------

# NOTE: We use these variables in the R-script CFA_20112017_balanced_v1.R.
# All percentiles are based on the average total assets of banks
# To be sure, we add several percentiles columns.

# Calculate average total assets
mean_ta = df_balanced.groupby('IDRSSD').ta.mean()

# Get percentiles
percentiles_4 = pd.qcut(mean_ta, q = 4, labels = False)
percentiles_4.rename('percentiles_4', inplace = True)

percentiles_5 = pd.qcut(mean_ta, q = 5, labels = False)
percentiles_5.rename('percentiles_5', inplace = True)

percentiles_6 = pd.qcut(mean_ta, q = 6, labels = False)
percentiles_6.rename('percentiles_6', inplace = True)

# Merge with df_balanced
df_balanced = df_balanced.merge(percentiles_4, on = 'IDRSSD')
df_balanced = df_balanced.merge(percentiles_5, on = 'IDRSSD')
df_balanced = df_balanced.merge(percentiles_6, on = 'IDRSSD')

# Add dummy variables based on quantiles total Assets
ids_quant_ta95 = mean_ta[mean_ta < mean_ta.quantile(.95)].index.tolist()
dum_ta95 = [1 if row['IDRSSD'] in ids_quant_ta95 else 0 for i, row in df_balanced.iterrows()]
df_balanced['dum_ta95'] = dum_ta95

ids_quant_ta99 = mean_ta[mean_ta < mean_ta.quantile(.99)].index
dum_ta99 = [1 if row['IDRSSD'] in ids_quant_ta99 else 0 for i, row in df_balanced.iterrows()]
df_balanced['dum_ta99'] = dum_ta99

# Add dummy that equals one when a bank has at least 3 non-zero values for all proxies
# Set var names
var_names = ['cr_as_sbo','cr_as_rmbs','cr_as_abs','hmda_sec_amount',
            'cr_secveh_ta','cr_sec_income','cr_serv_fees','cr_cds_purchased',
            'cr_abcp_ta','cr_abcp_uc_own','cr_abcp_ce_own', 'cr_abcp_uc_oth']

# Determine non-zero values per bank and per columns
nonzeroes_idrssd = df_balanced.groupby('IDRSSD')[var_names].agg(lambda x: x.ne(0).sum())
nonzeroes_date = df_balanced.groupby('date')[var_names].agg(lambda x: x.ne(0).sum())

nonzeroes_idrssd.to_csv('Data/df_nonzeroes_idrssd.csv')
nonzeroes_date.to_csv('Data/df_nonzeroes_date.csv')

# -----------------------------------
# Aggregate data per year
# -----------------------------------

# Unweighted aggregated data
df_agg = df_balanced.groupby('IDRSSD').sum()

# Check some statistics
means_agg = df_agg.mean()
vars_agg = df_agg.var()
groupbydata_agg = df_agg.groupby('date')[var_names].agg(lambda x: x.ne(0).sum())

# Weighted aggregated data
# NOTE weights are: TA_i / sum_i TA_i * n
df_agg_weighted = df_agg.multiply(df_agg.ta / df_agg.ta.sum() * df_agg.shape[0], axis = 0)
# Run to check the sum of the weights: (df_agg.ta / df_agg.ta.sum()).sum()

# Scale balanced with TA_i
df_agg_ta = df_agg.divide(df_agg.ta, axis = 0)

# Drop columns in df_agg and df_agg_weighted
df_agg.drop(columns = ['date','ta','percentiles_4','percentiles_5','percentiles_6','dum_ta95','dum_ta99'],
            inplace = True)
df_agg_weighted.drop(columns = ['date','ta','percentiles_4','percentiles_5','percentiles_6','dum_ta95','dum_ta99'],
            inplace = True)
df_agg_ta.drop(columns = ['date','ta','percentiles_4','percentiles_5','percentiles_6','dum_ta95','dum_ta99'],
            inplace = True)

# -----------------------------------
# Save DataFrame
# -----------------------------------

df_balanced.to_csv('Data/df_sec_note_20112017_balanced.csv')
df_agg.to_csv('Data/df_sec_note_20112017_balanced_agg.csv')
df_agg_weighted.to_csv('Data/df_sec_note_20112017_balanced_agg_weighted.csv')
df_agg_ta.to_csv('Data/df_sec_note_20112017_balanced_agg_ta.csv')
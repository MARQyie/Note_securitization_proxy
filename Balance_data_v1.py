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

# -----------------------------------
# Save DataFrame
# -----------------------------------

df_balanced.to_csv('Data/df_sec_note_20112017_balanced.csv')
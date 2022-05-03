# --------------------------------------------
# Subsampling for simulation study
# Mark van der Plaat
# April 2022
# --------------------------------------------

'''This script creates weighted bootstrap samples
    for a simulation study into the effects of
    zeros in the sample.'''

# --------------------------------------------
# Import Packages
# --------------------------------------------

# Data manipulation
import pandas as pd
import numpy as np
from scipy import stats

# Plotting
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

sns.set(style='whitegrid', font_scale=2.75)

# Set WD
import os

os.chdir(r'D:\RUG\PhD\Materials_papers\01-Note_on_securitization')

# --------------------------------------------
# Load data
# --------------------------------------------

df = pd.read_csv('Data\df_sec_note_binary_20112017.csv', index_col=0)

# Only use the relevant columns
var_names = ['cr_as_sbo','cr_as_rmbs',
             'cr_as_abs','hmda_sec_amount',
             'cr_secveh_ta','cr_cds_purchased',
             'cr_abcp_ta','cr_abcp_uc_own',
             'cr_abcp_ce_own','cr_abcp_uc_oth']

df = df[['IDRSSD','date'] + var_names]

# Add column to base the weights
df['weights'] = (df[var_names].sum(axis = 1) >0) * 1

# --------------------------------------------
# Get subsamples and save
# --------------------------------------------

# prelims
dict_args = dict(
                n = df.shape[0],
                replace = True,
                random_state = 10,
                ignore_index = True)

# Set weight list
lst_weights = [(df.weights - 1).abs() * (i/10) + 1 for i in range(1,10+1)] # Give the zeros more weight with increments of 10%

# Set file names list
file_names = ['Data/Data_weighted_bootstrap_{}.csv'.format(i + 100) for i in range(10, 100+10, 10)]

for weights, file_name in zip(lst_weights,file_names):
    df_wb = df.sample(weights = weights, **dict_args)
    df_wb.to_csv(file_name)
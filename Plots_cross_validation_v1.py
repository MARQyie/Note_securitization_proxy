# --------------------------------------------
# Plots for cross validation
# Mark van der Plaat
# September 2020 -- Update: November 2021
# --------------------------------------------

# --------------------------------------------
# Import Packages
# --------------------------------------------

# Data manipulation
import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid', font_scale=2)

# Set WD
import os
os.chdir(r'D:\RUG\PhD\Materials_papers\01-Note_on_securitization')

#--------------------------------------------
# Load data
#--------------------------------------------

# Get file names from the directory
lst_directory = os.listdir('Results/Cross_validation')
lst_fitfiles = [file for file in lst_directory if '_fit_' in file]
lst_r2files = [file for file in lst_directory if '_r2_' in file]

# Load data and concat
def readFiles(files):
    data_lst = list()

    for file in files:
        # Load file
        data = pd.read_csv('Results/Cross_validation/' + file, index_col = 0).T

        # Append to list
        data_lst.append(data)

    # Make df and reset index
    dataframe = pd.concat(data_lst)
    dataframe.reset_index(drop = True, inplace = True)

    return dataframe

df_fit = readFiles(lst_fitfiles)
df_r2 = readFiles(lst_r2files)

#--------------------------------------------
# Plot fit indices
#--------------------------------------------

# Plot SRMR (absolute fit)
fig, ax = plt.subplots(figsize=(8,8))
ax.set(ylabel='Value', xlabel = 'SRMR')
sns.violinplot(y = 'srmr',
               data = df_fit,
               inner = None,
               alpha=0.1)
sns.boxplot(y = 'srmr',
            data = df_fit,
            boxprops={'zorder': 2}).set(ylabel = '')
plt.tight_layout()

fig.savefig('Results/Cross_validation/Box_absolute_fit.png')

# Plot RMSEA and RNI (parsimonious fit)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
ax1.set(ylabel='Value', xlabel = 'RMSEA')
ax2.set(ylabel='', xlabel = 'RNI')
sns.violinplot(ax=ax1,
               y = 'rmsea.robust',
               data = df_fit,
               inner = None,
               alpha=0.1)
sns.boxplot(ax=ax1,
            y = 'rmsea.robust',
            data = df_fit,
            boxprops={'zorder': 2}).set(ylabel = '')
sns.violinplot(ax=ax2,
               y = 'rni.robust',
               data = df_fit,
               inner = None,
               alpha=0.1)
sns.boxplot(ax=ax2,
            y = 'rni.robust',
            data = df_fit,
            boxprops={'zorder': 2}).set(ylabel = '')
plt.tight_layout()
fig.savefig('Results/Cross_validation/Box_parsimonious_fit.png')

# Plot CFI and TLI (comparative and relative fit)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
ax1.set(ylabel='Value', xlabel = 'CFI')
ax2.set(ylabel='', xlabel = 'TLI')
sns.violinplot(ax=ax1,
               y = 'cfi.robust',
               data = df_fit,
               inner = None,
               alpha=0.1)
sns.boxplot(ax=ax1,
            y = 'cfi.robust',
            data = df_fit,
            boxprops={'zorder': 2}).set(ylabel = '')
sns.violinplot(ax=ax2,
               y = 'tli.robust',
               data = df_fit,
               inner = None,
               alpha=0.1)
sns.boxplot(ax=ax2,
            y = 'tli.robust',
            data = df_fit,
            boxprops={'zorder': 2}).set(ylabel = '')
plt.tight_layout()
fig.savefig('Results/Cross_validation/Box_comparative_fit.png')

#--------------------------------------------
# Plot communalities
#--------------------------------------------

# Set labels
lst_vars = df_r2.columns.tolist()
lst_labels = ['Small Bus. Obl. Transf.','Sec. Residential Loans',
              'Sec. Other Assets','Sec. Residential Mortgages',
              'TA Sec. Vehicles' ,
              'Sec. Income', 'Servicing Fees',
              'CDSs Purchased','TA ABCP Conduits',
              'Un. Com. Own ABCP Conduits',
              'Credit Exp. Own ABCP Conduits','Un. Com. Other ABCP Conduits']

# Plot
fig = plt.figure(figsize=(24,24))
for i in range(len(lst_vars)):
    ax = fig.add_subplot(4,3,i+1)
    ax.set(ylabel='', xlabel=lst_labels[i])
    sns.violinplot(ax=ax,
                   y=lst_vars[i],
                   data=df_r2,
                   inner=None,
                   alpha=0.1)
    sns.boxplot(ax=ax,
                y=lst_vars[i],
                data=df_r2,
                boxprops={'zorder': 2}).set(ylabel='')
plt.tight_layout()
fig.savefig('Results/Cross_validation/Box_communalities.png')


# -----------------------------------
# Distribution plots
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

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', font_scale=2)

# -----------------------------------
# Load DataFrames
# -----------------------------------

nonzeroes_idrssd = pd.read_csv('Data/df_nonzeroes_idrssd.csv', index_col = 0)
nonzeroes_date = pd.read_csv('Data/df_nonzeroes_date.csv', index_col = 0)

# -----------------------------------
# Plot proxes over IDRSSD
# -----------------------------------

# Set labels
lst_vars = nonzeroes_idrssd.columns.tolist()
lst_labels = ['Small Bus. Obl. Transf.','Sec. Residential Loans',
              'Sec. Other Assets','Sec. Residential Mortgages',
              'TA Sec. Vehicles' ,
              'Sec. Income', 'Servicing Fees',
              'CDSs Purchased','TA ABCP Conduits',
              'Un. Com. Own ABCP Conduits',
              'Credit Exp. Own ABCP Conduits','Un. Com. Other ABCP Conduits']

# Plot distributions
fig = plt.figure(figsize=(24,24))
for i in range(len(lst_vars)):
    ax = fig.add_subplot(4,3,i+1)
    ax.set(ylabel='', xlabel=lst_labels[i])
    sns.distplot(ax=ax,
                 a=nonzeroes_idrssd.iloc[:,i],
                 hist=True,
                 kde=True,
                 bins=80)
plt.tight_layout()
fig.savefig('Figures/Plot_dist_idrssd.png')

# Box plots
fig = plt.figure(figsize=(24,24))
for i in range(len(lst_vars)):
    ax = fig.add_subplot(4,3,i+1)
    ax.set(ylabel='', xlabel=lst_labels[i])
    sns.violinplot(ax=ax,
                   y=lst_vars[i],
                   data=nonzeroes_idrssd,
                   inner='box',
                   alpha=0.1)
plt.tight_layout()
fig.savefig('Figures/Plot_box_idrssd.png')

# -----------------------------------
# Plot proxies over time
# -----------------------------------

# Bar plot (over time)
fig = plt.figure(figsize=(24,24))
for i in range(len(lst_vars)):
    ax = fig.add_subplot(4,3,i+1)
    ax.set(ylabel='', xlabel=lst_labels[i])
    ax.bar(nonzeroes_date.index,
               nonzeroes_date.iloc[:,i])
plt.tight_layout()
fig.savefig('Figures/Plot_bar_date.png')

# Bar plot (over proxies)
fig = plt.figure(figsize=(24,24))
for i in range(nonzeroes_date.shape[0]):
    ax = fig.add_subplot(3,3,i+1)
    ax.set(ylabel='', xlabel=lst_labels[i])
    ax.bar(nonzeroes_date.columns,
           nonzeroes_date.iloc[i,:])
    plt.xticks(rotation=90)
plt.tight_layout()
fig.savefig('Figures/Plot_bar_proxy.png')

# Box plots
fig = plt.figure(figsize=(24,24))
for i in range(len(lst_vars)):
    ax = fig.add_subplot(4,3,i+1)
    ax.set(ylabel='', xlabel=lst_labels[i])
    sns.violinplot(ax=ax,
                   y=lst_vars[i],
                   data=nonzeroes_date,
                   inner='box',
                   alpha=0.1)
plt.tight_layout()
fig.savefig('Figures/Plot_box_date.png')
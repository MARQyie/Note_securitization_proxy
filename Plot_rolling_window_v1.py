# --------------------------------------------
# Plots for rolling window
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

sns.set(style='white', font_scale=2)

# Set WD
import os
os.chdir(r'D:\RUG\PhD\Materials_papers\01-Note_on_securitization')

#--------------------------------------------
# Load data
#--------------------------------------------

# Get file names from the directory
lst_directory = os.listdir('Results')
lst_fitfiles = [file for file in lst_directory if '_fit_' in file and 'balanced_loopyear_' in file]
lst_r2files = [file for file in lst_directory if '_r2_' in file and 'balanced_loopyear_' in file]

lst_fitfiles_scaled = [file for file in lst_directory if '_fit_' in file and 'balancedscaled_loopyear_' in file]
lst_r2files_scaled = [file for file in lst_directory if '_r2_' in file and 'balancedscaled_loopyear_' in file]

lst_fitfiles_alt = [file for file in lst_directory if '_fit_' in file and 'balanced_loopyearalt_' in file]
lst_r2files_alt = [file for file in lst_directory if '_r2_' in file and 'balanced_loopyearalt_' in file]

# Load data and concat
def readFiles(files):
    data_lst = list()

    for file in files:
        # Load file
        data = pd.read_csv('Results/' + file, index_col = 0).T

        # Append to list
        data_lst.append(data)

    # Make df and reset index
    dataframe = pd.concat(data_lst)
    dataframe.reset_index(drop = True, inplace = True)

    return dataframe

df_fit = readFiles(lst_fitfiles)
df_r2 = readFiles(lst_r2files)
df_fit_scaled = readFiles(lst_fitfiles_scaled)
df_r2_scaled = readFiles(lst_r2files_scaled)
df_fit_alt = readFiles(lst_fitfiles_alt)
df_r2_alt = readFiles(lst_r2files_alt)

#--------------------------------------------
# Plot fit indices
#--------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
ax1.set(ylabel='', xlabel = 'Middle Year Rolling Window')
ax2.set(ylabel='',  xlabel = 'Middle Year Rolling Window')
ax1.plot(range(2012,2017), df_fit['srmr'],
        color = 'black',
        linestyle = '-',
        label = 'SRMR')
ax1.plot(range(2012,2017), df_fit['rmsea.robust'],
        color='black',
        linestyle='--',
        label = 'RMSEA')
ax1.legend()

# Note: Do not plot the RNI, because it is equivalent to CFI
ax2.plot(range(2012,2017), df_fit['cfi.robust'],
         color='black',
         linestyle='-',
         label = 'CFI')
ax2.plot(range(2012,2017), df_fit['tli.robust'],
         color='black',
         linestyle='--',
         label = 'TLI')
ax2.legend()
plt.tight_layout()

plt.savefig('Figures/Plot_rollingwindow_fit.png')

# Scaled by TA
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
ax1.set(ylabel='', xlabel = 'Middle Year Rolling Window')
ax2.set(ylabel='',  xlabel = 'Middle Year Rolling Window')
ax1.plot(range(2012,2017), df_fit_scaled['srmr'],
        color = 'black',
        linestyle = '-',
        label = 'SRMR')
ax1.plot(range(2012,2017), df_fit_scaled['rmsea.robust'],
        color='black',
        linestyle='--',
        label = 'RMSEA')
ax1.legend()

# Note: Do not plot the RNI, because it is equivalent to CFI
ax2.plot(range(2012,2017), df_fit_scaled['cfi.robust'],
         color='black',
         linestyle='-',
         label = 'CFI')
ax2.plot(range(2012,2017), df_fit_scaled['tli.robust'],
         color='black',
         linestyle='--',
         label = 'TLI')
ax2.legend()
plt.tight_layout()

plt.savefig('Figures/Plot_rollingwindow_fit_scaled.png')

#--------------------------------------------
# Plot fit indices
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
    ax.plot(range(2012, 2017), df_r2[lst_vars[i]],
             color='black',
             linestyle='-',
             label='CFI')
plt.tight_layout()
plt.savefig('Figures/Plot_rollingwindow_r2.png')

# Plot scaled
fig = plt.figure(figsize=(24,24))
for i in range(len(lst_vars)):
    ax = fig.add_subplot(4,3,i+1)
    ax.set(ylabel='', xlabel=lst_labels[i])
    ax.plot(range(2012, 2017), df_r2_scaled[lst_vars[i]],
             color='black',
             linestyle='-',
             label='CFI')
plt.tight_layout()
plt.savefig('Figures/Plot_rollingwindow_r2_scaled.png')

# Plot with TA
lst_vars = df_r2.columns.tolist() + ['ta']
lst_labels = ['Small Bus. Obl. Transf.','Sec. Residential Loans',
              'Sec. Other Assets','Sec. Residential Mortgages',
              'TA Sec. Vehicles' ,
              'Sec. Income', 'Servicing Fees',
              'CDSs Purchased','TA ABCP Conduits',
              'Un. Com. Own ABCP Conduits',
              'Credit Exp. Own ABCP Conduits','Un. Com. Other ABCP Conduits',
              'Total Assets']

fig = plt.figure(figsize=(24,24))
for i in range(len(lst_vars)):
    ax = fig.add_subplot(4,4,i+1)
    ax.set(ylabel='', xlabel=lst_labels[i])
    ax.plot(range(2012, 2017), df_r2_alt[lst_vars[i]],
             color='black',
             linestyle='-',
             label='CFI')
plt.tight_layout()
plt.savefig('Figures/Plot_rollingwindow_r2_ta.png')
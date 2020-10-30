#--------------------------------------------
# KNN for Note
# Mark van der Plaat
# September 2020
#--------------------------------------------

#--------------------------------------------
# Import Packages
#--------------------------------------------
    
# Data manipulation
import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid', font_scale = 2)

# Machine learning packages
from sklearn.cluster import KMeans

# Set WD
import os
os.chdir(r'D:\RUG\PhD\Materials_papers\01-Note_on_securitization')

#--------------------------------------------
# Load data and merge
#--------------------------------------------

# Load data
## Securitization data
df_sec = pd.read_csv('Data\df_sec_note.csv', index_col = 0)

## Other data
df_oth = pd.read_csv('Data\df_ri_rc_note.csv')

# Merge data
df = df_sec.merge(df_oth, how = 'inner', on = ['date','IDRSSD'])

#--------------------------------------------
# Prelims
#--------------------------------------------

# Variable names
## Call Reports
vars_cr = df.columns[df.columns.str.contains('cr')].tolist()

## HMDA
vars_hmda = df.columns[df.columns.str.contains('hmda')].tolist()

## Total
vars_tot = vars_cr + vars_hmda + ['ta']

# Set variables
vars_oth = ['t1_reglev', 't1_regcap', 'cap_ratio', 'dep_ratio', 'loan_ratio',\
            'ra_ratio', 'ci_ratio', 'agri_ratio', 'cons_ratio', 'othl_ratio',\
            'loan_hhi', 'roa', 'liq_ratio', 'cti', 'nii_nor', 'rwata', 'npl',\
            'co_ratio', 'all_ratio', 'prov_ratio']

# Drop na
df.dropna(inplace = True)

#--------------------------------------------
# KNN
#--------------------------------------------

# Determine number of clusters
intertia = np.zeros(10, dtype=np.float)

for cluster in range(1,11):
    kmeans_det = KMeans(n_clusters=cluster)
    kmeans_det.fit(df[vars_tot + vars_oth])
    
    intertia[cluster - 1] = kmeans_det.inertia_
    
## Plot
fig, ax = plt.subplots(figsize=(20,12)) 
ax.set(ylabel='Within-Cluster Sum-of-Squares (1e20)', xlabel = 'Number of Clusters')
ax.plot(np.arange(1,11,1),intertia / 1e20)
plt.xticks(np.arange(1,11,1))
plt.tight_layout()
fig.savefig('Figures/Cluster_analysis/WCSS.png')

'''NOTE There is a clear elbow at three clusters. Proceed.
    '''

# Run KNN algorithm with three clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(df[vars_tot + vars_oth])

## Get centroids and transform to pandas DataFrame
centroids  = kmeans.cluster_centers_ 
centroids = pd.DataFrame(centroids, columns = vars_tot + vars_oth, index = ['C1','C2','C3']).T

## Get number of observations per cluster
cluster_n = np.unique(kmeans.labels_, return_counts = True)[1]

#--------------------------------------------
# Latex table method
#--------------------------------------------

def resultsToLatex(results, caption = '', label = '', size_string = '\\scriptsize \n',  note_string = None, sidewaystable = False):
    # Prelim
    function_parameters = dict(na_rep = '',
                               index_names = True,
                               column_format = 'p{3.5cm}' + 'p{2cm}' * results.shape[1],
                               escape = False,
                               multicolumn = True,
                               multicolumn_format = 'c',
                               caption = caption,
                               label = label)
  
    # To Latex
    latex_table = results.to_latex(**function_parameters)
       
    # Add string size
    location_size = latex_table.find('\centering\n')
    latex_table = latex_table[:location_size + len('\centering\n')] + size_string + latex_table[location_size + len('\centering\n'):]
    
    # Add note to the table
    if note_string is not None:
        location_note = latex_table.find('\end{tabular}\n')
        latex_table = latex_table[:location_note + len('\end{tabular}\n')]\
            + '\\begin{tablenotes}\n\\scriptsize\n\\item ' + note_string + '\\end{tablenotes}\n' + latex_table[location_note + len('\end{tabular}\n'):]
            
    # Add midrule above 'Observations'
    if latex_table.find('N                       &') >= 0:
        size_midrule = '\\midrule '
        location_mid = latex_table.find('N                       &')
        latex_table = latex_table[:location_mid] + size_midrule + latex_table[location_mid:]
           
    # Makes sidewaystable
    if sidewaystable:
        latex_table = latex_table.replace('{table}','{sidewaystable}',2)
        
    # Make threeparttable
    location_centering = latex_table.find('\centering\n')
    latex_table = latex_table[:location_centering + len('\centering\n')] + '\\begin{threeparttable}\n' + latex_table[location_centering + len('\centering\n'):]
    
    location_endtable = latex_table.find('\\end{tablenotes}\n')
    latex_table = latex_table[:location_endtable + len('\\end{tablenotes}\n')] + '\\end{threeparttable}\n' + latex_table[location_endtable + len('\\end{tablenotes}\n'):]
        
    return latex_table

#--------------------------------------------
# Make nice table
#--------------------------------------------

# Concat centroids and cluster, label and round
## Concat
df_clus = centroids.append(pd.DataFrame(cluster_n, index = ['C1','C2','C3'], columns = ['N']).T)

# Round and add commas
## Divide proxies by 1e3
df_clus.loc[vars_tot, :] = df_clus.loc[vars_tot, :].divide(1e3)
df_clus.iloc[:,0] = df_clus.iloc[:,0].round(4).apply(lambda x : "{:,}".format(x))
df_clus.iloc[:,1] = df_clus.iloc[:,1].round(4).apply(lambda x : "{:,}".format(x))
df_clus.iloc[:,2] = df_clus.iloc[:,2].round(4).apply(lambda x : "{:,}".format(x))

## Labels
labels = ['Sec. Income','CD Sold',\
          'CD Purchased',\
          'Assets Sold and Sec.','Asset Sold and Not Sec.',\
          'Cred. Exp. Oth.','TA Sec. Veh.','TA ABCP','TA Other VIEs',\
          'HDMA GSE','HMDA Private','HMDA Sec.','TA','T1 lev.',\
              'T1 cap. ', 'Cap. Ratio',\
          'Dep. ratio', 'Loan Ratio','RA ratio',\
          'CI ratio', 'Agri. ratio',\
          'Cons. ratio', 'Other loan ratio', 'loan HHI',\
          'ROA', 'Liq. Ratio', 'CTI',\
          'NII/NOR', 'RWA/TA', 'NPL Ratio',\
          'Charge-off', 'Allowance', 'Provision', 'N']
df_clus.index = labels

# Split table
df_clus_sec = df_clus.iloc[list(range(13)) + [-1],:] # Add N to the table
df_clus_oth = df_clus.iloc[13:,:]

# To latex
caption_sec = 'Cluster Centroids: Securitization Proxies'
label_sec = 'tab:cluster_centroids_sec'
size_string = '\\footnotesize \n'
note_sec = "\\textit{Notes.} The cluster centroids are calculated with a K-means cluster algorithm, $K = 3$. In million USD."

caption_control = 'Cluster Centroids: Control Variables'
label_control = 'tab:cluster_centroids_sec'
note_control = "\\textit{Notes.} The cluster centroids are calculated with a K-means cluster algorithm, $K = 3$."

df_clus_sec_latex = resultsToLatex(df_clus_sec, caption_sec, label_sec,\
                                 size_string = size_string, note_string = note_sec,\
                                 sidewaystable = False)

df_clus_oth_latex = resultsToLatex(df_clus_oth, caption_control, label_control,\
                                 size_string = size_string, note_string = note_control,\
                                 sidewaystable = False)
    
# Save
df_clus_sec.to_excel('Tables/Cluster_centroids_sec.xlsx')

text_ss_latex = open('Tables/Cluster_centroids_sec.tex', 'w')
text_ss_latex.write(df_clus_sec_latex)
text_ss_latex.close()

df_clus_oth.to_excel('Tables/Cluster_centroids_control.xlsx')

text_ss_latex = open('Tables/Cluster_centroids_control.tex', 'w')
text_ss_latex.write(df_clus_oth_latex)
text_ss_latex.close()


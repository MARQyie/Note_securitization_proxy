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
sns.set(style = 'whitegrid', font_scale = 2.5)

# Machine learning packages
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

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

#--------------------------------------------
# Determine number of clusters
## Dendrogram
'''Uses a agglomerative hierarchical method of clustering to display the 
    distance between each subsequent cluster '''
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    fig, ax = plt.subplots(figsize=(15,9)) 
    dendrogram(linkage_matrix, ax = ax, **kwargs)
    
# Setup the model
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

# Fit model
model = model.fit(df[vars_tot + vars_oth])
plt.title('Hierarchical Clustering Dendrogram')

# Plot
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.ylabel('Distance')
plt.savefig('Figures/Cluster_analysis/Dendrogram_hierarch_clustering.png')
plt.close()

## Elbow Plot
'''Uses the Within-Cluster Sum-of-Squares and selects the model with the lowest
    sum of squares '''
### Initialize model and load package
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()

### Make the Elbow plot
visualizer = KElbowVisualizer(model, k=(2,15), timings= True, size = (1500,900))
visualizer.fit(df[vars_tot + vars_oth])        
visualizer.show('Figures/Cluster_analysis/WCSS.png')   
plt.clf()

'''OLD
fig, ax = plt.subplots(figsize=(20,12)) 
ax.set(ylabel='Within-Cluster Sum-of-Squares (1e20)', xlabel = 'Number of Clusters')
ax.plot(np.arange(2,12,1),intertia / 1e20)
plt.xticks(np.arange(2,12,1))
plt.tight_layout()
fig.savefig('Figures/Cluster_analysis/WCSS.png') '''

'''NOTE There is a clear elbow at three clusters. Proceed.
    '''
## Silhouette plot
### Initialize model
model = KMeans()

### Make the Silhouette plot
visualizer = KElbowVisualizer(model, k=(2,15),metric='silhouette', timings= True, size = (1500,900))
visualizer.fit(df[vars_tot + vars_oth])        
visualizer.show(outpath = 'Figures/Cluster_analysis/Silhouette.png')
plt.clf()
'''NOTE: elbow at 3 '''

## Calinski-Harabasz Index
'''The Calinski-Harabasz Index is based on the idea that clusters that are 
        (1) themselves very compact and 
        (2) well-spaced from each other are good clusters'''
### Initialize model
model = KMeans()

## Make Calinski-Harabasz plot
visualizer = KElbowVisualizer(model, k=(2,15),metric='calinski_harabasz', timings= True, size = (1500,900))
visualizer.fit(df[vars_tot + vars_oth])        
visualizer.show(outpath = 'Figures/Cluster_analysis/Calinski_harabasz.png')
plt.clf()

## Davies-Bouldin Index
'''DB index captures both the separation and compactness of the clusters.'''
from sklearn.metrics import davies_bouldin_score

### make method
def get_kmeans_score(data, center):
    '''
    returns the kmeans score regarding Davies Bouldin for points to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    OUTPUT:
        score - the Davies Bouldin score for the kmeans model fit to the data
    '''
    #instantiate kmeans
    kmeans = KMeans(n_clusters=center)# Then fit the model to your data using the fit method
    model = kmeans.fit_predict(df[vars_tot + vars_oth])
    
    # Calculate Davies Bouldin score
    score = davies_bouldin_score(df[vars_tot + vars_oth], model)
    
    return score

### Get scores
scores = []
centers = list(range(2,15))

for center in centers:
    scores.append(get_kmeans_score(df[vars_tot + vars_oth], center))
    
min_db = scores.index(min(scores))

### Plot
fig, ax = plt.subplots(figsize=(15,9)) 
ax.plot(centers, scores, linestyle='--', marker='o', color='b')
ax.axvline(min_db + 2, color = 'black', linestyle = '--')
ax.set(ylabel='Davies Bouldin score', xlabel = 'K')
fig.savefig('Figures/Cluster_analysis/Davies_Bouldin.png')

#--------------------------------------------
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
label_control = 'tab:cluster_centroids_oth'
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
#--------------------------------------------
# Data Setup Securitization Note: Non-securitization data only
# Mark van der Plaat
# September 2020
#--------------------------------------------
#--------------------------------------------
# Import Packages
#--------------------------------------------

# Data manipulation
import pandas as pd
import numpy as np

# Parallelization
import multiprocessing as mp
from joblib import Parallel, delayed

# Finding duplicates
import collections

# Set WD
import os
os.chdir(r'D:\RUG\PhD\Materials_papers\01-Note_on_securitization')

#--------------------------------------------
# Prelims
#--------------------------------------------

# Set start and end date
start = 2011
end = 2020

# Get the number of cores
num_cores = mp.cpu_count()


#--------------------------------------------
# Call Reports
#--------------------------------------------

''' We loop over all years and all necessary reporting schedules to 
    make a dataset for the Call Reports. We merge RCFD and RCON where possible.
    We do not remove any banks or outliers, this has been done in the
    securitization dataset.
'''

#--------------------------------------------
# Setup

# Get file paths
path_call = r'D:/RUG/Data/Data_call_reports_FFIEC2'

# Get filenames per schedule
## RI
file_ri = r'/{}/FFIEC CDR Call Schedule RI 1231{}.txt'

## RC
file_rc = r'/{}/FFIEC CDR Call Schedule RC 1231{}.txt'

# Get variable names per schedule
## RI
vars_ri = '|'.join(['IDRSSD','4301'])

## RC-D
vars_rc = '|'.join(['IDRSSD','2170'])

#--------------------------------------------
# Set functions
# Functions for loading data

def loadGeneral(i, file, var_list):
    ''' A General function for loading call reports data, no breaks in file
        names'''
    
    df_load = pd.read_csv((path_call + file).format(i,i), sep='\t',  skiprows = [1,2])
    df_load['date'] = int('{}'.format(i))  
    
    return(df_load.loc[:,df_load.columns.str.contains(var_list + '|date')])

# Functions to combine variables
def combineVars(data, elem):
    ''' Function to combine RCFD and RCON into one variable '''
    data['RC' + elem] = data.apply(lambda x: x['RCFD' + elem] if not np.isnan(x['RCFD' + elem]) and  round(x['RCFD' + elem]) != 0 else (x['RCON' + elem]), axis = 1) 
    
    return(data['RC' + elem])

#--------------------------------------------
# Load Data

# Run functions
if __name__ == '__main__':
    df_ri = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_ri, vars_ri) for i in range(start, end)))
    df_rc = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_rc, vars_rc) for i in range(start, end)))

# Concat all dfs
df_raw = df_ri.set_index(['IDRSSD','date']).join([df_rc.set_index(['IDRSSD','date'])], how = 'inner')

#--------------------------------------------
# Transform variables

# Get double variable
vars_raw = df_raw.columns.str[4:]
var_num = [item for item, count in collections.Counter(vars_raw).items() if count > 1]
    
# Combine variables
if __name__ == '__main__':
    df_raw_combvars = pd.concat(Parallel(n_jobs=num_cores)(delayed(combineVars)(df_raw, elem) for elem in var_num), axis = 1)

# Remove old variables
cols_remove =  [col for col in df_raw.columns if not col[4:] in var_num]      
df = pd.concat([df_raw[cols_remove], df_raw_combvars], axis = 1) 

#--------------------------------------------
# Save Data
#--------------------------------------------

df.to_csv('Data\df_ri_rc_note.csv')
 
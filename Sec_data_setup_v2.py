#--------------------------------------------
# Data Setup Securitization Note
# Mark van der Plaat
# September 2020
#--------------------------------------------

''' This script loads data from the Call Reports, Statistcs on
    Depository Institutions (SDI), and the Home Mortgage Disclosure
    Act (HMDA), and creates a dataset containing possible proxies for 
    securitization. We use the Call Reports as main dataset. We merge 
    HMDA data with the SDI data, and merge this entire dataset to the 
    Call Reports. This dataset only includes proxies for securitization 
    and variables used for cleaning the data.
    '''

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
    make a dataset for the Call Reports. We merge RCFD and RCON where possible,
    and we remove banks that are not insured, not commercial and not in one
    of the 50 states.
    
    For all schedules we use a trickto get all RCFD and RCON by looking for
    the four-digit variable name. We then merge those together after loading.
    
    UPDATE 2020-10-14: Securitization variables from the SDI are almost perfectly
    colinear with the Call Reports. We therefore remove the SDI data from the 
    dataset. See version 1 of this script to get a dataset that includes the SDI.
'''

#--------------------------------------------
# Setup

# Get file paths
path_info = r'D:/RUG/Data/Data_call_reports_fed'
path_call = r'D:/RUG/Data/Data_call_reports_FFIEC2'

# Get filenames per schedule
## Info
file_info = r'/call{}12.xpt'

## RI
file_ri = r'/{}/FFIEC CDR Call Schedule RI 1231{}.txt'

## RC-D
file_rcd = r'/{}/FFIEC CDR Call Schedule RCD 1231{}.txt'

## RC-L
file_rcl_1 = r'/{}/FFIEC CDR Call Schedule RCL 1231{}(1 of 2).txt'
file_rcl_2 = r'/{}/FFIEC CDR Call Schedule RCL 1231{}(2 of 2).txt'

## RC-S
file_rcs = r'/{}/FFIEC CDR Call Schedule RCS 1231{}.txt'

## SU
file_su = r'/{}/FFIEC CDR Call Schedule SU 1231{}.txt' #from 2017 only

## RC-V
file_rcv = r'/{}/FFIEC CDR Call Schedule RCV 1231{}.txt'

# Get variable names per schedule
## Info
vars_info = ['IDRSSD', 'RSSD9048', 'RSSD9424', 'RSSD9210']

## RI
vars_ri = '|'.join(['IDRSSD','B493'])

## RC-D
vars_rcd = '|'.join(['IDRSSD','F654'])

## RC-L
vars_rcl = '|'.join(['IDRSSD', 'C968', 'C970', 'C972', 'C974',\
                     'C969', 'C971', 'C973', 'C975'])

## RC-S
vars_rcs = '|'.join(['IDRSSD', 'B705', 'B706', 'B707', 'B708',\
                     'B709', 'B710', 'B711', 'B790', 'B791',\
                     'B792', 'B793', 'B794', 'B795', 'B796',\
                     'B776', 'B777', 'B778', 'B779', 'B780',\
                     'B781', 'B782',])

## SU
vars_su = '|'.join(['IDRSSD', 'FT08', 'FT10', 'FT14'])

## RC-V
vars_rcv = '|'.join(['IDRSSD'] + ['J{}'.format(i) for i in range(981, 999 + 1)] +\
                    ['K{}'.format(str(i).zfill(3)) for i in range(1, 14 + 1)] +\
                    ['K030', 'K031', 'K032'] +\
                    ['HU{}'.format(i) for i in range(20, 23 + 1)] +\
                    ['JF84', 'JF87', 'JF89', 'JF91', 'JF90', 'JF77'])

#--------------------------------------------
# Set functions
# Functions for loading data
def loadInfo(i):
    ''' Function to load the info data'''
    
    df_load = pd.read_sas((path_info + file_info).format(i))
    df_load.rename(columns = {'RSSD9001':'IDRSSD'}, inplace = True)
    df_load['date'] = int('20{}'.format(i))
        
    return(df_load[vars_info + ['date']])


def loadGeneral(i, file, var_list):
    ''' A General function for loading call reports data, no breaks in file
        names'''
    
    df_load = pd.read_csv((path_call + file).format(i,i), sep='\t',  skiprows = [1,2])
    df_load['date'] = int('{}'.format(i))  
    
    return(df_load.loc[:,df_load.columns.str.contains(var_list + '|date')])
    
def loadRCL(i):
    ''' Dedicated function for loading RC-L data '''
    global path_call

    df_load_1 = pd.read_csv((path_call + file_rcl_1).format(i,i), sep='\t',  skiprows = [1,2])
    df_load_2 = pd.read_csv((path_call + file_rcl_2).format(i,i), sep='\t',  skiprows = [1,2])
    df_load = df_load_1.merge(df_load_2, on = 'IDRSSD', how = 'left')

    df_load['date'] = int('{}'.format(i))  
    
    return(df_load.loc[:,df_load.columns.str.contains(vars_rcl + '|date')])

# Functions to combine variables
def combineVars(data, elem):
    ''' Function to combine RCFD and RCON into one variable '''
    data['RC' + elem] = data.apply(lambda x: x['RCFD' + elem] if not np.isnan(x['RCFD' + elem]) and  round(x['RCFD' + elem]) != 0 else (x['RCON' + elem]), axis = 1) 
    
    return(data['RC' + elem])

#--------------------------------------------
# Load Data

# Run functions
if __name__ == '__main__':
    df_info = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadInfo)(i) for i in range(start - 2000, end - 2000)))
    df_ri = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_ri, vars_ri) for i in range(start, end)))
    df_rcd = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_rcd, vars_rcd) for i in range(start, end)))
    df_rcl = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadRCL)(i) for i in range(start, end)))
    df_rcs = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_rcs, vars_rcs) for i in range(start, end)))
    df_su = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_su, vars_su) for i in range(2017, end)))
    df_rcv = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_rcv, vars_rcv) for i in range(start, end)))
    
# Concat all dfs
df_cr_raw = df_info.set_index(['IDRSSD','date']).join([df_ri.set_index(['IDRSSD','date']),\
                           df_rcd.set_index(['IDRSSD','date']),\
                           df_rcl.set_index(['IDRSSD','date']),\
                           df_rcs.set_index(['IDRSSD','date']),\
                           df_rcv.set_index(['IDRSSD','date'])], how = 'inner')
df_cr_raw = df_cr_raw.merge(df_su.set_index(['IDRSSD','date']), how = 'left', left_index = True, right_index = True)
    
#--------------------------------------------
# Transform variables

# Get double variable
vars_cr_raw = df_cr_raw.columns.str[4:]
var_num = [item for item, count in collections.Counter(vars_cr_raw).items() if count > 1]
    
# Combine variables
if __name__ == '__main__':
    df_cr_raw_combvars = pd.concat(Parallel(n_jobs=num_cores)(delayed(combineVars)(df_cr_raw, elem) for elem in var_num), axis = 1)

# Remove old variables
cols_remove =  [col for col in df_cr_raw.columns if not col[4:] in var_num]      
df_cr_raw = pd.concat([df_cr_raw[cols_remove], df_cr_raw_combvars], axis = 1)  
    
#--------------------------------------------
# Make clean dataset

## Select insured and commercial banks
'''RSSD9048 == 200, RSSD9424 != 0'''
df_cr = df_cr_raw[(df_cr_raw.RSSD9048 == 200) & (df_cr_raw.RSSD9424 != 0)]

## Only take US states, no territories
''' Based on the state codes provided by the FRB'''
df_cr = df_cr[df_cr.RSSD9210.isin(range(1,57))]
    

#--------------------------------------------
# Home Mortgage Disclosure Act
#--------------------------------------------

''' Again similar setup. Because HMDA contains large files, we first
    clean the data and than aggregate to cert. We also load the lender
    file to be able to merge with SDI '''

#--------------------------------------------
# Setup

# Get Paths
path_lf = r'D:/RUG/Data/Data_HMDA_lenderfile/'
path_hmda = r'D:/RUG/Data/Data_HMDA/LAR/'
path_hmda_panel = r'D:/RUG/Data/Data_HMDA/Panel/'

# Get file names
file_lf = r'hmdpanel17.dta'
file_hmda_1017 = r'hmda_{}_nationwide_originated-records_codes.zip'
file_hmda_1819 = r'year_{}.csv'
file_hmda_panel = r'{}_public_panel_csv.csv'

## Set d-types and na-vals for HMDA LAR
dtypes_col_hmda = {'state_code':'str', 'county_code':'str'}
na_values = ['NA   ', 'NA ', '...',' ','NA  ','NA      ','NA     ','NA    ','NA', 'Exempt', 'N/AN/', 'na']

# Get variable names
vars_hmda = ['respondent_id', 'hmda_gse_amount', 'hmda_priv_amount',\
             'hmda_sec_amount', 'loan_amount_000s']
vars_hmda_panel = ['lei', 'arid_2017', 'tax_id']
vars_lf = ['hmprid'] + ['ENTITY{}'.format(str(year)[2:4]) for year in range(start, end - 2)]

#--------------------------------------------
# Set functions

def loadCleanHMDA(year):
    #Load the dataframe in a temporary frame
    if year < 2018:
        df_chunk = pd.read_csv(path_hmda + file_hmda_1017.format(year),\
                               index_col = 0, chunksize = 1e6, na_values = na_values,\
                               dtype = dtypes_col_hmda)
    else: # From 2018 onward structure of the data changes
        df_chunk = pd.read_csv(path_hmda + file_hmda_1819.format(year),\
                               index_col = 0, chunksize = 1e6, na_values = na_values,\
                               dtype = dtypes_col_hmda)
        df_panel = pd.read_csv(path_hmda_panel + file_hmda_panel.format(year))
        
    chunk_list = []  # append each chunk df here 

    ### Loop over the chunks
    for chunk in df_chunk:
        # Merge with df_panel and change column names if year is 2018 or 2019
        if year >= 2018:
            ## Merge
            chunk = chunk.merge(df_panel.loc[:,vars_hmda_panel], how = 'left', on = 'lei')
                    
            ## Add column of loan amount in thousand USD
            chunk['loan_amount_000s'] = chunk.loan_amount / 1e3
            
            ## Format respondent_id column  (form arid_2017: remove agency code (first string), and zero fill. Replace -1 with non)
            ## NOTE Fill the missings with the tax codes. After visual check, some arid_2017 are missing. However, the tax_id corresponds to the resp_id pre 2018
            chunk['respondent_id'] = chunk.apply(lambda x: x.tax_id if x.arid_2017 == '-1' else str(x.arid_2017)[1:], axis = 1).replace('-1',np.nan).str.zfill(10)
            
        # Filter data
        ## Make a fips column and remove separate state and county
        if year < 2018:
            chunk['fips'] = chunk['state_code'].str.zfill(2) + chunk['county_code'].str.zfill(3)
        else:
            chunk['fips'] =  chunk['county_code'].str.zfill(5)
    
        ## Add variables
        ''' List of new vars
            Dummies loan sales, securitizatioin
            Amounts loan sales, securitization
        '''
        
        ### Loan sale dummies
        chunk['hmda_gse_dum'] = (chunk.purchaser_type.isin(range(1,4+1))) * 1
        chunk['hmda_priv_dum'] = (chunk.purchaser_type.isin(list(range(1,9+1)) + [71, 72])) * 1
        chunk['hmda_sec_dum'] = (chunk.purchaser_type == 5) * 1
        
        ### Loan sale amount
        chunk['hmda_gse_amount'] = chunk.hmda_gse_dum * chunk.loan_amount_000s
        chunk['hmda_priv_amount'] = chunk.hmda_priv_dum * chunk.loan_amount_000s
        chunk['hmda_sec_amount'] = chunk.hmda_sec_dum * chunk.loan_amount_000s
        
        ## Drop na in fips
        chunk.dropna(subset = ['fips'], inplace = True)
        
        ## Subset the df
        chunk_subset = chunk[vars_hmda]

        # Add the chunk to the list
        chunk_list.append(chunk_subset)
        
    # concat the list into dataframe
    chunk_concat = pd.concat(chunk_list)

    ## Aggregate data
    chunk_agg = chunk_concat.groupby('respondent_id').sum()
    
    ## Add a date var
    chunk_agg['date'] = year
    
    return chunk_agg 

#--------------------------------------------
# Get data
    
# Run function
if __name__ == '__main__':    
    df_hmda = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadCleanHMDA)(i) for i in range(start, end)))

# df LF
df_lf = pd.read_stata(path_lf + file_lf, columns = vars_lf)

# Reduce dimensions df_lf
df_lf.dropna(how = 'all', subset = vars_lf[1:], inplace = True) # drop rows with all na
df_lf = df_lf[~(df_lf[vars_lf[1:]] == 0.).any(axis = 1)] # Remove ENTITY with value 0.0
df_lf = df_lf[df_lf[vars_lf[1:]].all(axis = 1)] # Drop rows that have different column values (nothing gets deleted: good)

#--------------------------------------------
# Merge Data
#--------------------------------------------

''' Merging procedure is as follows:
        1) Merge SDI dataframe with the lenderfile on FDIC certificate number --> df_sdilf
            Since the LF has a wide structure, we need to loop over the years
        2) Merge the HMDA data with df_dfilf on FDIC certificate number --> df_hmdasdi
        3) Merge df_hmdasdi with the Call Reports data on ID RSSD --> df_raw
    '''

#--------------------------------------------
# Set functions
    
def mergeCallReportsLF(year):
    '''Function to merge SDI and LF'''
    ## Prelims
    if year < 2018:
        entity = 'ENTITY{}'.format(str(year)[2:4])
    else:
        entity = 'ENTITY17'
       
    ## Merge on year
    df_load = df_cr.reset_index()[df_cr.reset_index().date == year].merge(df_lf.dropna(subset = [entity]),\
                            how = 'left', left_on = 'IDRSSD',\
                            right_on = entity)
    
    ## Return concatenated pd DataFrame
    return (df_load)

#--------------------------------------------
# Merge data
    
# 1) CR LF
if __name__ == "__main__":
    df_crlf = pd.concat(Parallel(n_jobs=num_cores)(delayed(mergeCallReportsLF)(year) for year in range(start,end)))
    
## Drop ENTITY.. columns and drop nans in hmprid
df_crlf = df_crlf[df_crlf.columns[~df_crlf.columns.str.contains('ENTITY')]].dropna(subset = ['hmprid'])

# 2) CR HMDA
df_raw = df_crlf.reset_index().merge(df_hmda, left_on = ['date','hmprid'],\
                            right_on = ['date','respondent_id'], how = 'left')

## Drop na in hmprid
df_raw.dropna(subset = ['hmprid'], inplace = True)

#--------------------------------------------
# make new df
#--------------------------------------------
''' With the complete dataframe we make a new variables and df'''

# New df
df = df_raw[['date','IDRSSD', 'hmda_gse_amount', 'hmda_priv_amount',\
             'hmda_sec_amount']]

#--------------------------------------------
# Add new variables

# Net securitization income
df['cr_sec_income'] = df_raw.RIADB493

''' Only for banks with >1billion of trading assets. also includes ABSs etc. 
# Loans pending securitization
df['cr_sec_pending'] = df_raw.RCF654
'''

# Sold protection credit derivatives
df['cr_cd_sold'] = df_raw.loc[:,['RCC968','RCC970','RCC972','RCC974']].sum(axis = 1)

# Purchased protection credit derivatives
df['cr_cd_purchased'] = df_raw.loc[:,['RCC969','RCC971','RCC973','RCC975']].sum(axis = 1)

'''OLD
# On-balance sheet securitization exposures
df['cr_se_on'] = df_raw.loc[:,['RCFDS475','RCFDS480','RCFDS485','RCFDS490']].sum(axis = 1)

# Off-balance sheet securitization exposures 
df['cr_se_off'] = df_raw.RCFDS495
'''
# Assets sold and securitized with recourse
df['cr_as_sec'] = df_raw.loc[:,['RCB{}'.format(i) for i in range(705,711+1)] +\
  ['RCONFT08']].sum(axis = 1)

# Assets sold and not securitized with recourse
df['cr_as_nonsec'] = df_raw.loc[:,['RCB{}'.format(i) for i in range(790,796+1)] +\
  ['RCONFT10']].sum(axis = 1)

# Maximum amount of credit exposure provided to other institutions
df['cr_ce_sec'] = df_raw.loc[:,['RCB{}'.format(i) for i in range(776,782+1)] +\
  ['RCONFT14']].sum(axis = 1)

# Total Assets Securitization Vehicles
vars_secveh = ['RCJ{}'.format(i) for i in range(981,999+1,3)] +\
              ['RCK{}'.format(str(i).zfill(3)) for i in range(3,12+1,3)] + \
              ['RCK030','RCHU20','RCHU22','RCJF91']
df['cr_ta_secveh'] = df_raw.loc[:,vars_secveh].sum(axis = 1)

# Total Assets ABCP Conduits
vars_abcp = ['RCJ{}'.format(i) for i in range(982,997+1,3)] +\
              ['RCK{}'.format(str(i).zfill(3)) for i in range(1,13+1,3)] + \
              ['RCK032','RCJF77']
df['cr_ta_abcp'] = df_raw.loc[:,vars_abcp].sum(axis = 1)

# Total Assets Other VIE
vars_vie_other = ['RCJ{}'.format(i) for i in range(983,988+1,3)] +\
              ['RCK{}'.format(str(i).zfill(3)) for i in range(2,14+1,3)] + \
              ['RCK031','RCHU21','RCHU23','RCJF89','RCJF90','RCJF87']
df['cr_ta_vie_other'] = df_raw.loc[:,vars_vie_other].sum(axis = 1)

#--------------------------------------------
# Fill NA
df.fillna(0, inplace = True)

#--------------------------------------------
# Save Data
#--------------------------------------------

df.to_csv('Data\df_sec_note.csv')

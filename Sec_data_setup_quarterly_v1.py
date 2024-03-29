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

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid', font_scale = 2)

# Set WD
import os
os.chdir(r'D:\RUG\PhD\Materials_papers\01-Note_on_securitization')

#--------------------------------------------
# Prelims
#--------------------------------------------

# Set start and end date
start = 2011
end = 2018

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
file_info = r'/call{}.xpt'

## RI
file_ri = r'/{}/FFIEC CDR Call Schedule RI {}.txt'

## RC
file_rc = r'/{}/FFIEC CDR Call Schedule RC {}.txt'

## RC-D
file_rcd = r'/{}/FFIEC CDR Call Schedule RCD {}.txt'

## RC-L
file_rcl_1 = r'/{}/FFIEC CDR Call Schedule RCL {}(1 of 2).txt'
file_rcl_2 = r'/{}/FFIEC CDR Call Schedule RCL {}(2 of 2).txt'

## RC-S
file_rcs = r'/{}/FFIEC CDR Call Schedule RCS {}.txt'

## RC-V
file_rcv = r'/{}/FFIEC CDR Call Schedule RCV {}.txt'

# Get variable names per schedule
## Info
vars_info = ['IDRSSD', 'RSSD9048', 'RSSD9424', 'RSSD9210']

## RI
vars_ri = '|'.join(['IDRSSD','B492','B493','5416'])

## RC
vars_rc = '|'.join(['IDRSSD','2170'])

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
                     'B781', 'B782','B806','B807','B808','B809',\
                     'A249'])

## RC-V
vars_rcv = '|'.join(['IDRSSD'] + ['J{}'.format(i) for i in range(981, 999 + 1)] +\
                    ['K{}'.format(str(i).zfill(3)) for i in range(1, 35 + 1)] +\
                    ['K030', 'K031', 'K032'])

#--------------------------------------------
# Set functions
# Functions for loading data
def loadInfo(i, j):
    ''' Function to load the info data'''
    
    df_load = pd.read_sas((path_info + file_info).format(i + j))
    df_load.rename(columns = {'RSSD9001':'IDRSSD'}, inplace = True)
    df_load['date'] = int('{}20{}'.format(j,i))
        
    return(df_load[vars_info + ['date']])


def loadGeneral(i, j, file, var_list):
    ''' A General function for loading call reports data, no breaks in file
        names'''
    
    df_load = pd.read_csv((path_call + file).format(i,j + i), sep='\t',  skiprows = [1,2])
    df_load['date'] = int('{}'.format(i + j[:2]))  
    
    return(df_load.loc[:,df_load.columns.str.contains(var_list + '|date')])
    
def loadRCL(i, j):
    ''' Dedicated function for loading RC-L data '''
    global path_call

    df_load_1 = pd.read_csv((path_call + file_rcl_1).format(i,j + i), sep='\t',  skiprows = [1,2])
    df_load_2 = pd.read_csv((path_call + file_rcl_2).format(i,j + i), sep='\t',  skiprows = [1,2])
    df_load = df_load_1.merge(df_load_2, on = 'IDRSSD', how = 'left')

    df_load['date'] = int('{}'.format(i + j[:2]))  
    
    return(df_load.loc[:,df_load.columns.str.contains(vars_rcl + '|date')])

# Functions to combine variables
def combineVars(data, elem):
    ''' Function to combine RCFD and RCON into one variable '''
    data['RC' + elem] = data['RCFD' + elem].fillna(data['RCON' + elem])
    
    return(data['RC' + elem])

#--------------------------------------------
# Load Data

# Set counters for df_info and other dfs
q_info = ['03','06','09','12']
q_rest = ['0331','0630','0930','1231']

# Run functions
if __name__ == '__main__':
    df_info = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadInfo)(str(i), j) for j in q_info for i in range(start - 2000, end - 2000)))
    df_ri = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(str(i), j, file_ri, vars_ri) for j in q_rest for i in range(start, end)))
    df_rc = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(str(i), j, file_rc, vars_rc) for j in q_rest for i in range(start, end)))
    df_rcd = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(str(i), j, file_rcd, vars_rcd) for j in q_rest for i in range(start, end)))
    df_rcl = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadRCL)(str(i), j) for j in q_rest for i in range(start, end)))
    df_rcs = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(str(i), j , file_rcs, vars_rcs) for j in q_rest for i in range(start, end)))
    df_rcv = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(str(i), j, file_rcv, vars_rcv) for j in q_rest for i in range(start, end)))
    
# Concat all dfs
df_cr_raw = df_info.set_index(['IDRSSD','date']).join([df_ri.set_index(['IDRSSD','date']),\
                           df_rc.set_index(['IDRSSD','date']),\
                           df_rcd.set_index(['IDRSSD','date']),\
                           df_rcl.set_index(['IDRSSD','date']),\
                           df_rcs.set_index(['IDRSSD','date']),\
                           df_rcv.set_index(['IDRSSD','date'])], how = 'inner')
    
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

## Remove all banks with less than 1 billion in total assets
df_cr = df_cr[df_cr.RC2170 >= 1e6]

## Reset index
df_cr.reset_index(inplace = True)

#--------------------------------------------
# make new df
#--------------------------------------------
''' With the complete dataframe we make a new variables and df'''

# New df
df = df_cr[['date','IDRSSD']]

#--------------------------------------------
# Add new variables
'''
# Net servicing fees
df['cr_serv_fees'] = df_cr.RIADB492

# Net securitization income
df['cr_sec_income'] = df_cr.RIADB493 '''

''' Only for banks with >1billion of trading assets. also includes ABSs etc. 
# Loans pending securitization
df['cr_sec_pending'] = df_raw.RCF654
'''

# Net gains loan sales
#df['cr_ls_income'] = df_cr.RIAD5416

# Sold protection credit derivatives plus sub-categories
'''
df['cr_cd_sold'] = df_cr.loc[:,['RCC968','RCC970','RCC972','RCC974']].sum(axis = 1)
df['cr_cds_sold'] = df_cr.RCC968 # Usual CDO structure
df['cr_trs_sold'] = df_cr.RCC970 # Other possible option, see Lancaster et al (2008): Structured Products and Related Credit Derivatives: A Comprehensive Guide for Investors 
df['cr_co_sold'] = df_cr.RCC972
df['cr_cdoth_sold'] = df_cr.RCC974 '''

# Purchased protection credit derivativesplus sub-categories
# NOTE If the bank does CDO securitization, it is most likely to purchase CDSs and TRSs. Other forms do exist
df['cr_cd_purchased'] = df_cr.loc[:,['RCC969','RCC971','RCC973','RCC975']].sum(axis = 1)
df['cr_cds_purchased'] = df_cr.RCC969
df['cr_trs_purchased'] = df_cr.RCC971
df['cr_co_purchased'] = df_cr.RCC973
df['cr_cdoth_purchased'] = df_cr.RCC975

'''OLD
# On-balance sheet securitization exposures
df['cr_se_on'] = df_raw.loc[:,['RCFDS475','RCFDS480','RCFDS485','RCFDS490']].sum(axis = 1)

# Off-balance sheet securitization exposures 
df['cr_se_off'] = df_raw.RCFDS495
'''
# Assets sold and securitized with recourse 
df['cr_as_sec'] = df_cr.loc[:,['RCB{}'.format(i) for i in range(705,711+1)]].sum(axis = 1)
df['cr_as_rmbs'] = df_cr.RCB705
df['cr_as_abs'] = df_cr.loc[:,['RCB{}'.format(i) for i in range(706,711+1)]].sum(axis = 1)

# Assets sold and not securitized with recourse
df['cr_as_nonsec'] = df_cr.loc[:,['RCB{}'.format(i) for i in range(790,796+1)]].sum(axis = 1)

# Maximum amount of credit exposure provided to other institutions
df['cr_ce_sec'] = df_cr.loc[:,['RCB{}'.format(i) for i in range(776,782+1)]].sum(axis = 1)
df['cr_ce_rmbs'] = df_cr.RCB776
df['cr_ce_abs'] = df_cr.loc[:,['RCB{}'.format(i) for i in range(777,782+1)]].sum(axis = 1)

# Outstanding principle balance small business obligations transferred with recourse
df['cr_as_sbo'] = df_cr.RCA249

# Maximum credit exposure to ABCP conduits (sponsered by the institution itself or by others)
df['cr_abcp_ce'] = df_cr.loc[:,['RCB806','RCB807']].sum(axis = 1)

# Unused commitments to provide liquidity to ABCP conduits (sponsered by the institution itself or by others)
df['cr_abcp_uc'] = df_cr.loc[:,['RCB808','RCB809']].sum(axis = 1)

# Total Assets Securitization Vehicles
# NOTE: We only use the total assets of the securitization vehicles, because there is no
# straight-forward variable measuring outstanding ABSs/CDOs. These vehicles can be both
# ABS SPVs or CDO SPVs (or anything similar but not ABCP conduits/SIV)
vars_secveh = ['RCJ{}'.format(i) for i in range(981,999+1,3)] +\
              ['RCK{}'.format(str(i).zfill(3)) for i in range(3,12+1,3)] + \
              ['RCK030']
df['cr_secveh_ta'] = df_cr.loc[:,vars_secveh].sum(axis = 1)

# Total Assets ABCP Conduits
# NOTE: The real exposure to ABCP comes from the conduits liabilities, especially
# its commercial papers and repos. We included total assets in case these two 
# variables do not have enough variation 
vars_abcp = ['RCJ{}'.format(i) for i in range(982,997+1,3)] +\
              ['RCK{}'.format(str(i).zfill(3)) for i in range(1,13+1,3)] + \
              ['RCK032']
df['cr_abcp_ta'] = df_cr.loc[:,vars_abcp].sum(axis = 1)
df['cr_abcp_cp'] = df_cr.RCK022
df['cr_abcp_repo'] = df_cr.RCK016 # ALL ZERO

# Total Assets Other VIE
vars_vie_other = ['RCJ{}'.format(i) for i in range(983,988+1,3)] +\
              ['RCK{}'.format(str(i).zfill(3)) for i in range(2,14+1,3)] + \
              ['RCK031']
df['cr_ta_vie_other'] = df_cr.loc[:,vars_vie_other].sum(axis = 1)

# Total assets
df['ta'] = df_cr.RC2170

#--------------------------------------------
# Fill NA
df.fillna(0, inplace = True)

# Remove negative values in cr_ta_secveh
df = df[df.cr_secveh_ta >= 0]

# NOTE: All bank-years have at least one non-zero value 

'''
# Remove outliers
vars_tot = df.columns[2:]
lst_outliers = []

## Make box plots
def boxPlots(var):
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.title('{}'.format(var))
    
    data = df[var].dropna()
    ax.boxplot(data)
    
    plt.xticks([1], ['Securitizers Only'])
    
    fig.savefig('Figures/Box_plots/Box_{}.png'.format(var)) 
    plt.clf()

for var in vars_tot:
    boxPlots(var)
'''
## Drop cr_abcp_repo
df.drop(columns = 'cr_abcp_repo', inplace = True)
    
## Remove outliers
#df = df.loc[df.cr_serv_fees != df.cr_serv_fees.min(),:]

#--------------------------------------------
# Save Data
#--------------------------------------------

df.to_csv('Data\df_sec_note_quarterly.csv')

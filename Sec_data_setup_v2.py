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
start = 2007
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
file_info = r'/call{}12.xpt'

## RI
file_ri = r'/{}/FFIEC CDR Call Schedule RI 1231{}.txt'

## RC
file_rc = r'/{}/FFIEC CDR Call Schedule RC 1231{}.txt'

## RC-D
file_rcd = r'/{}/FFIEC CDR Call Schedule RCD 1231{}.txt'

## RC-L
file_rcl1 = r'/{}/FFIEC CDR Call Schedule RCL 1231{}.txt'
file_rcl2_1 = r'/{}/FFIEC CDR Call Schedule RCL 1231{}(1 of 2).txt'
file_rcl2_2 = r'/{}/FFIEC CDR Call Schedule RCL 1231{}(2 of 2).txt'

## RC-S
file_rcs = r'/{}/FFIEC CDR Call Schedule RCS 1231{}.txt'

## RC-V
file_rcv = r'/{}/FFIEC CDR Call Schedule RCV 1231{}.txt'

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
                     'A249','B783','B784','B785','B786','B787','B788','B789'])

## RC-V
vars_rcv = '|'.join(['IDRSSD'] + ['J{}'.format(i) for i in range(981, 999 + 1)] +\
                    ['K{}'.format(str(i).zfill(3)) for i in range(1, 35 + 1)] +\
                    ['K030', 'K031', 'K032'])

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

    if i < 2009:
        df_load = pd.read_csv((path_call + file_rcl1).format(i,i), sep='\t',  skiprows = [1,2])
    else:
        df_load_1 = pd.read_csv((path_call + file_rcl2_1).format(i,i), sep='\t',  skiprows = [1,2])
        df_load_2 = pd.read_csv((path_call + file_rcl2_2).format(i,i), sep='\t',  skiprows = [1,2])
        df_load = df_load_1.merge(df_load_2, on = 'IDRSSD', how = 'left')

    df_load['date'] = int('{}'.format(i))  
    
    return(df_load.loc[:,df_load.columns.str.contains(vars_rcl + '|date')])

# Functions to combine variables
def combineVars(data, elem):
    ''' Function to combine RCFD and RCON into one variable '''
    data['RC' + elem] = data['RCFD' + elem].fillna(data['RCON' + elem])
    
    return(data['RC' + elem])

#--------------------------------------------
# Load Data

# Run functions
if __name__ == '__main__':
    df_info = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadInfo)(str(i).zfill(2)) for i in range(start - 2000, end - 2000)))
    df_ri = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_ri, vars_ri) for i in range(start, end)))
    df_rc = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_rc, vars_rc) for i in range(start, end)))
    df_rcd = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_rcd, vars_rcd) for i in range(start, end)))
    df_rcl = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadRCL)(i) for i in range(start, end)))
    df_rcs = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_rcs, vars_rcs) for i in range(start, end)))
    #df_rcv = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_rcv, vars_rcv) for i in range(start, end)))
    
# Concat all dfs
df_cr_raw = df_info.set_index(['IDRSSD','date']).join([df_ri.set_index(['IDRSSD','date']),\
                           df_rc.set_index(['IDRSSD','date']),\
                           df_rcd.set_index(['IDRSSD','date']),\
                           df_rcl.set_index(['IDRSSD','date']),\
                           df_rcs.set_index(['IDRSSD','date'])], how = 'inner')
    
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

#--------------------------------------------
# Home Mortgage Disclosure Act
#--------------------------------------------

''' Again similar setup. Because HMDA contains large files, we first
    clean the data and than aggregate to cert. We also load the lender
    file to be able to merge with SDI '''

#--------------------------------------------
# Setup
# NOTE: Script can also handle data from 2018-19
# Get Paths
path_lf = r'D:/RUG/Data/Data_HMDA_lenderfile/'
path_hmda = r'D:/RUG/Data/Data_HMDA/LAR/'
path_hmda_panel = r'D:/RUG/Data/Data_HMDA/Panel/'

# Get file names
file_lf = r'hmdpanel17.dta'
file_lf18 = r'hmdpan2018b.dta' 
file_lf19 = r'hmdpan2019b.dta'
file_hmda_0717 = r'hmda_{}_nationwide_originated-records_codes.zip'
file_hmda_1819 = r'year_{}.csv'
file_hmda_panel_1819 = r'{}_public_panel_csv.csv'

## Set d-types and na-vals for HMDA LAR
dtypes_col_hmda = {'respondent_id':'object','state_code':'str', 'county_code':'str','msamd':'str',\
                   'census_tract_number':'str', 'derived_msa-md':'str'}
na_values = ['NA   ', 'NA ', '...',' ','NA  ','NA      ','NA     ','NA    ','NA', 'Exempt', 'N/AN/', 'na']

# Get variable names
vars_hmda = ['respondent_id', 'hmda_gse_amount', 'hmda_priv_amount',\
             'hmda_sec_amount', 'loan_amount_000s']
#vars_hmda_panel_1017 = ['Respondent ID','parent_rssd']
vars_hmda_panel_1819 = ['lei', 'arid_2017', 'tax_id']
vars_lf = ['hmprid'] + ['ENTITY{}'.format(str(year)[2:4]) for year in range(start, end)]

# df LF
df_lf = pd.read_stata(path_lf + file_lf, columns = vars_lf)
df_lf18 = pd.read_stata(path_lf + file_lf18, columns = ['hmprid', 'ENTITY18'])
df_lf19 = pd.read_stata(path_lf + file_lf19, columns = ['hmprid', 'ENTITY19'])

# Reduce dimensions df_lf
df_lf.dropna(how = 'all', subset = vars_lf[1:], inplace = True) # drop rows with all na
df_lf = df_lf[~(df_lf[vars_lf[1:]] == 0.).any(axis = 1)] # Remove ENTITY with value 0.0
df_lf18 = df_lf18[~(df_lf18['ENTITY18'] == 0.)] 
df_lf19 = df_lf19[~(df_lf19['ENTITY19'] == 0.)]

df_lf = df_lf[df_lf[vars_lf[1:]].all(axis = 1)] # Drop rows that have different column values (nothing gets deleted: good)

# Reshape
lf_reshaped_list = [df_lf.loc[:,df_lf.columns.str.contains('hmprid|{}'.format(str(year).zfill(2)))].dropna().rename(columns = {'ENTITY{}'.format(str(year).zfill(2)):'entity'}) for year in range(start - 2000, end - 2000)]

# Add dates
for  i, year in zip(range(len(range(start, end + 1))),range(start, end)):
    lf_reshaped_list[i]['date'] = year
    
# Add 2018 and 2019
lf_reshaped_list.append(df_lf18.dropna().rename(columns = {'ENTITY{}'.format(18):'entity'}))
lf_reshaped_list[-1]['date'] = 2018

lf_reshaped_list.append(df_lf19.dropna().rename(columns = {'ENTITY{}'.format(19):'entity'}))
lf_reshaped_list[-1]['date'] = 2019

# Make df
df_lf_reshaped = pd.concat(lf_reshaped_list)

## Make state code dictionary
statecodes = list(range(1,56+1))

for elem in statecodes:
    if elem in [3,7,14,43,52]:
        statecodes.remove(elem)
        
states =[x for x in ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]]
state_dict = dict(zip(states,statecodes))

#--------------------------------------------
# Set functions

def loadCleanHMDA(year):
    #Load the dataframe in a temporary frame
    if year < 2018:
        df_chunk = pd.read_csv(path_hmda + file_hmda_0717.format(year),\
                               index_col = 0, chunksize = 1e6, na_values = na_values,\
                               dtype = dtypes_col_hmda)
        #df_panel = pd.read_csv(path_hmda_panel + file_hmda_panel_1017.format(year))
        #df_panel.rename(columns = {'Respondent RSSD ID':'respondent_rssd',
        #                           'Parent RSSD ID':'parent_rssd'}, inplace = True)
    else: # From 2018 onward structure of the data changes
        df_chunk = pd.read_csv(path_hmda + file_hmda_1819.format(year),\
                               index_col = 0, chunksize = 1e6, na_values = na_values,\
                               dtype = dtypes_col_hmda)
        df_panel = pd.read_csv(path_hmda_panel + file_hmda_panel_1819.format(year))
        
    chunk_list = []  # append each chunk df here 

    ### Loop over the chunks
    for chunk in df_chunk:
        # Merge with df_panel and change column names if year is 2018 or 2019
        if year >= 2018:
            ## Merge
            chunk = chunk.merge(df_panel.loc[:,vars_hmda_panel_1819], how = 'left', on = 'lei')
                    
            ## Add column of loan amount in thousand USD
            chunk['loan_amount_000s'] = chunk.loan_amount / 1e3
            
            ## Format respondent_id column  (form arid_2017: remove agency code (first string), and zero fill. Replace -1 with non)
            ## NOTE Fill the missings with the tax codes. After visual check, some arid_2017 are missing. However, the tax_id corresponds to the resp_id pre 2018
            chunk['respondent_id'] = chunk.apply(lambda x: x.tax_id if x.arid_2017 == '-1' else str(x.arid_2017)[1:], axis = 1).replace('-1',np.nan).str.zfill(10)
        
        # Filter data
        ## Drop all negative and zero incomes
        chunk = chunk[chunk.applicant_income_000s > 0]
        
        ## Make a fips column and remove separate state and county
        if year < 2018:
            chunk['fips'] = chunk['state_code'].str.zfill(2) + chunk['county_code'].str.zfill(3)
        else:
            ## NOTE: we use numpy select to correct mistakes made in the county codes (aka when the state code part is missing)
            conditions = [(chunk.county_code.astype(float) < 1001), (chunk.county_code.astype(float) >= 1001)]
            choices  = [chunk.replace({'state_code':state_dict}).state_code.astype(str).str.zfill(2) + \
                    chunk['county_code'].str.zfill(5).str[2:], chunk['county_code'].str.zfill(5)]
            
            chunk['fips'] = np.select(conditions, choices, default = chunk['county_code'].str.zfill(5))
            

        chunk.drop(columns = ['state_code', 'county_code'], inplace = True)
        
        ## Remove all unknown MSAMDs and FIPS
        chunk = chunk[(chunk.msamd.astype(float) != 0) &\
                      (chunk.msamd.astype(float) != 99999) & (chunk.fips.astype(float) != 99999)]
    
        ## Add variables
        ''' List of new vars
            Dummies loan sales, securitization
            Amounts loan sales, securitization
        '''
        
        ### Loan sale dummies
        chunk['hmda_gse_dum'] = (chunk.purchaser_type.isin(range(1,4+1))) * 1
        chunk['hmda_priv_dum'] = (chunk.purchaser_type.isin([6, 7, 8, 9, 71, 72])) * 1
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
    
    # Add the reshaped lender file
    chunk_concat = chunk_concat.merge(df_lf_reshaped[df_lf_reshaped.date == year][['hmprid', 'entity']], how = 'left', left_on = 'respondent_id',right_on = 'hmprid')
    
    ## Aggregate data on RSSD
    chunk_agg = chunk_concat.groupby('entity').sum()
    
    ## Add a date var
    chunk_agg['date'] = year
    
    return chunk_agg 

#--------------------------------------------
# Get data

# Run function
if __name__ == '__main__':    
    df_hmda = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadCleanHMDA)(i) for i in range(start, end)))

#--------------------------------------------
# Merge Data
#--------------------------------------------

# 2) CR HMDA
df_cr.reset_index(inplace = True)

df_raw = df_cr.merge(df_hmda, left_on = ['date','IDRSSD'],\
                            right_on = ['date','entity'], how = 'left')

#--------------------------------------------
# make new df
#--------------------------------------------
''' With the complete dataframe we make a new variables and df'''

# New df
df = df_raw[['date','IDRSSD', 'hmda_gse_amount', 'hmda_priv_amount',\
             'hmda_sec_amount']]

#--------------------------------------------
# Add new variables

# Net servicing fees
df['cr_serv_fees'] = df_raw.RIADB492

# Net securitization income
df['cr_sec_income'] = df_raw.RIADB493

''' Only for banks with >1billion of trading assets. also includes ABSs etc. 
# Loans pending securitization
df['cr_sec_pending'] = df_raw.RCF654
'''

# Net gains loan sales
df['cr_ls_income'] = df_raw.RIAD5416

# Sold protection credit derivatives plus sub-categories
df['cr_cd_sold'] = df_raw.loc[:,['RCC968','RCC970','RCC972','RCC974']].sum(axis = 1)
df['cr_cds_sold'] = df_raw.RCC968 # Usual CDO structure
df['cr_trs_sold'] = df_raw.RCC970 # Other possible option, see Lancaster et al (2008): Structured Products and Related Credit Derivatives: A Comprehensive Guide for Investors 
df['cr_co_sold'] = df_raw.RCC972
df['cr_cdoth_sold'] = df_raw.RCC974

# Purchased protection credit derivativesplus sub-categories
# NOTE If the bank does CDO securitization, it is most likely to purchase CDSs and TRSs. Other forms do exist
df['cr_cd_purchased'] = df_raw.loc[:,['RCC969','RCC971','RCC973','RCC975']].sum(axis = 1)
df['cr_cds_purchased'] = df_raw.RCC969
df['cr_trs_purchased'] = df_raw.RCC971
df['cr_co_purchased'] = df_raw.RCC973
df['cr_cdoth_purchased'] = df_raw.RCC975

'''OLD
# On-balance sheet securitization exposures
df['cr_se_on'] = df_raw.loc[:,['RCFDS475','RCFDS480','RCFDS485','RCFDS490']].sum(axis = 1)

# Off-balance sheet securitization exposures 
df['cr_se_off'] = df_raw.RCFDS495
'''
# Assets sold and securitized with recourse
'''Note: separate the largest categories (see SIFMA 2017Q4, not counting CDOs). Ordered:
    1) RMBS
    2) Auto
    3) Student (not available)
    4) Credit card
    
    Because of data availability (not enough variation) we only keep RMBS, and auto, the
    group the others in a rest category''' 
df['cr_as_sec'] = df_raw.loc[:,['RCB{}'.format(i) for i in range(705,711+1)]].sum(axis = 1)
df['cr_as_abs'] = df_raw.loc[:,['RCB{}'.format(i) for i in range(706,711+1)]].sum(axis = 1)
df['cr_as_rmbs'] = df_raw.RCB705
df['cr_as_hel'] = df_raw.RCB706
df['cr_as_ccr'] = df_raw.RCB707
df['cr_as_auto'] = df_raw.RCB708
df['cr_as_ocl'] = df_raw.RCB709
df['cr_as_cil'] = df_raw.RCB710
df['cr_as_aol'] = df_raw.RCB711

# Assets sold and not securitized with recourse
# Note: split in same categories as cr_as_.... Auto does not have enough variation
df['cr_as_nonsec'] = df_raw.loc[:,['RCB{}'.format(i) for i in range(790,796+1)]].sum(axis = 1)
df['cr_as_nsoth'] = df_raw.loc[:,['RCB{}'.format(i) for i in range(791,796+1)]].sum(axis = 1)
df['cr_as_nsres'] = df_raw.RCB790
df['cr_as_nshel'] = df_raw.RCB791
df['cr_as_nsccr'] = df_raw.RCB792
df['cr_as_nsauto'] = df_raw.RCB793
df['cr_as_nsocl'] = df_raw.RCB794
df['cr_as_nscil'] = df_raw.RCB795
df['cr_as_nsaol'] = df_raw.RCB796

# Maximum amount of credit exposure provided to other institutions
df['cr_ce_sec'] = df_raw.loc[:,['RCB{}'.format(i) for i in range(776,782+1)]].sum(axis = 1)
df['cr_ce_abs'] = df_raw.loc[:,['RCB{}'.format(i) for i in range(777,782+1)]].sum(axis = 1)
df['cr_ce_rmbs'] = df_raw.RCB776
df['cr_ce_hel'] = df_raw.RCB777
df['cr_ce_ccr'] = df_raw.RCB778
df['cr_ce_auto'] = df_raw.RCB779
df['cr_ce_ocl'] = df_raw.RCB780
df['cr_ce_cil'] = df_raw.RCB781
df['cr_ce_aol'] = df_raw.RCB782

# Unused Commitments provided to other institutions securitization activities
df['cr_uc_sec'] = df_raw.loc[:,['RCB{}'.format(i) for i in range(783,789+1)]].sum(axis = 1)
df['cr_uc_abs'] = df_raw.loc[:,['RCB{}'.format(i) for i in range(784,789+1)]].sum(axis = 1)
df['cr_uc_rmbs'] = df_raw.RCB783

# Outstanding principle balance small business obligations transferred with recourse
df['cr_as_sbo'] = df_raw.RCA249

# Maximum credit exposure to ABCP conduits (sponsered by the institution itself or by others)
df['cr_abcp_ce'] = df_raw.loc[:,['RCB806','RCB807']].sum(axis = 1)
df['cr_abcp_ce_own'] = df_raw.RCB806
df['cr_abcp_ce_oth'] = df_raw.RCB807

# Unused commitments to provide liquidity to ABCP conduits (sponsered by the institution itself or by others)
df['cr_abcp_uc'] = df_raw.loc[:,['RCB808','RCB809']].sum(axis = 1)
df['cr_abcp_uc_own'] = df_raw.RCB808
df['cr_abcp_uc_oth'] = df_raw.RCB809

'''OLD
# Total Assets Securitization Vehicles
# NOTE: We only use the total assets of the securitization vehicles, because there is no
# straight-forward variable measuring outstanding ABSs/CDOs. These vehicles can be both
# ABS SPVs or CDO SPVs (or anything similar but not ABCP conduits/SIV)
vars_secveh = ['RCJ{}'.format(i) for i in range(981,999+1,3)] +\
              ['RCK{}'.format(str(i).zfill(3)) for i in range(3,12+1,3)]
df['cr_secveh_ta'] = df_raw.loc[:,vars_secveh].sum(axis = 1)
df['cr_secveh_bm'] = df_raw.loc[:,['RCK021','RCK024']].sum(axis = 1) # other borrowed money and commercial paper
df['cr_secveh_der'] = df_raw.RCK018

# Total Assets ABCP Conduits
# NOTE: The real exposure to ABCP comes from the conduits liabilities, especially
# its commercial papers and repos. We included total assets in case these two 
# variables do not have enough variation 
vars_abcp = ['RCJ{}'.format(i) for i in range(982,997+1,3)] +\
              ['RCK{}'.format(str(i).zfill(3)) for i in range(1,13+1,3)]
df['cr_abcp_ta'] = df_raw.loc[:,vars_abcp].sum(axis = 1)
df['cr_abcp_bm'] = df_raw.loc[:,['RCK022','RCK025']].sum(axis = 1) # other borrowed money and commercial paper
df['cr_abcp_cp'] = df_raw.RCK022
df['cr_abcp_repo'] = df_raw.RCK016 # ALL ZERO

# Total Assets Other VIE
vars_vie_other = ['RCJ{}'.format(i) for i in range(983,988+1,3)] +\
              ['RCK{}'.format(str(i).zfill(3)) for i in range(2,14+1,3)]
df['cr_ta_vie_other'] = df_raw.loc[:,vars_vie_other].sum(axis = 1)
'''
# Total assets
df['ta'] = df_raw.RC2170

#--------------------------------------------
# Fill NA
df.fillna(0, inplace = True)

# Remove negative values in cr_ta_secveh
#df = df[df.cr_secveh_ta >= 0]

# NOTE: All bank-years have at least one non-zero value 

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

'''Notes:
    Only one clear outlier (nsmallest) in cr_serv_fees (1)
    '''

## Drop cr_abcp_repo
#df.drop(columns = 'cr_abcp_repo', inplace = True)
    
## Remove outliers
df = df.loc[df.cr_serv_fees != df.cr_serv_fees.min(),:]

#--------------------------------------------
# Save Data
#--------------------------------------------

df.to_csv('Data\df_sec_note.csv')

#--------------------------------------------
# Data Setup Securitization Note: Non-securitization data only
# Mark van der Plaat
# September 2020  -- Update: March 2021
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
start = 2009
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
path_info = r'D:/RUG/Data/Data_call_reports_fed'

# Get filenames per schedule
## INFO
file_info1 = r'/call0{}12.xpt'
file_info2 = r'/call{}12.xpt'

## RI
file_ri = r'/{}/FFIEC CDR Call Schedule RI 1231{}.txt'

## RI-B
file_ribi = r'/{}/FFIEC CDR Call Schedule RIBI 1231{}.txt'
file_ribii = r'/{}/FFIEC CDR Call Schedule RIBII 1231{}.txt'

## RC
file_rc = r'/{}/FFIEC CDR Call Schedule RC 1231{}.txt'

## RC-B
file_rcb = r'/{}/FFIEC CDR Call Schedule RCB 1231{}(1 of 2).txt'

## RC-C
file_rcc = r'/{}/FFIEC CDR Call Schedule RCCI 1231{}.txt'

## RC-N
file_rcn = r'/{}/FFIEC CDR Call Schedule RCN 1231{}.txt'
file_rcn1 = r'/{}/FFIEC CDR Call Schedule RCN 1231{}(1 of 2).txt'
file_rcn2 = r'/{}/FFIEC CDR Call Schedule RCN 1231{}(2 of 2).txt'

## RC-R
file_rcr1_rcfd = r'/{}/FFIEC CDR Call Schedule RCR 1231{}(1 of 2).txt'
file_rcr2_rcfd = r'/{}/FFIEC CDR Call Schedule RCRII 1231{}(1 of 2).txt' #2014
file_rcr3_rcfd = r'/{}/FFIEC CDR Call Schedule RCRII 1231{}(1 of 3).txt' #from 2015
file_rcr4_rcfd = r'/{}/FFIEC CDR Call Schedule RCRII 1231{}(1 of 4).txt' #from 2017
file_rcr1_rcon = r'/{}/FFIEC CDR Call Schedule RCR 1231{}(2 of 2).txt'
file_rcr2_rcon = r'/{}/FFIEC CDR Call Schedule RCRII 1231{}(2 of 2).txt' #2014
file_rcr3_rcon = r'/{}/FFIEC CDR Call Schedule RCRII 1231{}(2 of 3).txt' #from 2015
file_rcr4_rcon = r'/{}/FFIEC CDR Call Schedule RCRII 1231{}(2 of 4).txt' #from 2017

file_rcria = r'/{}/FFIEC CDR Call Schedule RCRIA 1231{}.txt' #2014
file_rcrib = r'/{}/FFIEC CDR Call Schedule RCRIB 1231{}.txt' #2014
file_rcri2 = r'/{}/FFIEC CDR Call Schedule RCRI 1231{}.txt' #from 2015

# Get variable names per schedule
## INFO
vars_info = ['RSSD9001','RSSD9999','RSSD9048','RSSD9424','RSSD9170','RSSD9210','RSSD9364']

## RI
vars_ri = '|'.join(['IDRSSD', '4301', '4073', '4508', '0093', 'A517',\
                    'A518', 'HK03', 'HK04', '4340', '4074', '4079',\
                    '4093', '4010', '4065', '4107', '4115', 'B488',\
                    'B489', '4060', '4069', '4020', '4185', '4180',\
                    '4185', '4200', '4518'])

## RI-B
vars_rib = '|'.join(['IDRSSD', '4230', '3123', '4635','4605'])

## RC
vars_rc = '|'.join(['IDRSSD','2170', '0071', '0081', '2200', '3210',\
                    '2948','B529','B987','B989', '3545','B993','B995',\
                    '3548'])

## RC-B
vars_rcb = '|'.join(['IDRSSD','1773','1754'])

## RC-C
vars_rcc = '|'.join(['IDRSSD','2122', '2123', 'F159', 'F158', '1420',\
                     '1797', '5367', '5368', '1460', 'F160', 'F161',\
                     '1763', '1764', '1590', 'B538', 'B539', 'K137',\
                     'K207']) 

## RC-N
vars_rcn = '|'.join(['IDRSSD','F172', 'F174', 'F176', 'F173', 'F175', 'F177',\
                     '3493', '3494', '3495', '5398', '5399', '5400',\
                     'C236', 'C237','C229', 'C238', 'C239', 'C230',\
                     '3499', '3500', '3501', 'F178', 'F180', 'F182',\
                     'F179', 'F181', 'F183', 'B834', 'B835', 'B836',\
                     '1606', '1607', '1608', 'B575', 'B576', 'B577',\
                     'K213', 'K214', 'K215', 'K216', 'K217', 'K218',\
                     '5389', '5390', '5391', '5459', '5460', '5461',\
                     '1226','1227','1228'])

## RC-R
vars_rcr = '|'.join(['IDRSSD','7204','7205','7206', 'A223', '8274'])

#--------------------------------------------
# Set functions
# Functions for loading data

def loadInfo(i,break_point):
    ''' Function to load the info data'''
    global path_info, file_info1, file_info2, vars_info
    
    if i < break_point:
        df_load = pd.read_sas((path_info + file_info1).format(i))
    else:
        df_load = pd.read_sas((path_info + file_info2).format(i))
        
    return(df_load[vars_info])

def loadGeneral(i, file, var_list):
    ''' A General function for loading call reports data, no breaks in file
        names'''
    
    df_load = pd.read_csv((path_call + file).format(i,i), sep='\t',  skiprows = [1,2])
    df_load['date'] = int('{}'.format(i))  
    
    return(df_load.loc[:,df_load.columns.str.contains(var_list + '|date')])

def loadTwoFiles(i, file1, file2, var_list):
    ''' Dedicated function to load RIB/RCN'''
    df_load_1 = pd.read_csv((path_call + file1).format(i,i), sep='\t',  skiprows = [1,2])
    df_load_2 = pd.read_csv((path_call + file2).format(i,i), sep='\t',  skiprows = [1,2])
    df_load = df_load_1.merge(df_load_2, on = 'IDRSSD', how = 'left')
    df_load['date'] = int('{}'.format(i))
    
    return(df_load.loc[:,df_load.columns.str.contains(var_list + '|date')])

def loadRCN(i, file1, file2, file3, var_list):
    ''' Dedicated function to load RIB/RCN'''
    if i in (2009, 2010):
        df_load = pd.read_csv((path_call + file1).format(i,i), sep='\t',  skiprows = [1,2])
    else:
        df_load_1 = pd.read_csv((path_call + file2).format(i,i), sep='\t',  skiprows = [1,2])
        df_load_2 = pd.read_csv((path_call + file3).format(i,i), sep='\t',  skiprows = [1,2])
        df_load = df_load_1.merge(df_load_2, on = 'IDRSSD', how = 'left')
    
    df_load['date'] = int('{}'.format(i))
    
    return(df_load.loc[:,df_load.columns.str.contains(var_list + '|date')])

def loadRCR(i):
    '''Function to load the RC-R data'''
    if i == 2014:
        df_load_rcfd = pd.read_csv((path_call + file_rcr2_rcfd).format(i,i), \
                 sep='\t',  skiprows = [1,2]) 
        df_load_rcon = pd.read_csv((path_call + file_rcr2_rcon).format(i,i), \
                 sep='\t',  skiprows = [1,2])
        df_load_rcria = pd.read_csv((path_call + file_rcria).format(i,i), \
                 sep='\t',  skiprows = [1,2])
        df_load_rcrib = pd.read_csv((path_call + file_rcrib).format(i,i), \
                 sep='\t',  skiprows = [1,2])
    
        df_load = df_load_rcfd.merge(df_load_rcon, on = 'IDRSSD', how = 'left')
        df_load = df_load.merge(df_load_rcria, on = 'IDRSSD', how = 'left')
        df_load = df_load.merge(df_load_rcrib, on = 'IDRSSD', how = 'left')
        
    elif i == 2015 or i == 2016:
        df_load_rcfd = pd.read_csv((path_call + file_rcr3_rcfd).format(i,i), \
                 sep='\t',  skiprows = [1,2]) 
        df_load_rcon = pd.read_csv((path_call + file_rcr3_rcon).format(i,i), \
                 sep='\t',  skiprows = [1,2])
        df_load_rcri = pd.read_csv((path_call + file_rcri2).format(i,i), \
                 sep='\t',  skiprows = [1,2])
        
        df_load = df_load_rcfd.merge(df_load_rcon, on = 'IDRSSD', how = 'left')
        df_load = df_load.merge(df_load_rcri, on = 'IDRSSD', how = 'left')
        
    elif i > 2016:
        df_load_rcfd = pd.read_csv((path_call + file_rcr4_rcfd).format(i,i), \
                 sep='\t',  skiprows = [1,2]) 
        df_load_rcon = pd.read_csv((path_call + file_rcr4_rcon).format(i,i), \
                 sep='\t',  skiprows = [1,2])
        df_load_rcri = pd.read_csv((path_call + file_rcri2).format(i,i), \
                 sep='\t',  skiprows = [1,2])
        
        df_load = df_load_rcfd.merge(df_load_rcon, on = 'IDRSSD', how = 'left')
        df_load = df_load.merge(df_load_rcri, on = 'IDRSSD', how = 'left')
        
    else:
        df_load_rcfd = pd.read_csv((path_call + file_rcr1_rcfd).format(i,i), \
                 sep='\t',  skiprows = [1,2]) 
        df_load_rcon = pd.read_csv((path_call + file_rcr1_rcon).format(i,i), \
                 sep='\t',  skiprows = [1,2]) 
 
        df_load = df_load_rcfd.merge(df_load_rcon, on = 'IDRSSD', how = 'left')
        
    df_load['date'] = int('{}'.format(i))
    
    return(df_load.loc[:,df_load.columns.str.contains(vars_rcr + '|date')])

# Functions to combine variables
def combineVars(data, elem):
    ''' Function to combine RCFD and RCON into one variable '''
    data['RC' + elem] = data['RCFD' + elem].fillna(data['RCON' + elem])
    
    return(data['RC' + elem])

def combineVarsAlt(data, elem):
    data['RC' + elem] = data['RCF' + elem].fillna(data['RCO' + elem])
    
    return(data['RC' + elem])

#--------------------------------------------
# Load Data

# Run functions
if __name__ == '__main__':
    df_info = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadInfo)(i,10) for i in range(start - 2000, end - 2000)))
    df_ri = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_ri, vars_ri) for i in range(start, end)))
    df_rib = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadTwoFiles)(i, file_ribi, file_ribii, vars_rib) for i in range(start, end))) 
    df_rc = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_rc, vars_rc) for i in range(start, end)))
    df_rcb = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_rcb, vars_rcb) for i in range(start, end)))
    df_rcc = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_rcc, vars_rcc) for i in range(start, end)))
    df_rcn = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadRCN)(i, file_rcn, file_rcn1, file_rcn2, vars_rcn) for i in range(start, end)))
    df_rcr = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadRCR)(i) for i in range(start, end)))
    
# Change variables in df_info
## Rename RSSD9001 and RSSD9999
df_info.rename(columns = {'RSSD9001':'IDRSSD', 'RSSD9999':'date'}, inplace = True)

## Change date to only the year
df_info.date = (df_info.date.round(-4) / 1e4).astype(int)

# Concat all dfs
df_raw = df_ri.set_index(['IDRSSD','date']).join([df_rib.set_index(['IDRSSD','date']),\
                                                  df_rc.set_index(['IDRSSD','date']),\
                                                  df_rcb.set_index(['IDRSSD','date']),\
                                                  df_rcc.set_index(['IDRSSD','date']),\
                                                  df_rcn.set_index(['IDRSSD','date']),\
                                                  df_rcr.set_index(['IDRSSD','date']),\
                                                  df_info.set_index(['IDRSSD','date'])],how = 'inner')

#--------------------------------------------
# Transform variables

# Get double variable
## NOTE: Don't use RCFN
vars_raw = df_raw.columns[~df_raw.columns.str.contains('RCFN|RCOA|RCOW|RCFA|RCFW')].str[4:]
var_num_raw = [item for item, count in collections.Counter(vars_raw).items() if count > 1]

## Remove the regulatory variables
var_num_reg = ['7204','7205','7206', 'A223']
var_num = [item for item in var_num_raw if item not in var_num_reg]

# Combine variables
if __name__ == '__main__':
    df_raw_combvars = pd.concat(Parallel(n_jobs=num_cores)(delayed(combineVars)(df_raw, elem) for elem in var_num), axis = 1)

## Transform regulatory data 
### Drop variables not needed
vars_drop = '|'.join(['RCFW','RCOW'])
df_rcr = df_rcr.loc[:,~df_rcr.columns.str.contains(vars_drop)]

### Transform RCFA and RCOA to float
vars_trans = '|'.join(['RCFA7','RCOA7'])
df_rcr.loc[:,df_rcr.columns.str.contains(vars_trans)] = df_rcr.loc[:,df_rcr.columns.str.contains(vars_trans)].apply(lambda x: x.str.strip('%').astype(float) / 100)

### Make the variables
for elem in var_num_reg:
    df_rcr['RCF{}'.format(elem)] = df_rcr.loc[:,['RCFA{}'.format(elem), 'RCFD{}'.format(elem)]].sum(axis = 1)
    df_rcr['RCO{}'.format(elem)] = df_rcr.loc[:,['RCOA{}'.format(elem), 'RCON{}'.format(elem)]].sum(axis = 1)
    
if __name__ == '__main__':
    df_rcr_combvars = pd.concat(Parallel(n_jobs=num_cores)(delayed(combineVarsAlt)(df_rcr, elem) for elem in var_num_reg), axis = 1)

# Remove old variables and make new df
cols_remove =  [col for col in df_raw.columns if not col[4:] in var_num]
df_raw_combvars[['RC{}'.format(elem) for elem in var_num_reg]] = np.array(df_rcr_combvars)       
df = pd.concat([df_raw[cols_remove], df_raw_combvars], axis = 1) 

# Sort index
df.sort_index(level = 0,inplace = True)

#--------------------------------------------
# Clean df
#--------------------------------------------

# Select insured and commercial banks
'''RSSD9048 == 200, RSSD9424 != 0'''
df = df[(df.RSSD9048 == 200) & (df.RSSD9424 != 0)]

# Only take US states, no territories
''' Based on the state codes provided by the FRB'''
df = df[df.RSSD9210.isin(range(1,57))]

# Drop all banks with three years or fewer of observations
idrssds = df.index.get_level_values('IDRSSD')
drop_banks = idrssds.value_counts()[idrssds.value_counts() <= 2].index.unique().tolist()
df = df[~idrssds.isin(drop_banks)]

# Drop missings in Total assets, loans, deposits
nandrop_cols = ['RC2170','RC2122','RCON2200']
df.dropna(subset = nandrop_cols , inplace = True)

# Drop negative and zero values in Total assets, loans, deposits, tier 1 capital ratio
df = df[(df.RC2170 > 0) & (df.RC2122 > 0) & (df.RCON2200 > 0)]

#--------------------------------------------
# Make new dataset 
#--------------------------------------------

# Set new df
df_clean = pd.DataFrame()

#--------------------------------------------
# Add variables

# Z-score
## Return on average assets (ROAA)
#ave_assets = df.groupby('IDRSSD').rolling(window = 2, min_periods = 2).RC2170.mean().tolist()
#df_clean['roaa'] = df.RIAD4301.divide(ave_assets)
df_clean['roa'] = df.RIAD4301.divide(df.RC2170)

## Equity capital to total assets
df_clean['cap_ratio'] = df.RC3210.divide(df.RC2170)

## std ROAA
#df_clean['std_roaa'] = df_clean.groupby('IDRSSD').rolling(window = 3, min_periods = 3, center = True).roaa.std().tolist()
#df_clean['std_roaa'] = df_clean.groupby('IDRSSD').roaa.transform(np.std)
df_clean['std_roa'] = df_clean.groupby('IDRSSD').roa.transform(np.std)

## Z-score
df_clean['zscore'] = (df_clean.loc[:,['roa','cap_ratio']].sum(axis = 1)).divide(df_clean.std_roa)

# Size
df_clean['ln_ta'] = np.log(1 + df.RC2170)

# Net interest margin
df_clean['nim'] = df.RIAD4074.divide(df.RC2170)

# Cost to income (efficiency ratio)
df_clean['cti'] = df.RIAD4093.divide(df[['RIAD4074','RIAD4079']].sum(axis = 1))

# Liquidity ratio
df_clean['liq_ratio'] = df[['RC0071', 'RC0081', 'RC1754', 'RC1773','RCONB987','RCB989','RC3545']].sum(axis = 1).divide(df[['RCON2200','RCONB993','RCB995','RC3548']].sum(axis = 1))

# Loan ratio
df_clean['loan_ratio'] = df.RCB529.divide(df.RC2170)

# Gap between liquid assets and short-term liabilities 
df_clean['gap'] = (df[['RC0071', 'RC0081', 'RC1754', 'RC1773']].sum(axis = 1).subtract(df[['RCON2200','RCONB993','RCB995','RC3548']].sum(axis = 1))).divide(df.RC2170)

# Tier 1 capital ratio
#df_clean['t1_regcap'] = df.RC7205

# Non-interest income to interest income 
df_clean['nni_ii'] = df.RIAD4079.divide(df.RIAD4107)

# Log net loans
df_clean['loan_log'] = np.log(1 + (df.RC2122 + df.RC2123))

# Gross loans
df_clean['loan_gross'] = df.RC2122 + df.RC2123

# Net loans
df_clean['loan_net'] = df.RCB529

#--------------------------------------------
# Remove outliers

# Drop inf in liq_ratio
df_clean = df_clean[df_clean.liq_ratio < np.inf]

# remove outliers in liq_ratio and cti
df_clean = df_clean[(df_clean.liq_ratio.between(0,df_clean.liq_ratio.quantile(q = .995), inclusive = True)) & (df_clean.cti.between(0,df_clean.cti.quantile(q = .995), inclusive = True))]

'''OLD
# Operational income 
df_clean['oper_inc'] = df.RIAD4301

# Total Assets
df_clean['ta'] = df.RC2170

# Liquidity ratio
df_clean['liq_ratio'] = df[['RC0071', 'RC0081', 'RC1754', 'RC1773']].sum(axis = 1).divide(df.RC2170)

# Trading asset ratio
df_clean['trad_ratio'] = df.RC3545.divide(df.RC2170)

# Loan Ratio
df_clean['loan_ratio'] = df[['RC2122', 'RC2123']].sum(axis = 1).divide(df.RC2170)

# Loans-to-deposits
## NOTE: RC1400 == RC2122 + RC2123
df_clean['loan_dep'] = df[['RC2122', 'RC2123']].sum(axis = 1).divide(df.RCON2200)

# Deposit ratio
df_clean['dep_ratio'] = df.RCON2200.divide(df.RC2170)

# Capital ratio
df_clean['cap_ratio'] = df.RC3210.divide(df.RC2170)

# Real estate loan ratio
df_clean['ra_ratio'] = df[['RCF158','RCF159','RC1420',\
                           'RC1797','RC5367','RC5368', 'RC1460',\
                           'RCF160', 'RCF161']].sum(axis = 1).divide(df[['RC2122', 'RC2123']].sum(axis = 1))

# Commercial and industrial loan ratio
df_clean['ci_ratio'] = df[['RC1763','RC1764']].sum(axis = 1).divide(df[['RC2122', 'RC2123']].sum(axis = 1))

# Agricultural loan ratio
df_clean['agri_ratio'] = df.RC1590.divide(df[['RC2122', 'RC2123']].sum(axis = 1))

# Consumer loan ratio
df_clean['cons_ratio'] = df[['RCB538', 'RCB539', 'RCK137', 'RCK207']].sum(axis = 1).divide(df[['RC2122', 'RC2123']].sum(axis = 1))

# Other loan ratio
df_clean['othl_ratio'] = df[['RC2122', 'RC2123']].sum(axis = 1).subtract(df[['RCF158','RCF159','RC1420',\
                           'RC1797','RC5367','RC5368', 'RC1460',\
                           'RCF160', 'RCF161', 'RC1763','RC1764',\
                           'RC1590', 'RCB538', 'RCB539', 'RCK137',\
                           'RCK207']].sum(axis = 1)).divide(df[['RC2122', 'RC2123']].sum(axis = 1))

# loan HHI
df_clean['loan_hhi'] = df_clean.ra_ratio**2 + df_clean.ci_ratio**2 +\
                       df_clean.agri_ratio**2 + df_clean.cons_ratio**2 +\
                       df_clean.othl_ratio**2

# Tier 1 leverage 
df_clean['t1_reglev'] = df.RC7204

# Tier 1 capital ratio
df_clean['t1_regcap'] = df.RC7205

# Total regulatory capital ratio
df_clean['tot_regcap'] = df.RC7206

# RWA / TA
df_clean['rwata'] = df.RCA223.divide(df.RC2170)

# NPL Ratio
## Set column labels
### NOTE: Use a loop to get all correct variable names, see FFIEC 031/041 for the structure
npl90 = df.columns[df.columns.str.contains('|'.join([vars_rcn[len('IDRSSD') + 1:].split('|')[i] for i in range(1,len(vars_rcn[len('IDRSSD') + 1:].split('|')),3)]))]
nplna = df.columns[df.columns.str.contains('|'.join([vars_rcn[len('IDRSSD') + 1:].split('|')[i] for i in range(2,len(vars_rcn[len('IDRSSD') + 1:].split('|')),3)]))]

## Make variables
df_clean['npl'] = df[npl90.append(nplna)].sum(axis = 1).divide(df[['RC2122', 'RC2123']].sum(axis = 1))

# Charge-off ratio
df_clean['co_ratio'] = df.RIAD4635.subtract(df.RIAD4605_x).divide(df[['RC2122', 'RC2123']].sum(axis = 1))

# Allowance ratio
df_clean['all_ratio'] = df.RIAD3123.divide(df[['RC2122', 'RC2123']].sum(axis = 1))

# Provision ratio
df_clean['prov_ratio'] = df.RIAD4230.divide(df[['RC2122', 'RC2123']].sum(axis = 1))

# Interest expense / Total liabilities
df_clean['intexp'] = df.RIAD4073.divide(df.RC2948)

# Interest expense deposits / Total deposits
df_clean['intexp_dep'] = df[['RIAD4508', 'RIAD0093', 'RIADA517', 'RIADA518',\
                             'RIADHK03', 'RIADHK04']].sum(axis = 1).divide(df.RCON2200)

# Return on equity
df_clean['roe'] = df.RIAD4340.divide(df.RC3210)

# Return on assets
df_clean['roa'] = df.RIAD4340.divide(df.RC2170)

# Net interest margin
df_clean['nim'] = df.RIAD4074.divide(df.RC2170)

# Cost to income
df_clean['cti'] = df.RIAD4093.divide(df[['RIAD4074','RIAD4079']].sum(axis = 1))

# Revenue HHI
df_clean['rev_hhi'] = df.RIAD4074.divide(df[['RIAD4074','RIAD4079']].sum(axis = 1))**2 +\
                      df.RIAD4079.divide(df[['RIAD4074','RIAD4079']].sum(axis = 1))**2

# Non-insterest income / net operating revenue 
df_clean['nii_nor'] = df.RIAD4079.divide(df[['RIAD4074','RIAD4079']].sum(axis = 1))

# Interest income: loans
df_clean['ii_loans'] = df[['RIAD4010','RIAD4065']].sum(axis = 1).divide(df.RIAD4107)

# Interest income: depository institutions
df_clean['ii_depins'] = df.RIAD4115.divide(df.RIAD4107)

# Interest income: securities
df_clean['ii_sec'] = df[['RIADB488','RIADB489','RIAD4060']].sum(axis = 1).divide(df.RIAD4107)

# Interest income: trading assets
df_clean['ii_trad'] = df.RIAD4069.divide(df.RIAD4107)

# Interest income: REPO
df_clean['ii_repo'] = df.RIAD4020.divide(df.RIAD4107)

# Interest income: other
df_clean['ii_oth'] = df.RIAD4518.divide(df.RIAD4107)

# Interest income HHI 
df_clean['ii_hhi'] = df_clean.ii_loans**2 + df_clean.ii_depins**2 + df_clean.ii_sec**2 +\
                     df_clean.ii_repo**2 + df_clean.ii_oth**2

# Interest expenses: deposits
df_clean['ie_dep'] = df[['RIAD4508','RIAD0093','RIADA517', 'RIADA518']].sum(axis = 1).divide(df.RIAD4073)

# Interest expenses: repo
df_clean['ie_repo'] = df.RIAD4180.divide(df.RIAD4073)


# Interest expenses: Trading liabilities and other borrowed money
df_clean['ie_trad'] = df.RIAD4185.divide(df.RIAD4073)

# Interest expenses: subordinated notes
df_clean['ie_sub'] = df.RIAD4200.divide(df.RIAD4073)

# Interest expenses HHI
df_clean['ie_hhi'] = df_clean.ie_dep**2 + df_clean.ie_repo**2 +\
                     df_clean.ie_trad**2 + df_clean.ie_sub**2
'''

#--------------------------------------------
# Save Data
#--------------------------------------------

df_clean.to_csv('Data\df_ri_rc_note.csv')
 
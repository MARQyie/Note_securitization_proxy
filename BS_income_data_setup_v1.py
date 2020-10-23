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
file_rcn1 = r'/{}/FFIEC CDR Call Schedule RCN 1231{}(1 of 2).txt'
file_rcn2 = r'/{}/FFIEC CDR Call Schedule RCN 1231{}(2 of 2).txt'

## RC-D
file_rcd = r'/{}/FFIEC CDR Call Schedule RCD 1231{}.txt'

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
                    '2948'])

## RC-B
vars_rcb = '|'.join(['IDRSSD','1773','1754'])

## RC-C
vars_rcc = '|'.join(['IDRSSD','2122', '2123', 'F159', 'F158', '1420',\
                     '1797', '5367', '5368', '1460', 'F160', 'F161',\
                     '1763', '1764', '1590', 'B538', 'B539', 'K137',\
                     'K207']) 

## RC-D
vars_rcd = '|'.join(['IDRSSD','3545'])

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
vars_rcr = '|'.join(['IDRSSD','7204','7205','7206', 'A223'])

#--------------------------------------------
# Set functions
# Functions for loading data

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
    data['RC' + elem] = data.apply(lambda x: x['RCFD' + elem] if not np.isnan(x['RCFD' + elem]) and  round(x['RCFD' + elem]) != 0 else (x['RCON' + elem]), axis = 1) 
    
    return(data['RC' + elem])

#--------------------------------------------
# Load Data

# Run functions
if __name__ == '__main__':
    df_ri = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_ri, vars_ri) for i in range(start, end)))
    df_rib = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadTwoFiles)(i, file_ribi, file_ribii, vars_rib) for i in range(start, end))) 
    df_rc = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_rc, vars_rc) for i in range(start, end)))
    df_rcb = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_rcb, vars_rcb) for i in range(start, end)))
    df_rcc = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_rcc, vars_rcc) for i in range(start, end)))
    df_rcd = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadGeneral)(i, file_rcd, vars_rcd) for i in range(start, end)))
    df_rcn = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadTwoFiles)(i, file_rcn1, file_rcn2, vars_rcn) for i in range(start, end)))
    df_rcr = pd.concat(Parallel(n_jobs=num_cores)(delayed(loadRCR)(i) for i in range(start, end)))

# Concat all dfs
df_raw = df_ri.set_index(['IDRSSD','date']).join([df_rib.set_index(['IDRSSD','date']),\
                                                  df_rc.set_index(['IDRSSD','date']),\
                                                  df_rcb.set_index(['IDRSSD','date']),\
                                                  df_rcc.set_index(['IDRSSD','date']),\
                                                  df_rcd.set_index(['IDRSSD','date']),\
                                                  df_rcn.set_index(['IDRSSD','date']),\
                                                  df_rcr.set_index(['IDRSSD','date'])], how = 'inner')

#--------------------------------------------
# Transform variables

# Get double variable
## NOTE: Don't use RCFN
vars_raw = df_raw.columns[~df_raw.columns.str.contains('RCFN|RCOA|RCOW|RCFA|RCFW')].str[4:]
var_num = [item for item, count in collections.Counter(vars_raw).items() if count > 1]

# Combine variables
if __name__ == '__main__':
    df_raw_combvars = pd.concat(Parallel(n_jobs=num_cores)(delayed(combineVars)(df_raw, elem) for elem in var_num), axis = 1)

## Transform regulatory data 
### Transform RCFA and RCOA to float
vars_trans = '|'.join(['RCFA7','RCOA7'])
df_rcr.loc[:,df_rcr.columns.str.contains(vars_trans)] = df_rcr.loc[:,df_rcr.columns.str.contains(vars_trans)].apply(lambda x: x.str.strip('%').astype(float) / 100)

## #Make the variables
var_num_reg = ['7204','7205','7206', 'A223']

for elem in var_num_reg:
    df_rcr['RCF{}'.format(elem)] = df_rcr.apply(lambda x: x['RCFA{}'.format(elem)] if x.date > 2014 else (x['RCFD{}'.format(elem)]), axis = 1)
    df_rcr['RCO{}'.format(elem)] = df_rcr.apply(lambda x: x['RCOA{}'.format(elem)] if x.date > 2014 else (x['RCON{}'.format(elem)]), axis = 1) 

# Remove old variables
cols_remove =  [col for col in df_raw.columns if not col[4:] in var_num]      
df = pd.concat([df_raw[cols_remove], df_raw_combvars], axis = 1) 

#--------------------------------------------
# Make clean dataset 
#--------------------------------------------

# Set new df
df_clean = pd.DataFrame()

#--------------------------------------------
# Add variables

# Operational income 
df_clean['oper_inc'] = df.RIAD4301

# Total Assets
df_clean['ta'] = df.RC2170

# Liquidity ratio
df_clean['liq_ratio'] = df[['RC0071', 'RC0081', 'RC1754', 'RC1773']].sum(axis = 1).divide(df.RC2170)

# Trading asset ratio
df_clean['trad_ratio'] = df.RC3545.divide(df.RC2170)

# Loan Ratio
df_clean['loan_ratio'] = df.RC2122.divide(df.RC2170)

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
                     df_clean.ii_trad**2 + df_clean.ii_repo**2 + df_clean.ii_oth**2

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

#--------------------------------------------
# Save Data
#--------------------------------------------

df_clean.to_csv('Data\df_ri_rc_note.csv')
 
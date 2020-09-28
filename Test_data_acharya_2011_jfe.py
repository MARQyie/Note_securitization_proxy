# Test data Acharya et al 2011 JFE

# Import packages
import pandas as pd
import numpy as np

import os
os.chdir('D:\RUG\Data\Data_ABCPs_Acharya_2013_JFE')

# Load data
## ABCP exposure
df_exp = pd.read_stata('abcp_exposure.dta')

## ABCP sponsor
df_sponsor = pd.read_stata('abcp_sponsor.dta')

''' NOTE:
    
    The data could possible be matched to U.S. commercial bank data (call reports).
    Doing so would result in a low number of observations, since many of the banks
    in the sample are European banks. 
    
    Conclusion: Do not use for note
    '''
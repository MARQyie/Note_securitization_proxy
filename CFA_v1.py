#--------------------------------------------
# Confirmatory Factor analysis for Note
# Mark van der Plaat
# March 2021
#--------------------------------------------

''' This script performs a confirmatory factor analysis
    on the proxies of securitization. 
    
    We use the procedure as described in Brown, T. A. (2015). Confirmatory
    Factor Analysis for Applied Research. In T. D. Little (Ed.), 
    (2nd ed., Vol. 53). New York: The Guilford Press.
    '''
#--------------------------------------------
# Import Packages
#--------------------------------------------
    
# Data manipulation
import pandas as pd
import numpy as np
from scipy.stats import chi2

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid', font_scale = 2)

# Machine learning packages
from sklearn import preprocessing

# CFA/SEM package
from semopy import Model, stats
# Set WD
import os
os.chdir(r'D:\RUG\PhD\Materials_papers\01-Note_on_securitization')

#--------------------------------------------
# Load data and merge
#--------------------------------------------

# Load data
df = pd.read_csv('Data\df_sec_note.csv', index_col = 0, dtype = np.float64)

# set variable names
# Variable names
vars_tot = df.columns[2:].tolist()

# Subset data to only include securitizers
unique_idrssd = df[(df[vars_tot] > 0).any(axis = 1)].IDRSSD.unique()
df_sec = df.loc[df.IDRSSD.isin(unique_idrssd),vars_tot]

# standardize data
df_standard = pd.DataFrame(preprocessing.scale(df_sec[vars_tot]), columns = vars_tot)

# Check definiteness of the var-cov matrix
cov = df[vars_tot].cov()
definite_cov = np.all(np.linalg.eigvals(cov) > 0) # True

cov_std = df_standard[vars_tot].cov()
definite_cov_std = np.all(np.linalg.eigvals(cov_std) > 0) # True

#--------------------------------------------
# Setup factor analysis function
#--------------------------------------------

# TODO: implement modification indices and expected parameter change

def CFA(mod, data):
    # Set model
    model = Model(mod)
    
    # Estimate model
    est = model.fit(data)
    
    # Get results
    res = model.inspect(se_robust = True)
    
    # Model evaluation
    ## Overall goodness of fit
    ''' NOTE The following should be true for a good fit
        SB Chi2 pval > 0.05
        SRMR close to or below 0.08
        RMSEA close to or below 0.06
        CFI/TLI close to or greater than 0.95
        '''
    ### Prelims
    dof = stats.calc_dof(model)
    
    baseline_model = stats.get_baseline_model(model)
    baseline_res = baseline_model.fit(df_standard)
    dof_base = stats.calc_dof(baseline_model)
    
    ### SB scaled chi2 
    chi_stat = (model.n_samples * model.last_result.fun) / (df_standard[model.vars['observed']].kurtosis().mean())
    chi_p = 1 - chi2.cdf(chi_stat, dof)
    
    chi_stat_base = (baseline_model.n_samples * baseline_model.last_result.fun) / (df_standard[baseline_model.vars['observed']].kurtosis().mean())
    
    ### SRMR
    #### Calculate residual covariance matrix
    s = model.mx_cov
    sigma = model.calc_sigma()[0]
    resid_cov = s - sigma
    
    #### Transform residual cov to corr
    resid_corr = np.zeros([len(s),len(s)])
    for i in range (len(resid_cov)):
        for j in range(len(resid_cov)):
            i_std, j_std = np.sqrt(s[i,i]), np.sqrt(s[j,j])
            
            resid_corr[i,j] = resid_cov[i,j] / (i_std * j_std)
    
    #### Calculate SRMR
    srmr = np.sum(np.tril(resid_corr)**2) / (len(s) * (len(s) + 1) / 2)
    
    ### RMSEA
    rmsea = stats.calc_rmsea(model, chi_stat, dof)
    
    ### CFI/TFI
    cfi = stats.calc_cfi(model, chi_stat, dof, dof_base, chi_stat_base)
    tli = stats.calc_tli(model, chi_stat, dof, dof_base, chi_stat_base)
    
    ### GFI/AGFI
    gfi = stats.calc_gfi(model, chi_stat, chi_stat_base)
    agfi = stats.calc_agfi(model, dof, dof_base, gfi)
    
    ### AIC/BIC
    aic = stats.calc_aic(model)
    bic = stats.calc_bic(model)
    
    ### Make pandas dataframe from the stats
    model_eval = pd.DataFrame([['chi_stat','chi_p','chi_stat_base','srmr', 'rmsea','cfi','tli','gfi','agfi','aic','bic'],[chi_stat, chi_p, chi_stat_base, srmr, rmsea, cfi, tli, gfi, agfi, aic, bic]]).T
    
    # Localized areas of ill fit
    # TODO: fix --> returns too many standard errors atm
    ## Standardized residuals (z-score) of the cov-var matrix
    se = stats.calc_se(model, robust = True)
    z = stats.calc_zvals(model, std_errors = se)
    z_p = stats.calc_pvals(model,z)
    
    ### Make df
    illfit_z = pd.DataFrame([se, z, z_p], index = ['se','z','p_val']).T
    
    return model, est, res, model_eval, illfit_z

#--------------------------------------------
# Setup Factor model
#--------------------------------------------

# Model specification
'''NOTE: We test various model specifications in this script. All are based 
    on theory/previous exploratory factor analyses. Main give aways:
        1) There are three types of securitization: ABS, CDO, and ABCP
        2) ABCP is not a pure form of securitization
        3) All three types are nested under a general securitization nomer
            because of the similarities in technique, structure, outcomes,
            incentives, etc.
        4) Loan sales is a seperate technique but is tightly related to securitization
        5) EFA rules there that four factors are optimum.
        6) EFA also shows that there are some bad behaving indicators (multiple
            high loadings or no high loadings). The severity is only low.
                                                                       
    We test the following factor models
        1) Simple 2 factor model: LS and SEC
        2) Simple 3 factor model: LS, ABS, CDO
        3) Simple 3 factor model: LS, SPVs, ABS, Other
        4) Simple 4 factor model: LS, ABS, CDO, ABCP
        5) Nested 4 fator model: lS, ABS, CDO, SEC =~ ABS + CDO
    '''

model_formula0 = '''LS =~ hmda_gse_amount + hmda_priv_amount + cr_ls_income + cr_as_nonsec
                    SEC =~ hmda_sec_amount + cr_as_sec + cr_serv_fees + cr_sec_income + cr_ce_sec + cr_ta_secveh + cr_cd_sold + cr_cd_purchased + cr_ta_abcp '''

model_formula1 = '''ABS =~ hmda_sec_amount + cr_as_sec + cr_serv_fees + cr_sec_income + cr_ce_sec + cr_ta_secveh 
                   CDO =~ cr_cd_sold + cr_cd_purchased + cr_serv_fees + cr_sec_income + cr_ce_sec + cr_ta_secveh 
                   LS =~ hmda_gse_amount + hmda_priv_amount + cr_serv_fees + cr_ls_income + cr_as_nonsec'''

model_formula2 = '''ABS =~ hmda_sec_amount + cr_as_sec
                   SPV =~ cr_ta_abcp + cr_ta_secveh 
                   OTH =~ cr_cd_sold + cr_cd_purchased + cr_serv_fees + cr_sec_income + cr_ce_sec
                   LS =~ hmda_gse_amount + hmda_priv_amount + cr_serv_fees + cr_ls_income + cr_as_nonsec'''
                   
model_formula3 = '''ABS =~ hmda_sec_amount + cr_as_sec + cr_serv_fees + cr_sec_income + cr_ce_sec + cr_ta_secveh 
                   CDO =~ cr_cd_sold + cr_cd_purchased + cr_serv_fees + cr_sec_income + cr_ce_sec + cr_ta_secveh 
                   ABCP =~ cr_ta_abcp + cr_serv_fees + cr_sec_income + cr_ce_sec
                   LS =~ hmda_gse_amount + hmda_priv_amount + cr_serv_fees + cr_ls_income + cr_as_nonsec'''
                   
model_formula4 = '''ABS =~ hmda_sec_amount + cr_as_sec + cr_serv_fees + cr_sec_income + cr_ce_sec + cr_ta_secveh 
                   CDO =~ cr_cd_sold + cr_cd_purchased + cr_serv_fees + cr_sec_income + cr_ce_sec + cr_ta_secveh 
                   LS =~ hmda_gse_amount + hmda_priv_amount + cr_serv_fees + cr_ls_income + cr_as_nonsec
                   
                   GEN =~ ABS + CDO'''
                  
#--------------------------------------------
# Run CFA models
#--------------------------------------------

# Make list of model formulas
models = [model_formula0, model_formula1, model_formula2,model_formula3]

# Setup list to add results to
res_lst = []

# Loop over 
for mod in models:
    res_lst.append(CFA(mod, df_standard))
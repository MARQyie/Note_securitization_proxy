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
from scipy.stats import chi2, norm

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid', font_scale = 2)

# Machine learning packages
from sklearn import preprocessing

# CFA/SEM package
from semopy import Model, stats, semplot

# Zip file manipulation
from zipfile import ZipFile

# Set WD
import os
os.chdir(r'D:\RUG\PhD\Materials_papers\01-Note_on_securitization')
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

#--------------------------------------------
# Load data and merge
#--------------------------------------------

# Load data
df = pd.read_csv('Data\df_sec_note.csv', index_col = 0, dtype = np.float64)

# Add net credit derivatives
## Note: CD purchased and sold are highly colinear. For that reason we calculate
## a net CD variable: CD purchased - CD sold. Positive net CD means that the
## the bank has purchased more protection than it has sold
#df['cr_cd_net'] = df.cr_cd_purchased - df.cr_cd_sold

# Sort columns
df = df[df.columns.sort_values()]

# set variable names
# Variable names
vars_tot = [var for var in df.columns if var not in ['IDRSSD','date','cr_cd_purchased','cr_cd_sold', 'cr_cds_sold', 'cr_trs_sold', 'cr_co_sold', 'cr_cdoth_sold']]

# Subset data to only include securitizers
# TODO REMOVE
#unique_idrssd = df[(df[vars_tot] > 0).any(axis = 1)].IDRSSD.unique()
#df_sec = df.loc[df.IDRSSD.isin(unique_idrssd),vars_tot]
df_sec = df.loc[(df[vars_tot] != 0).any(axis = 1),vars_tot]

# Take logs of the data
df_log = np.log(df_sec - df_sec.min() + 1) 
# Todo: think of something for the income variables

# standardize data
#df_standard = pd.DataFrame(preprocessing.scale(df_sec[vars_tot]), columns = vars_tot)
#df_normal = pd.DataFrame(preprocessing.normalize(df_sec[vars_tot]), columns = vars_tot)

# binary data
#df_binary = (df_sec != 0) * 1

# Use power transformation to force normality on the data
#from sklearn.preprocessing import PowerTransformer
#pt = PowerTransformer(method = 'yeo-johnson', standardize = False)
#df_power = pd.DataFrame(pt.fit_transform(df_sec), columns = vars_tot)

# Check definiteness of the var-cov matrix
cov = df_sec[vars_tot].cov()
definite_cov = np.all(np.linalg.eigvals(cov) > 0) # True

cov_log = df_log[vars_tot].cov()
definite_cov_log = np.all(np.linalg.eigvals(cov_log) > 0) # True

#cov_std = df_standard[vars_tot].cov()
#definite_cov_std = np.all(np.linalg.eigvals(cov_std) > 0) # True

#--------------------------------------------
# Setup factor analysis class funciton
#--------------------------------------------

class ModelEval:
    '''Helper class to CFA. Calculates the model evaluation statistics
    
    The following should be true for a good fit
        SB Chi2 pval > 0.05
        SRMR close to or below 0.08
        RMSEA close to or below 0.06
        GFI > 0.95
        CFI/TLI close to or greater than 0.95
        NFI > 0.95
        '''
    
    def Misc(self, model, dataframe):
        '''Calculates miscellaneous statistics '''    
        dof = stats.calc_dof(model)
    
        baseline_model = stats.get_baseline_model(model)
        baseline_res = baseline_model.fit(dataframe, obj = self.obj)
        dof_base = stats.calc_dof(baseline_model)
        
        return dof, dof_base, baseline_model
    
    def Chi2(self, model, dof, dataframe):
        if self.bool_robust: # SB scaling
            chi_stat = (model.n_samples * model.last_result.fun) / (dataframe[model.vars['observed']].kurtosis().mean()) #KLOPT NIET
        else:
            chi_stat = model.n_samples * model.last_result.fun
        chi_p = 1 - chi2.cdf(chi_stat, dof)
        
        return chi_stat, chi_p
    
    def SRMR(self):
        '''Calculate the standardized root mean squared residual '''
        # Calculate residual covariance matrix
        s = self.mod_semopy.mx_cov
        sigma = self.mod_semopy.calc_sigma()[0]
        resid_cov = s - sigma
        
        # Transform residual cov to corr
        resid_cov_std = np.zeros([len(s),len(s)])
        for i in range (len(resid_cov)):
            for j in range(len(resid_cov)):
                i_std, j_std = np.sqrt(s[i,i]), np.sqrt(s[j,j])
                
                resid_cov_std[i,j] = resid_cov[i,j] / (i_std * j_std)
        
        # Calculate SRMR
        srmr = np.sum(np.tril(resid_cov_std)**2) / (len(s) * (len(s) + 1) / 2)
        
        return s, sigma, resid_cov, resid_cov_std, srmr
    
    def GoodnessOfFit(self, model, base_model, dataframe, standardized_stats):
        ## chi2
        chi_stat, chi_p = self.Chi2(model, self.dof, dataframe)
        chi_base_stat, chi_base_p = self.Chi2(base_model, self.dof_base, dataframe)
        
        ## SRMR
        s, sigma, epsilon, epsilon_standard, srmr = self.SRMR()
        
        ## RMSEA
        if standardized_stats:
            rmsea = stats.calc_rmsea(model, chi_stat, self.dof_std)
        else:
            rmsea = stats.calc_rmsea(model, chi_stat, self.dof)
        
        ## CFI/TFI
        if standardized_stats:
            cfi = stats.calc_cfi(model, self.dof_std, chi_stat,  self.dof_std_base, chi_base_stat)
            tli = stats.calc_tli(model, self.dof_std, chi_stat, self.dof_std_base, chi_base_stat)
        else:   
            cfi = stats.calc_cfi(model, self.dof, chi_stat,  self.dof_base, chi_base_stat)
            tli = stats.calc_tli(model, self.dof, chi_stat, self.dof_base, chi_base_stat)
        
        ## GFI/AGFI
        if standardized_stats:
            gfi = stats.calc_gfi(model, chi_stat, chi_base_stat)
            agfi = stats.calc_agfi(model, self.dof_std, self.dof_std_base, gfi)
        else:   
            gfi = stats.calc_gfi(model, chi_stat, chi_base_stat)
            agfi = stats.calc_agfi(model, self.dof, self.dof_base, gfi)
    
        ## AIC/BIC
        aic = stats.calc_aic(model)
        bic = stats.calc_bic(model)
        
        return chi_stat, chi_p, chi_base_stat, chi_base_p, srmr, rmsea, cfi, tli, gfi, agfi, aic, bic
    
    def IllnessOfFit(self):
        # Standardized residuals
        _, _, _, std_resid, _ = self.SRMR() 

        return std_resid
    
    def tableGoodnessOfFit(self, chi_stat, chi_p, chi_base_stat, chi_base_p, srmr, rmsea, cfi, tli, gfi, agfi, aic, bic):
        return pd.DataFrame([['chi_stat','chi_p','chi_base_stat', 'chi_base_p', 'srmr', 'rmsea','cfi','tli','gfi','agfi','aic','bic'],[chi_stat, chi_p, chi_base_stat, chi_base_p, srmr, rmsea, cfi, tli, gfi, agfi, aic, bic]]).T
    
    def EvalStats(self, model, dataframe, standardized_stats = False):
        '''Wrapper function to calculate 'the 'statistics'''
        # prelim
        if standardized_stats:
            self.dof_std, self.dof_std_base, self.mod_std_base_semopy = self.Misc(model, dataframe)
            base_model = self.mod_std_base_semopy
        else:   
            self.dof, self.dof_base, self.mod_base_semopy = self.Misc(model, dataframe)
            base_model = self.mod_base_semopy
        
        # calculate goodness of fit statistics
        chi_stat, chi_p, chi_base_stat, chi_base_p, srmr, rmsea, cfi, tli, gfi, agfi, aic, bic = self.GoodnessOfFit(model, base_model, dataframe, standardized_stats)
    
        ## Make pandas dataframe from the stats
        eval_stats = self.tableGoodnessOfFit(chi_stat, chi_p, chi_base_stat, chi_base_p, srmr, rmsea, cfi, tli, gfi, agfi, aic, bic)
        
        return chi_stat, chi_p, chi_base_stat, chi_base_p, srmr, rmsea, cfi, tli, gfi, agfi, aic, bic, eval_stats

class CFA(ModelEval):
    '''
    This class performs a confirmatory factor analysis based on the package
    semopy, and adds robust overall goodness-of-fit statistics and several
    localized area of ill fit statistics
    '''
    
    def __init__(self, equation, data, robust = False, standardize = False, mimic_lavaan = False, baseline = False, obj = 'WLS', solver = 'L-BFGS-B'):
        '''
        Instantiate the class 
    
        Parameters
        ----------
        equation : str
            model equation
        data : pandas DataFrame
            n * m matrix, where the columns are the variables
        robust : boolean
            If True, robust results (sandwich errors) are used
        standardize : boolean
            If True, class also returns standardized results next to the
            original results
    
        Returns
        ----------
        None
        '''
        self.equation = equation
        self.data = data
        self.bool_robust = robust
        self.bool_standardize = standardize
        self.mimic_lavaan = mimic_lavaan
        self.bool_baseline = baseline
        self.obj = obj
        self.solver = solver
        
    def standardizeData(self, dataframe):
        '''Standardizes the data, goes after the estimation of the model
            on the non-standardized dataset'''
        columns = self.mod_semopy.names_theta[0]
        return pd.DataFrame(preprocessing.scale(dataframe[columns]), columns = columns)
        
    def semopyFit(self, dataframe):
        ''''Fits the model with semopy. Helper function. '''
        
        # Set model
        model = Model(self.equation, mimic_lavaan = self.mimic_lavaan, baseline = self.bool_baseline)
    
        # Estimate model
        est = model.fit(dataframe, obj = self.obj, solver = self.solver)
    
        # Get results
        res = model.inspect(se_robust = self.bool_robust)
        
        return model, est, res
    
    def fit(self):
        ''' Fit model to data '''
        # Get model results
        self.mod_semopy, self.est_semopy, self.results = self.semopyFit(self.data)
        
        # Model implied variances
        self.sigma = pd.DataFrame(self.mod_semopy.calc_sigma()[0], index = self.mod_semopy.vars['observed'],\
                                  columns = self.mod_semopy.vars['observed'])
        
        # Get model evaluation stats
        self.chi_stat, self.chi_p, self.chi_base_stat, self.chi_base_p, self.srmr, self.rmsea, self.cfi, self.tli, self.gfi, self.agfi, self.aic, self.bic, self.eval_stats = self.EvalStats(self.mod_semopy, self.data, False)
        
        # Illness of fit
        self.std_resid = self.IllnessOfFit()
        
        # Get standardized results
        if self.bool_standardize:
            ## Standardize data
            self.data_std = self.standardizeData(self.data)
            
            ## Get standardized results
            self.mod_std_semopy, self.est_std_semopy, self.results_std = self.semopyFit(self.data_std)
        
        # Get standardized evaluation stats
            self.chi_stat_std, self.chi_p_std, self.chi_base_stat_std, self.chi_base_p_std, self.srmr_std, self.rmsea_std, self.cfi_std, self.tli_std, self.gfi_std, self.agfi_std, self.aic_std, self.bic_std, self.eval_stats_std = self.EvalStats(self.mod_std_semopy, self.data_std, True)
        
        return self

#--------------------------------------------
# Setup Factor model
#--------------------------------------------

#TODO: Determine which variables to declare 'censored'
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
    '''

# Main Model
formula0 = '''LS =~ cr_as_nonsec + 1*hmda_gse_amount + hmda_priv_amount + cr_ls_income + cr_serv_fees
              ABS =~ cr_as_sec + 1*hmda_sec_amount + cr_ce_sec + cr_serv_fees + cr_ta_secveh + cr_sec_income
              CDO =~ cr_cds_purchased + cr_trs_purchased + cr_co_purchased + cr_cdoth_purchased + cr_ta_secveh + cr_sec_income
              ABCP =~ cr_ta_abcp + cr_sec_income 
              GENSEC =~ ABS + CDO + ABCP 
              
              DEFINE(latent) LS ABS CDO ABCP GENSEC
              
              LS ~~ ABS
              ''' 

# Alternative Models
## Alternative multilevel
formula1 = '''LS =~ cr_as_nonsec + 1*hmda_gse_amount + hmda_priv_amount + cr_ls_income + cr_serv_fees
              SEC1 =~ cr_as_sec + 1*hmda_sec_amount + cr_ce_sec + cr_serv_fees + cr_ta_secveh + cr_sec_income
              SEC2 =~ cr_cds_purchased + cr_trs_purchased + cr_co_purchased + cr_cdoth_purchased + cr_ta_secveh + cr_ta_abcp + cr_sec_income + cr_serv_fees
              GENSEC =~ SEC1 + SEC2
              
              DEFINE(latent) LS SEC1 SEC2 GENSEC
              
              LS ~~ GENSEC
              ''' 

## No multilevel
formula2 = '''# Measurement model
              LS =~ cr_as_nonsec + hmda_gse_amount + hmda_priv_amount + cr_ls_income + cr_serv_fees
              ABS =~ cr_as_sec + hmda_sec_amount + cr_ce_sec + cr_serv_fees + cr_ta_secveh + cr_sec_income
              CDOABCP =~ cr_cds_purchased + cr_trs_purchased + cr_co_purchased + cr_cdoth_purchased + cr_ta_secveh + cr_ta_abcp + cr_sec_income
              
              # Latent variables
              DEFINE(latent) LS ABS CDOABCP 
              
              # Correlations
              LS ~~ ABS
              ABS ~~ CDOABCP
              ''' 

formula3 = '''# Measurement model
              LS =~ cr_as_nonsec + hmda_gse_amount + hmda_priv_amount + cr_ls_income + cr_serv_fees
              ABS =~ cr_as_sec + hmda_sec_amount + cr_ce_sec
              CDOABCP =~ cr_cds_purchased + cr_trs_purchased + cr_co_purchased + cr_cdoth_purchased + cr_ta_abcp
              GENSEC =~ cr_sec_income + cr_serv_fees + cr_ta_secveh 
              
              # Latent variables
              DEFINE(latent) LS ABS CDOABCP GENSEC
              
              # Correlations
              LS ~~ ABS
              ABS ~~ CDOABCP
              ABS ~~ GENSEC
              CDOABCP ~~ GENSEC
              ''' 

## No loan sales
formula3= ''' SEC1 =~ cr_as_sec + hmda_sec_amount + cr_ce_sec + cr_serv_fees + cr_ta_secveh + cr_sec_income
              SEC2 =~ cr_cds_purchased + cr_trs_purchased + cr_co_purchased + cr_cdoth_purchased + cr_ta_secveh + cr_ta_abcp + cr_sec_income
              GENSEC =~ SEC1 + SEC2
              
              DEFINE(latent) SEC1 SEC2 GENSEC
              ''' 



# formula1 =  '''LS =~ cr_as_nonsec + hmda_priv_amount + cr_ls_income + hmda_gse_amount + cr_serv_fees
#                GENSEC =~ cr_serv_fees + cr_ta_secveh + cr_ta_vie_other + cr_sec_income 
#                ABS =~ cr_as_sec + hmda_sec_amount
#                CDOABCP =~ cr_cds_purchased + cr_trs_purchased + cr_co_purchased + cr_cdoth_purchased + cr_ta_abcp'''


# formula2 = '''LS =~ hmda_gse_amount + hmda_priv_amount + cr_ls_income + cr_as_nonsec + cr_serv_fees
#               SEC1 =~ cr_as_sec + hmda_sec_amount + cr_serv_fees + cr_ta_secveh + cr_sec_income
#               SEC2 =~ cr_cd_net + cr_ta_abcp + cr_ta_secveh + cr_sec_income
              
#               LS ~~ SEC1
#               LS ~~ SEC2
#               SEC1 ~~ SEC2''' 
             
# formula3 = '''LS =~ hmda_gse_amount + hmda_priv_amount + cr_ls_income + cr_as_nonsec + cr_serv_fees
#               SECINC =~ cr_serv_fees + cr_sec_income
#               SECASSETS =~ cr_as_sec + hmda_sec_amount + cr_cd_net + cr_ta_abcp + cr_ta_secveh
              
#               LS ~~ SECINC
#               LS ~~ SECASSETS
#               SECINC ~~ SECASSETS
#           '''
          
# formula4 = '''TA =~ cr_ta_abcp + cr_ta_secveh + cr_ta_vie_other
#               INC =~ cr_serv_fees + cr_sec_income + cr_ls_income
#               OTHER =~ hmda_gse_amount + hmda_priv_amount + cr_as_nonsec + cr_as_sec + hmda_sec_amount + cr_cd_net
          
#               TA ~~ INC
#               TA ~~ OTHER
#               INC ~~ OTHER'''
              
# formula5 = '''LS =~ hmda_gse_amount + hmda_priv_amount + cr_ls_income + cr_as_nonsec + cr_serv_fees
#               ABS =~ cr_as_sec + hmda_sec_amount + cr_ta_secveh + cr_sec_income  + cr_serv_fees
#               CDO =~ cr_cd_net + cr_ta_secveh + cr_sec_income  + cr_serv_fees
              
#               LS ~~ ABS
#               LS ~~ CDO
#               ABS ~~ CDO
#            '''

#--------------------------------------------
# Run CFA models
#--------------------------------------------

# Set procedure
def CFAWrapper(formula, obj, filename, robust = False, solver = 'SLSQP'):
    ### Fit model
    res = CFA(formula, df_log, obj = obj, robust = robust, solver = solver).fit()
    
    ### Get results
    #### Sample covariance matrix 
    cov = df_sec[res.mod_semopy.vars['observed']].cov()
    
    #### Tobit covariance/correlation matrix (polychoric/polyserial)
    poly_cov = pd.DataFrame(res.mod_semopy.mx_cov, index = res.mod_semopy.vars['observed'], columns = res.mod_semopy.vars['observed'])
    
    #### Model implied covariance matrix
    sigma = pd.DataFrame(res.mod_semopy.calc_sigma()[0], index = res.mod_semopy.vars['observed'], columns = res.mod_semopy.vars['observed'])
    
    #### Results table
    results = res.results
    
    #### Model evaluation statistics 
    #### NOTE: SRMR not correct with poly... covariances
    eval_stats = res.eval_stats
    
    #### Standardized residuals
    std_resid = pd.DataFrame(res.std_resid, index = res.mod_semopy.vars['observed'], columns = res.mod_semopy.vars['observed'])
    
    ### Save to a zip file
    with ZipFile('Results/{}.zip'.format(filename), 'w') as csv_zip:    
        csv_zip.writestr('cov.csv', data = cov.to_csv())
        csv_zip.writestr('poly_cov.csv', data = poly_cov.to_csv())
        csv_zip.writestr('sigma.csv', data = sigma.to_csv())
        csv_zip.writestr('results.csv', data = results.to_csv())
        csv_zip.writestr('eval_stats.csv', data = eval_stats.to_csv())
        csv_zip.writestr('std_resid.csv', data = std_resid.to_csv())
        
    ### Draw path diagram
    semplot(res.mod_semopy, r'Figures\CFA_path_diagrams\{}.PNG'.format(filename),  plot_covs = True, show = False)
    
# Loop over all formulas
## Waller/Muthen (1991) Generic Tobit Factor Analysis (GTFA) procedure
## TODO

## ADF procedure
formulas = [formula0, formula1, formula3]
filenames = ['formula0', 'formula1']

for formula, filename in zip(formulas,filenames):
    CFAWrapper(formula, 'WLS', filename, robust = False, solver = 'SLSQP')

res = CFA(formula0, df_log, obj = 'WLS', robust = False, solver = 'SLSQP').fit()
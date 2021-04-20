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
from semopy import Model, stats, semplot
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
df['cr_cd_net'] = df.cr_cd_purchased - df.cr_cd_sold

# Sort columns
df = df[df.columns.sort_values()]

# set variable names
# Variable names
vars_tot = [var for var in df.columns if var not in ['IDRSSD','date','cr_cd_purchased','cr_cd_sold']]

# Subset data to only include securitizers
#unique_idrssd = df[(df[vars_tot] > 0).any(axis = 1)].IDRSSD.unique()
#df_sec = df.loc[df.IDRSSD.isin(unique_idrssd),vars_tot]
df_sec = df.loc[(df[vars_tot] > 0).any(axis = 1),vars_tot]

# standardize data
df_standard = pd.DataFrame(preprocessing.scale(df_sec[vars_tot]), columns = vars_tot)

# Use power transformation to force normality on the data
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(standardize = False)
df_power = pd.DataFrame(pt.fit_transform(df_sec), columns = vars_tot)

# Check definiteness of the var-cov matrix
cov = df[vars_tot].cov()
definite_cov = np.all(np.linalg.eigvals(cov) > 0) # True

cov_std = df_standard[vars_tot].cov()
definite_cov_std = np.all(np.linalg.eigvals(cov_std) > 0) # True

#--------------------------------------------
# Setup factor analysis class funciton
#--------------------------------------------

class ModelEval:
    '''Helper class to CFA. Calculates the model evaluation statistics
    
    The following should be true for a good fit
        SB Chi2 pval > 0.05
        SRMR close to or below 0.08
        RMSEA close to or below 0.06
        CFI/TLI close to or greater than 0.95
        '''
    
    def Misc(self, model, dataframe):
        '''Calculates miscellaneous statistics '''    
        dof = stats.calc_dof(model)
    
        baseline_model = stats.get_baseline_model(model)
        baseline_res = baseline_model.fit(dataframe)
        dof_base = stats.calc_dof(baseline_model)
        
        return dof, dof_base, baseline_model
    
    def Chi2(self, model, dataframe):
        if self.bool_robust:
            chi_stat = (model.n_samples * model.last_result.fun) / (dataframe[model.vars['observed']].kurtosis().mean())
        else:
            chi_stat = model.n_samples * model.last_result.fun
        chi_p = 1 - chi2.cdf(chi_stat, self.dof)
        
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
        
        #### Calculate SRMR
        return s, sigma, resid_cov, resid_cov_std, np.sum(np.tril(resid_cov_std)**2) / (len(s) * (len(s) + 1) / 2)
    
    def GoodnessOfFit(self, model, dataframe, standardized_stats):
        ## SB scaled chi2
        chi_stat, chi_p = self.Chi2(model, dataframe)
        chi_base_stat, chi_base_p = self.Chi2(model, dataframe)
        
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
        pass
    # MI: test.mod_semopy.grad_mlw(test.mod_semopy.param_vals).T @ np.linalg.inv(test.mod_semopy.calc_fim()) @ test.mod_semopy.grad_mlw(test.mod_semopy.param_vals) # non robust
    
    def tableGoodnessOfFit(self, chi_stat, chi_p, chi_base_stat, chi_base_p, srmr, rmsea, cfi, tli, gfi, agfi, aic, bic):
        return pd.DataFrame([['chi_stat','chi_p','chi_base_stat', 'chi_base_p', 'srmr', 'rmsea','cfi','tli','gfi','agfi','aic','bic'],[chi_stat, chi_p, chi_base_stat, chi_base_p, srmr, rmsea, cfi, tli, gfi, agfi, aic, bic]]).T
    
    def EvalStats(self, model, dataframe, standardized_stats = False):
        '''Wrapper function to calculate 'the 'statistics'''
        # prelim
        if standardized_stats:
            self.dof_std, self.dof_std_base, self.mod_std_base_semopy = self.Misc(model, dataframe)
        else:   
            self.dof, self.dof_base, self.mod_base_semopy = self.Misc(model, dataframe)
        
        # calculate goodness of fit statistics
        chi_stat, chi_p, chi_base_stat, chi_base_p, srmr, rmsea, cfi, tli, gfi, agfi, aic, bic = self.GoodnessOfFit(model, dataframe, standardized_stats)
    
        ## Make pandas dataframe from the stats
        eval_stats = self.tableGoodnessOfFit(chi_stat, chi_p, chi_base_stat, chi_base_p, srmr, rmsea, cfi, tli, gfi, agfi, aic, bic)
        
        # Calculate Illness of fit statistics
        ## NOTE: standardized residuals are calculated in the SRMR function
        
        return chi_stat, chi_p, chi_base_stat, chi_base_p, srmr, rmsea, cfi, tli, gfi, agfi, aic, bic, eval_stats

class CFA(ModelEval):
    '''
    This class performs a confirmatory factor analysis based on the package
    semopy, and adds robust overall goodness-of-fit statistics and several
    localized area of ill fit statistics
    '''
    
    def __init__(self, equation, data, robust = True, standardize = False, mimic_lavaan = False, baseline = False):
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
        est = model.fit(dataframe)
    
        # Get results
        res = model.inspect(se_robust = self.bool_robust)
        
        return model, est, res
    
    def fit(self):
        ''' Fit model to data '''
        # Get model results
        self.mod_semopy, self.est_semopy, self.results = self.semopyFit(self.data)
        
        # Get model evaluation stats
        self.chi_stat, self.chi_p, self.chi_base_stat, self.chi_base_p, self.srmr, self.rmsea, self.cfi, self.tli, self.gfi, self.agfi, self.aic, self.bic, self.eval_stats = self.EvalStats(self.mod_semopy, self.data, False)
        
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
                    SEC =~ hmda_sec_amount + cr_as_sec + cr_serv_fees + cr_sec_income + cr_ce_sec + cr_ta_secveh + cr_cd_net + cr_ta_abcp
                    LS ~~ SEC''' 


# model_formula1 = '''ABS =~ hmda_sec_amount + cr_as_sec + cr_serv_fees + cr_sec_income + cr_ce_sec + cr_ta_secveh 
#                    CDO =~ cr_cd_sold + cr_cd_purchased + cr_serv_fees + cr_sec_income + cr_ce_sec + cr_ta_secveh 
#                    LS =~ hmda_gse_amount + hmda_priv_amount + cr_serv_fees + cr_ls_income + cr_as_nonsec'''

# model_formula2 = '''ABS =~ hmda_sec_amount + cr_as_sec
#                    SPV =~ cr_ta_abcp + cr_ta_secveh 
#                    OTH =~ cr_cd_sold + cr_cd_purchased + cr_serv_fees + cr_sec_income + cr_ce_sec
#                    LS =~ hmda_gse_amount + hmda_priv_amount + cr_serv_fees + cr_ls_income + cr_as_nonsec'''
                   
# model_formula3 = '''ABS =~ hmda_sec_amount + cr_as_sec + cr_serv_fees + cr_sec_income + cr_ce_sec + cr_ta_secveh 
#                    CDO =~ cr_cd_sold + cr_cd_purchased + cr_serv_fees + cr_sec_income + cr_ce_sec + cr_ta_secveh 
#                    ABCP =~ cr_ta_abcp + cr_serv_fees + cr_sec_income + cr_ce_sec
#                    LS =~ hmda_gse_amount + hmda_priv_amount + cr_serv_fees + cr_ls_income + cr_as_nonsec'''
                   
# model_formula4 = '''ABS =~ hmda_sec_amount + cr_as_sec + cr_serv_fees + cr_sec_income + cr_ce_sec + cr_ta_secveh 
#                    CDO =~ cr_cd_sold + cr_cd_purchased + cr_serv_fees + cr_sec_income + cr_ce_sec + cr_ta_secveh 
#                    LS =~ hmda_gse_amount + hmda_priv_amount + cr_serv_fees + cr_ls_income + cr_as_nonsec
                   
#                    GEN =~ ABS + CDO'''
                  
#--------------------------------------------
# Run CFA models
#--------------------------------------------
res = CFA(model_formula0, df, standardize = True).fit()

g = semplot(res.mod_semopy, r"D:\RUG\PhD\Materials_papers\01-Note_on_securitization\Figures\CFA_path_diagrams\test.PNG")

                  
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
df = pd.read_csv('Data\df_sec_note_OLD.csv', index_col = 0, dtype = np.float64)

# Add net credit derivatives
## Note: CD purchased and sold are highly colinear. For that reason we calculate
## a net CD variable: CD purchased - CD sold. Positive net CD means that the
## the bank has purchased more protection than it has sold
#df['cr_cd_net'] = df.cr_cd_purchased - df.cr_cd_sold

# Sort columns
df = df[df.columns.sort_values()]

# set variable names
# Variable names
vars_tot = [var for var in df.columns if var not in ['IDRSSD','date','cr_cd_purchased','cr_cd_sold', 'cr_cds_sold', 'cr_trs_sold', 'cr_co_sold', 'cr_cdoth_sold', 'cr_ta_vie_other', 'cr_as_sec', 'cr_ce_sec','ta']]
vars_tot_nols = [var for var in vars_tot if var not in ['hmda_gse_amount', 'hmda_priv_amount', 'cr_as_nonsec', 'cr_as_sbo']]

# Scale to total assets
#df = df[vars_tot].div(df.ta, axis = 0)

# Take logs of the data
df_log = np.log(df[vars_tot] - df[vars_tot].min() + 1) 

# Use power transformation to force normality on the data
#from sklearn.preprocessing import PowerTransformer
#pt = PowerTransformer(method = 'yeo-johnson', standardize = False)
#df_power = pd.DataFrame(pt.fit_transform(df[vars_tot] + 1), columns = vars_tot)

# Check definiteness of the var-cov matrix
cov_log = df_log[vars_tot].cov()
definite_cov_log = np.all(np.linalg.eigvals(cov_log) > 0) # True

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
    
    def Bootstrapwrapper(self, chi):
        '''Wrapper method for ADF Chi2 bootstrap'''
        # Run boostrap
        s, gamma = self.BootstrapChiADF()
        s = pd.DataFrame(s, index = self.mod_semopy.vars['observed'],\
                         columns = self.mod_semopy.vars['observed'])
            
        # Get new results (no need to run inspector)
        # NOTE: in May 2021 semopy cannot handle custom weighting matrices for 
        # its WLS procedure. Make sure you mod semopy such that it can handle them
        model = Model(self.equation, mimic_lavaan = self.mimic_lavaan, baseline = self.bool_baseline)
        est = model.fit(self.data, cov = s, obj = self.obj, solver = self.solver, custom_w = gamma)
        
        # Get Chi2_b, Corrected Chi2 and the chi2 bias
        chi_b = stats.calc_chi2(model)[0]
        chi_corr = 2 * chi - chi_b
        bias_chi2 = chi_b - chi
        
        return chi_b, chi_corr, bias_chi2
    
    def BootstrapChiADF(self):
        '''Method that implements the bootstrap procedure for the 
            ADF Chi2 statistic based on Yung/Bentler 1994.
            Not meant for the baseline Chi2.'''
        x = self.data
        variables = self.mod_semopy.vars['observed']
        s = list()
        B = self.B
        
        # Load packages and set seed
        from sklearn.utils import resample
        from semopy.utils import cov
        np.random.seed(0)
        
        for i in range(B):
            zero_bool = True
            
            while zero_bool:
                x_res = resample(x[variables], replace = True)
            
                # Check zero columns, is zero column resample
                if not (x_res == 0).all().any():
                    zero_bool = False
            x_res.reset_index(inplace = True, drop = True)
            
            # Get covariance matrix
            s.append(cov(x_res))
        
        # Get mean of the covariance matrix and vectorize the upper triangle
        s_mean = sum(s) / B
        s_mean_tri = s_mean[np.triu_indices(s_mean.shape[0])]
        
        # Get Gamma matrix
        # TODO Correct?
        s_smean = [mat[np.triu_indices(mat.shape[0])] - s_mean_tri for mat in s]
        gamma_mean = x.shape[0] * sum([np.einsum('i,j->ij', mat, mat) for mat in s_smean]) / (B-1)
        
        return s_mean, gamma_mean
    
    def Chi2(self, model, dof, dataframe):
        if self.bool_robust: 
            chi_stat = model.n_samples * model.last_result.fun
            #chi_stat = (model.n_samples * model.last_result.fun) / (1 + (model.n_samples * model.last_result.fun) / model.n_obs) # Yuan Bentler T2
            #TODO FIX, Klopt niet!
        elif self.bool_bootstrapchi: # ADF Chi2 correction a la Yung Bentler 1994
            chi_stat = model.n_samples * model.last_result.fun
            chi_b, chi_corr, bias_chi = self.Bootstrapwrapper(chi_stat)
            
            chi_p = 1 - chi2.cdf(chi_stat, dof)
            chi_corr_p = 1 - chi2.cdf(chi_corr, dof)
            
            return chi_stat, chi_p, chi_b, chi_corr, chi_corr_p, bias_chi
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
    
    def GoodnessOfFit(self, model, base_model, dataframe):
        ## chi2
        if self.bool_bootstrapchi:
            chi_orig_stat, chi_orig_p, chi_b, chi_stat, chi_p, bias_chi = self.Chi2(model, self.dof, dataframe)
            chi_base_stat, chi_base_p, _, _, _, _ = self.Chi2(base_model, self.dof_base, dataframe, )
        else:
            chi_stat, chi_p = self.Chi2(model, self.dof, dataframe)
            chi_base_stat, chi_base_p = self.Chi2(base_model, self.dof_base, dataframe)
        
        ## SRMR
        s, sigma, epsilon, epsilon_standard, srmr = self.SRMR()
        
        # TODO: Checken welke CHI2
        ## RMSEA
        rmsea = stats.calc_rmsea(model, chi_stat, self.dof)
        
        ## CFI/TFI
        cfi = stats.calc_cfi(model, self.dof, chi_stat,  self.dof_base, chi_base_stat)
        tli = stats.calc_tli(model, self.dof, chi_stat, self.dof_base, chi_base_stat)
        
        ## GFI/AGFI
        gfi = stats.calc_gfi(model, chi_stat, chi_base_stat)
        agfi = stats.calc_agfi(model, self.dof, self.dof_base, gfi)
    
        ## AIC/BIC
        aic = stats.calc_aic(model)
        bic = stats.calc_bic(model)
        
        if self.bool_bootstrapchi:
            return chi_orig_stat, chi_orig_p, chi_b, chi_stat, chi_p, bias_chi, chi_base_stat, chi_base_p, srmr, rmsea, cfi, tli, gfi, agfi, aic, bic
        else:
            return chi_stat, chi_p, chi_base_stat, chi_base_p, srmr, rmsea, cfi, tli, gfi, agfi, aic, bic
    
    def IllnessOfFit(self):
        # Standardized residuals
        _, _, _, std_resid, _ = self.SRMR() 

        return std_resid
    
    def tableGoodnessOfFit(self, chi_stat, chi_p, chi_base_stat, chi_base_p, srmr, rmsea, cfi, tli, gfi, agfi, aic, bic):
        return pd.DataFrame([['chi_stat','chi_p','chi_base_stat', 'chi_base_p', 'srmr', 'rmsea','cfi','tli','gfi','agfi','aic','bic'],[chi_stat, chi_p, chi_base_stat, chi_base_p, srmr, rmsea, cfi, tli, gfi, agfi, aic, bic]]).T
    
    def tableGoodnessOfFitBootstrap(self, chi_orig_stat, chi_orig_p, chi_b, chi_stat, chi_p, bias_chi, chi_base_stat, chi_base_p, srmr, rmsea, cfi, tli, gfi, agfi, aic, bic):
        return pd.DataFrame([['chi_orig_stat', 'chi_orig_p', 'chi_b', 'chi_stat', 'chi_p', 'bias_chi','chi_base_stat', 'chi_base_p', 'srmr', 'rmsea','cfi','tli','gfi','agfi','aic','bic'],[chi_orig_stat, chi_orig_p, chi_b, chi_stat, chi_p, bias_chi, chi_base_stat, chi_base_p, srmr, rmsea, cfi, tli, gfi, agfi, aic, bic]]).T
    
    def EvalStats(self, model, dataframe):
        '''Wrapper function to calculate 'the 'statistics'''
        # prelim
        self.dof, self.dof_base, self.mod_base_semopy = self.Misc(model, dataframe)
        base_model = self.mod_base_semopy
        
        ## Make pandas dataframe from the stats
        if self.bool_bootstrapchi:
            chi_orig_stat, chi_orig_p, chi_b, chi_stat, chi_p, bias_chi, chi_base_stat, chi_base_p, srmr, rmsea, cfi, tli, gfi, agfi, aic, bic = self.GoodnessOfFit(model, base_model, dataframe)
            eval_stats = self.tableGoodnessOfFitBootstrap(chi_orig_stat, chi_orig_p, chi_b, chi_stat, chi_p, bias_chi, chi_base_stat, chi_base_p, srmr, rmsea, cfi, tli, gfi, agfi, aic, bic)
            return chi_orig_stat, chi_orig_p, chi_b, chi_stat, chi_p, bias_chi, chi_base_stat, chi_base_p, srmr, rmsea, cfi, tli, gfi, agfi, aic, bic, eval_stats
        else:
            chi_stat, chi_p, chi_base_stat, chi_base_p, srmr, rmsea, cfi, tli, gfi, agfi, aic, bic = self.GoodnessOfFit(model, base_model, dataframe)
            eval_stats = self.tableGoodnessOfFit(chi_stat, chi_p, chi_base_stat, chi_base_p, srmr, rmsea, cfi, tli, gfi, agfi, aic, bic)
            return chi_stat, chi_p, chi_base_stat, chi_base_p, srmr, rmsea, cfi, tli, gfi, agfi, aic, bic, eval_stats

class CFA(ModelEval):
    '''
    This class performs a confirmatory factor analysis based on the package
    semopy, and adds robust overall goodness-of-fit statistics and several
    localized area of ill fit statistics
    '''
    
    def __init__(self, equation, data, robust = False, bootstrapchi = False, B = 1000, mimic_lavaan = False, baseline = False, obj = 'WLS', solver = 'L-BFGS-B'):
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
        self.bool_bootstrapchi = bootstrapchi
        self.B = B
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
        res = model.inspect(std_est = True, se_robust = self.bool_robust)
        
        return model, est, res
    
    def fit(self):
        ''' Fit model to data '''
        # Get model results
        self.mod_semopy, self.est_semopy, self.results = self.semopyFit(self.data)
        
        # Model implied variances
        self.sigma = pd.DataFrame(self.mod_semopy.calc_sigma()[0], index = self.mod_semopy.vars['observed'],\
                                  columns = self.mod_semopy.vars['observed'])
        
        # Get model evaluation stats
        if self.bool_bootstrapchi:
            self.chi_orig_stat, self.chi_orig_p, self.chi_b, self.chi_stat, self.chi_p, self.bias_chi, self.chi_base_stat, self.chi_base_p, self.srmr, self.rmsea, self.cfi, self.tli, self.gfi, self.agfi, self.aic, self.bic, self.eval_stats = self.EvalStats(self.mod_semopy, self.data)
        else:
            self.chi_stat, self.chi_p, self.chi_base_stat, self.chi_base_p, self.srmr, self.rmsea, self.cfi, self.tli, self.gfi, self.agfi, self.aic, self.bic, self.eval_stats = self.EvalStats(self.mod_semopy, self.data)
        
        # Illness of fit
        self.std_resid = self.IllnessOfFit()
            
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
    '''

# Main Model (no loan sales)
formula0a = '''LS =~ cr_as_nsres + cr_as_nsoth + hmda_gse_amount + hmda_priv_amount + cr_ls_income 
              ABS =~ cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_sec_income
              CDO =~ cr_cds_purchased + cr_trs_purchased + cr_co_purchased + cr_sec_income
              ABCP =~ cr_abcp_uc_own + cr_abcp_ce_own  + cr_sec_income
    
              hmda_gse_amount ~~ hmda_priv_amount
              hmda_sec_amount ~~ hmda_priv_amount
              hmda_gse_amount ~~ hmda_sec_amount
              hmda_gse_amount ~~ cr_as_nsres
              hmda_priv_amount ~~ cr_as_nsres
              hmda_sec_amount ~~ cr_as_rmbs
    
              LS ~~ ABS
              LS ~~ CDO
              ABS ~~ CDO
              ABS ~~ ABCP
              CDO ~~ ABCP
              
              DEFINE(latent) ABS CDO LS
            '''
            
formula0b = '''ABS =~ cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_ce_rmbs + cr_ce_abs + cr_secveh_ta 
              CDO =~ cr_cds_purchased + cr_trs_purchased + cr_co_purchased + cr_cdoth_purchased + cr_secveh_ta 
              ABCP =~ cr_abcp_ce + 1*cr_abcp_uc + cr_abcp_ta 
              GENSEC =~ ABS + CDO + ABCP 
              
              DEFINE(latent) ABS CDO ABCP GENSEC
            '''


# Main model with loan sales (to check the connection between ls and sec)
formula0a_ls = '''LS =~ cr_as_nonsec + hmda_gse_amount + hmda_priv_amount 
              ABS =~ cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_ce_rmbs + cr_ce_abs
              CDO =~ cr_cds_purchased + cr_trs_purchased + cr_co_purchased + cr_cdoth_purchased 
              ABCP =~ cr_abcp_ce + 1*cr_abcp_uc + cr_abcp_ta  
              
              DEFINE(latent) LS ABS CDO ABCP
              
              ABS, CDO ~~ ABCP
              ABS ~~ CDO
              LS ~~ 0*ABCP
              '''
            
formula0b_ls = '''LS =~ cr_as_nonsec + hmda_gse_amount + hmda_priv_amount 
              ABS =~ cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_ce_rmbs + cr_ce_abs 
              CDO =~ cr_cds_purchased + cr_trs_purchased + cr_co_purchased + cr_cdoth_purchased 
              ABCP =~ cr_abcp_ce + 1*cr_abcp_uc + cr_abcp_ta 
              GENSEC =~ ABS + CDO + ABCP 
              
              DEFINE(latent) LS ABS CDO ABCP GENSEC
              
              LS ~~ GENSEC
            '''

# Alternative Models
## 
formula1a = '''ABS =~ cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_ce_rmbs + cr_ce_abs + cr_secveh_ta 
              CDOABCP =~ cr_cds_purchased + cr_trs_purchased + cr_co_purchased + cr_cdoth_purchased + cr_abcp_ce + 1*cr_abcp_uc + cr_abcp_ta + cr_secveh_ta 
              
              DEFINE(latent) ABS CDOABCP
              
              ABS ~~ CDOABCP
            '''
            
formula1b = '''ABS =~ cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_ce_rmbs + cr_ce_abs + cr_secveh_ta 
              CDOABCP =~ cr_cds_purchased + cr_trs_purchased + cr_co_purchased + cr_cdoth_purchased + cr_abcp_ce + 1*cr_abcp_uc + cr_abcp_ta + cr_secveh_ta 
              GENSEC =~ ABS + CDOABCP 
              
              DEFINE(latent) ABS CDOABCP GENSEC
            '''
# NOTE 1b uses Cholesky: do not use
#--------------------------------------------
# Run CFA models
#--------------------------------------------

# Set procedure
def CFAWrapper(formula, obj, filename, robust = False, bootstrapchi = False, B = 1000, solver = 'SLSQP'):
    ### Fit model
    res = CFA(formula, df_log, obj = obj, robust = robust, bootstrapchi = bootstrapchi, B = B, solver = solver).fit()
    
    ### Get results
    #### Sample covariance matrix 
    cov = df_log[res.mod_semopy.vars['observed']].cov()
    
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
## ADF procedure
formulas = [formula0a, formula0b, formula0a_ls, formula0b_ls, formula1a]
filenames = ['formula0a', 'formula0b', 'formula0a_ls', 'formula0b_ls', 'formula1a']

for formula, filename in zip(formulas,filenames):
    CFAWrapper(formula, 'WLS', filename, robust = False, bootstrapchi = True, B = 1000, solver = 'SLSQP')

filenames_mlr = ['formula0a_mlr', 'formula0b_mlr', 'formula0a_ls_mlr', 'formula0b_ls_mlr', 'formula1a_mlr']
for formula, filename in zip(formulas,filenames_mlr):
    CFAWrapper(formula, 'MLW', filename, robust = True, bootstrapchi = False, solver = 'SLSQP')

#res = CFA(formula0a, df_log, obj = 'MLW', robust = True, bootstrapchi = False, B = 1000, solver = 'SLSQP').fit()
#CFAWrapper(formula0a, 'MLW', 'formula0a_ls_mlr', robust = True, bootstrapchi = False, B = 1000, solver = 'SLSQP')
# ---------------------------------------
# Confirmatory factor analysis for WP3a
# Mark van der Plaat
# September 2021 -- Update: December 2021
#----------------------------------------

# This script estimates the two-factor theory
# model with several robust estimators.
# These estimators are robust to non-normal data
# and some of them also to heteroskeasticity.

# We use the following estimators:
# WLS (ADF2), DLS, ULS
# ---------------------------------------
# Import packages and set wd
#----------------------------------------

# Set wd
setwd('D:/RUG/PhD/Materials_papers/01-Note_on_securitization')

# Import CFA package
library(lavaan)
library(semTools)

# ---------------------------------------
# import data
#----------------------------------------

# Import csvs
file_names <- c('Data/Data_weighted_bootstrap_110.csv',
                'Data/Data_weighted_bootstrap_120.csv',
                'Data/Data_weighted_bootstrap_130.csv',
                'Data/Data_weighted_bootstrap_140.csv',
                'Data/Data_weighted_bootstrap_150.csv',
                'Data/Data_weighted_bootstrap_160.csv',
                'Data/Data_weighted_bootstrap_170.csv',
                'Data/Data_weighted_bootstrap_180.csv',
                'Data/Data_weighted_bootstrap_190.csv',
                'Data/Data_weighted_bootstrap_200.csv')

# ---------------------------------------
# Loop over estimators
#----------------------------------------

# One factor model
#----------------------------------------
# NOTE: NoSBO performs better, so drop SBO from 1F
model_1f <-  '
# Measurement model
SEC =~ cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta  + cr_cds_purchased + cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
'

for (file in file_names){

  df_binary <- read.csv(file)

  fit_binary <- cfa(model_1f,
                    data = df_binary,
                    estimator = 'WLSMV',
                    ordered = TRUE)
  summary(fit_binary, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

  # #Save results
  # ## Parameter estimates
  # params_binary <- parameterEstimates(fit_binary, standardized = TRUE)
  # write.csv(params_binary, sprintf('Simulation_results/CFA_params_binary_1f_sim_%s.csv',substr(file, 30,32)))
  #
  # ## Fit measures
  # fitmeasures_binary <- fitMeasures(fit_binary, output = 'matrix')
  # write.csv(fitmeasures_binary, sprintf('Simulation_results/CFA_fitmeasures_binary_1f_sim_%s.csv',substr(file, 30,32)))
  #
  # ## R-square
  # r2_binary <- inspect(fit_binary, 'r2', output = 'matrix')
  # write.csv(r2_binary, sprintf('Simulation_results/CFA_r2_binary_1f_sim_%s.csv',substr(file, 30,32)))
  #
  # ## Residual standardized covariance matrix
  # resid_binary <- lavResiduals(fit_binary)$cov.z
  # write.csv(resid_binary, sprintf('Simulation_results/CFA_rescov_standard_binary_1f_sim_%s.csv',substr(file, 30,32)))
  #
  # ## Polychoric Matrix
  # pc_corr <- lavCor(fit_binary, ordered = TRUE, group = NULL, output = 'cor')
  # write.csv(pc_corr, sprintf('Simulation_results/CFA_polychoriccorr_binary_1f_sim_%s.csv',substr(file, 30,32)))
  #
  # ## Modindices
  # modin_binary <- modindices(fit_binary, sort = TRUE)
  # write.csv(modin_binary, sprintf('Simulation_results/CFA_modindices_binary_1f_sim_%s.csv',substr(file, 30,32)))
  #
  # ## Reliability
  # reli_binary <- reliability(fit_binary)
  # write.csv(reli_binary, sprintf('Simulation_results/CFA_reliability_binary_1f_%s.csv',substr(file, 30,32)))
}
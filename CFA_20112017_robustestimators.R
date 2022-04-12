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
library(blavaan)
library(semTools)

# ---------------------------------------
# import data
#----------------------------------------

# Import csv
## Continuous
file <- 'Data/df_sec_note_20112017.csv'
df <- read.csv(file)

# Binary
file <- 'Data/df_sec_note_binary_20112017.csv'
df_binary <- read.csv(file)

# Aggregated Binary data
file <- 'Data/df_sec_note_binary_agg_20112017.csv'
df_bagg <- read.csv(file)

# Log transform the data
df$cr_serv_fees <- df$cr_serv_fees - min(df$cr_serv_fees)
df$cr_ls_income <- df$cr_ls_income - min(df$cr_ls_income)
df$cr_sec_income <- df$cr_sec_income - min(df$cr_sec_income)

# df$cr_serv_fees <- abs(df$cr_serv_fees)
# df$cr_ls_income <- abs(df$cr_ls_income)
# df$cr_sec_income <- abs(df$cr_sec_income)

df_log <- log(df + 1)

# ---------------------------------------
# Loop over estimators for continuous data
#----------------------------------------

# Set model equation
model_theory <-  '
# Measurement model
ABSCDO =~ NA*cr_as_sbo + 1*cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta  + cr_cds_purchased
ABCP =~ cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_cds_purchased

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
'

# Set list of estimators
# Note: we do not use ML with bootstrap, because BS-ML cannot handle extreme non-normality well
lst_estimators <- c('WLS','DWLS','DLS','ULS')

# Loop over estimators
for (estimator in lst_estimators){
  fit_theory <- cfa(model_theory, data = df_log, estimator = estimator)
  # summary(fit_theory, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

  #Save results
  ## Parameter estimates
  params_theory <- parameterEstimates(fit_theory, standardized = TRUE)
  write.csv(params_theory, sprintf('Results/CFA_params_theory_%s.csv',estimator))

  ## Fit measures
  fitmeasures_theory <- fitMeasures(fit_theory, output = 'matrix')
  write.csv(fitmeasures_theory, sprintf('Results/CFA_fitmeasures_theory_%s.csv',estimator))

  ## R-square
  r2_theory <- inspect(fit_theory, 'r2', output = 'matrix')
  write.csv(r2_theory, sprintf('Results/CFA_r2_theory_%s.csv',estimator))
}

# ---------------------------------------
# Bayesian CFA, Continuous data
#----------------------------------------

# Fit the model with a bayesian algorithm.
fit_bayes <- bcfa(model_theory,
                  data = df_log)
# summary(fit_bayes, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

#Save results
## Parameter estimates
params_bayes <- parameterEstimates(fit_bayes, standardized = TRUE, se = TRUE) # NOTE slightly different table layout than lavaan
write.csv(params_bayes, 'Results/CFA_params_bayes.csv')

## Fit measures
fitmeasures_bayes <- fitMeasures(fit_bayes, output = 'matrix')
write.csv(fitmeasures_bayes,'Results/CFA_fitmeasures_bayes.csv')

## R-square
r2_bayes <- inspect(fit_bayes, 'r2', output = 'matrix')
write.csv(r2_bayes, 'Results/CFA_r2_bayes.csv')

# ---------------------------------------
# Binary data
#----------------------------------------

# Set list of estimators
# Note: we do not use ML with bootstrap, because BS-ML cannot handle extreme non-normality well
lst_estimators <- c('WLSMV','DWLS')

# Loop over estimators
for (estimator in lst_estimators){
  fit_binary <- cfa(model_theory,
                    data = df_binary,
                    estimator = estimator,
                    ordered = TRUE)
  # summary(fit_binary, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

  #Save results
  ## Parameter estimates
  params_binary <- parameterEstimates(fit_binary, standardized = TRUE)
  write.csv(params_binary, sprintf('Results/CFA_params_binary_%s.csv',estimator))

  ## Fit measures
  fitmeasures_binary <- fitMeasures(fit_binary, output = 'matrix')
  write.csv(fitmeasures_binary, sprintf('Results/CFA_fitmeasures_binary_%s.csv',estimator))

  ## R-square
  r2_binary <- inspect(fit_binary, 'r2', output = 'matrix')
  write.csv(r2_binary, sprintf('Results/CFA_r2_binary_%s.csv',estimator))

  ## Residual standardized covariance matrix
  resid_binary <- lavResiduals(fit_binary)$cov.z
  write.csv(resid_binary, sprintf('Results/CFA_rescov_standard_binary_%s.csv',estimator))

  ## Modindices
  modin_binary <- modindices(fit_binary, sort = TRUE)
  write.csv(modin_binary, sprintf('Results/CFA_modindices_binary_%s.csv',estimator))
}

# One factor model
#----------------------------------------
model_1f <-  '
# Measurement model
F1 =~ NA*cr_as_sbo + 1*cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta  + cr_cds_purchased + cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
'

for (estimator in lst_estimators){
  fit_binary <- cfa(model_1f,
                    data = df_binary,
                    estimator = estimator,
                    ordered = TRUE)
  # summary(fit_binary, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

  #Save results
  ## Parameter estimates
  params_binary <- parameterEstimates(fit_binary, standardized = TRUE)
  write.csv(params_binary, sprintf('Results/CFA_params_binary_1f_%s.csv',estimator))

  ## Fit measures
  fitmeasures_binary <- fitMeasures(fit_binary, output = 'matrix')
  write.csv(fitmeasures_binary, sprintf('Results/CFA_fitmeasures_binary_1f_%s.csv',estimator))

  ## R-square
  r2_binary <- inspect(fit_binary, 'r2', output = 'matrix')
  write.csv(r2_binary, sprintf('Results/CFA_r2_binary_1f_%s.csv',estimator))

  ## Residual standardized covariance matrix
  resid_binary <- lavResiduals(fit_binary)$cov.z
  write.csv(resid_binary, sprintf('Results/CFA_rescov_standard_binary_1f_%s.csv',estimator))

  ## Modindices
  modin_binary <- modindices(fit_binary, sort = TRUE)
  write.csv(modin_binary, sprintf('Results/CFA_modindices_binary_1f_%s.csv',estimator))
}

# Aggregated data
#----------------------------------------

# two-factor model
# Loop over estimators
for (estimator in lst_estimators){
  fit_binary <- cfa(model_theory,
                    data = df_bagg,
                    estimator = estimator,
                    ordered = TRUE)
  # summary(fit_binary, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

  #Save results
  ## Parameter estimates
  params_binary <- parameterEstimates(fit_binary, standardized = TRUE)
  write.csv(params_binary, sprintf('Results/CFA_params_binary_agg_%s.csv',estimator))

  ## Fit measures
  fitmeasures_binary <- fitMeasures(fit_binary, output = 'matrix')
  write.csv(fitmeasures_binary, sprintf('Results/CFA_fitmeasures_binary_agg_%s.csv',estimator))

  ## R-square
  r2_binary <- inspect(fit_binary, 'r2', output = 'matrix')
  write.csv(r2_binary, sprintf('Results/CFA_r2_binary_agg_%s.csv',estimator))

  ## Residual standardized covariance matrix
  resid_binary <- lavResiduals(fit_binary)$cov.z
  write.csv(resid_binary, sprintf('Results/CFA_rescov_standard_binary_agg_%s.csv',estimator))

  ## Modindices
  modin_binary <- modindices(fit_binary, sort = TRUE)
  write.csv(modin_binary, sprintf('Results/CFA_modindices_binary_agg_%s.csv',estimator))
}

# two-factor model
# Loop over estimators
for (estimator in lst_estimators){
  fit_binary <- cfa(model_1f,
                    data = df_bagg,
                    estimator = estimator,
                    ordered = TRUE)
  # summary(fit_binary, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

  #Save results
  ## Parameter estimates
  params_binary <- parameterEstimates(fit_binary, standardized = TRUE)
  write.csv(params_binary, sprintf('Results/CFA_params_binary_agg_1f_%s.csv',estimator))

  ## Fit measures
  fitmeasures_binary <- fitMeasures(fit_binary, output = 'matrix')
  write.csv(fitmeasures_binary, sprintf('Results/CFA_fitmeasures_binary_agg_1f_%s.csv',estimator))

  ## R-square
  r2_binary <- inspect(fit_binary, 'r2', output = 'matrix')
  write.csv(r2_binary, sprintf('Results/CFA_r2_binary_agg_1f_%s.csv',estimator))

  ## Residual standardized covariance matrix
  resid_binary <- lavResiduals(fit_binary)$cov.z
  write.csv(resid_binary, sprintf('Results/CFA_rescov_standard_binary_agg_1f_%s.csv',estimator))

  ## Modindices
  modin_binary <- modindices(fit_binary, sort = TRUE)
  write.csv(modin_binary, sprintf('Results/CFA_modindices_binary_agg_1f_%s.csv',estimator))
}
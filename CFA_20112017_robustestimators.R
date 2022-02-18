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

# Import csv
file <- 'Data/df_sec_note_20112017.csv'
df <- read.csv(file)

# Log transform the data
df$cr_serv_fees <- df$cr_serv_fees - min(df$cr_serv_fees)
df$cr_ls_income <- df$cr_ls_income - min(df$cr_ls_income)
df$cr_sec_income <- df$cr_sec_income - min(df$cr_sec_income)

df_log <- log(df + 1)

# ---------------------------------------
# Loop over estimators
#----------------------------------------

# Set model equation
model_theory <-  '
# Measurement model
ABSCDO =~ NA*cr_as_sbo + 1*cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta + cr_sec_income + cr_serv_fees+ cr_cds_purchased
ABCP =~ cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_sec_income + cr_serv_fees + cr_cds_purchased

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
'

# Set list of estimators
# lst_estimators <- c('WLS','DLS','ULS')
lst_estimators <- c('DLS','ULS')

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










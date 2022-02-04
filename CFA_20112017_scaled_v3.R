# ---------------------------------------
# Confirmatory factor analysis for WP3a
# Mark van der Plaat
# September 2021 -- Update: December 2021
#----------------------------------------

# This scrupt performs a confirmatory factor 
# analysis. All other scrips are in Python.
# Given that Python has no satisfactory CFA
# package, we use R. 

# We use the procedure as described in:
# Brown, T. A. (2015). Confirmatory
# Factor Analysis for Applied Research. In T. D. Little (Ed.), 
# (2nd ed., Vol. 53). New York: The Guilford Press.

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

df_scaled <- df / df$ta * 100

df_scaled$cr_serv_fees <- df_scaled$cr_serv_fees - min(df_scaled$cr_serv_fees)
df_scaled$cr_ls_income <- df_scaled$cr_ls_income - min(df_scaled$cr_ls_income)
df_scaled$cr_sec_income <- df_scaled$cr_sec_income - min(df_scaled$cr_sec_income)

df_log <- log(df_scaled + 1)

# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------
# One securitization factor model
#----------------------------------------
model_onesec <-  '
#Measurement model
SEC =~ NA*cr_as_sbo + 1*cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta + cr_sec_income + cr_serv_fees + cr_cds_purchased  + cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own '


fit_onesec <- cfa(model_onesec, data = df_log, estimator = 'MLR')
summary(fit_onesec, fit.measures=TRUE, standardized = TRUE)

#Save results
## Parameter estimates
params_onesec <- parameterEstimates(fit_onesec, standardized = TRUE)
write.csv(params_onesec, 'Results/CFA_params_onesec_scaled.csv')

## Model implied covariance matrix
#fitted_onesec <- fitted(fit_onesec)$cov
#write.csv(fitted_onesec, 'Results/CFA_modimplied_cov_onesec.csv')

## Residual covariance matrix
#resid_onesec <- lavResiduals(fit_onesec)$cov
#resid_std_onesec <- lavResiduals(fit_onesec)$cov.z
#write.csv(resid_onesec, 'Results/CFA_rescov_onesec.csv')
#write.csv(resid_std_onesec, 'Results/CFA_rescov_standard_onesec.csv')

## Fit measures
fitmeasures_onesec <- fitMeasures(fit_onesec, output = 'matrix')
write.csv(fitmeasures_onesec, 'Results/CFA_fitmeasures_onesec_scaled.csv')

## Modindices
#modin_onesec <- modindices(fit_onesec, sort = TRUE, maximum.number = 25)
#write.csv(modin_onesec, 'Results/CFA_modindices_onesec.csv')

## R-square
#r2_onesec <- inspect(fit_onesec, 'r2', output = 'matrix')
#write.csv(r2_onesec, 'Results/CFA_r2_onesec.csv')

## Reliability
#reli_onesec <- reliability(fit_onesec)
#write.csv(reli_onesec, 'Results/CFA_reliability_onesec.csv')

# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------
# EFA model
#----------------------------------------
model_efa <-  '
# Measurement model
ABSCDO =~ cr_as_rmbs + cr_as_abs + cr_sec_income + cr_serv_fees
ABCP =~ NA*cr_secveh_ta + 1*cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_cds_purchased

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
'

fit_efa <- cfa(model_efa, data = df_log, estimator = 'MLR')
summary(fit_efa, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

#Save results
## Parameter estimates
params_efa <- parameterEstimates(fit_efa, standardized = TRUE)
write.csv(params_efa, 'Results/CFA_params_efa_scaled.csv')

## Model implied covariance matrix
#fitted_efa <- fitted(fit_efa)$cov
#write.csv(fitted_efa, 'Results/CFA_modimplied_cov_efa.csv')

## Residual covariance matrix
#resid_efa <- lavResiduals(fit_efa)$cov
#resid_std_efa <- lavResiduals(fit_efa)$cov.z
#write.csv(resid_efa, 'Results/CFA_rescov_efa.csv')
#write.csv(resid_std_efa, 'Results/CFA_rescov_standard_efa.csv')

## Fit measures
fitmeasures_efa <- fitMeasures(fit_efa, output = 'matrix')
write.csv(fitmeasures_efa, 'Results/CFA_fitmeasures_efa_scaled.csv')

## Modindices
#odin_efa <- modindices(fit_efa, sort = TRUE, maximum.number = 25)
#write.csv(modin_efa, 'Results/CFA_modindices_efa.csv')

## R-square
#r2_efa <- inspect(fit_efa, 'r2', output = 'matrix')
#write.csv(r2_efa, 'Results/CFA_r2_efa.csv')

## Reliability
#reli_efa <- reliability(fit_efa)
#write.csv(reli_efa, 'Results/CFA_reliability_efa.csv')

# ---------------------------------------
# Theory model
#----------------------------------------

model_theory <-  '
# Measurement model
ABSCDO =~ NA*cr_as_sbo + 1*cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta + cr_sec_income + cr_serv_fees+ cr_cds_purchased
ABCP =~ cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_sec_income + cr_serv_fees+ cr_cds_purchased

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
'

fit_theory <- cfa(model_theory, data = df_log, estimator = 'MLR')
summary(fit_theory, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

#Save results
## Parameter estimates
params_theory <- parameterEstimates(fit_theory, standardized = TRUE)
write.csv(params_theory, 'Results/CFA_params_theory_scaled.csv')

## Model implied covariance matrix
#fitted_theory <- fitted(fit_theory)$cov
#write.csv(fitted_theory, 'Results/CFA_modimplied_cov_theory.csv')

## Residual covariance matrix
#resid_theory <- lavResiduals(fit_theory)$cov
#resid_std_theory <- lavResiduals(fit_theory)$cov.z
#write.csv(resid_theory, 'Results/CFA_rescov_theory.csv')
#write.csv(resid_std_theory, 'Results/CFA_rescov_standard_theory.csv')

## Fit measures
fitmeasures_theory <- fitMeasures(fit_theory, output = 'matrix')
write.csv(fitmeasures_theory, 'Results/CFA_fitmeasures_theory_scaled.csv')

## Modindices
#modin_theory <- modindices(fit_theory, sort = TRUE, maximum.number = 25)
#write.csv(modin_theory, 'Results/CFA_modindices_theory.csv')

## R-square
r2_theory <- inspect(fit_theory, 'r2', output = 'matrix')
write.csv(r2_theory, 'Results/CFA_r2_theory_scaled.csv')

## Reliability
#reli_theory <- reliability(fit_theory)
#write.csv(reli_theory, 'Results/CFA_reliability_theory.csv')

# ---------------------------------------
# Combined
#----------------------------------------

model_combined <-  '
# Measurement model
ABSCDO =~ NA*cr_as_sbo + 1*cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta + cr_sec_income + cr_serv_fees
ABCP =~ NA*cr_secveh_ta + 1*cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_cds_purchased

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
'

fit_combined <- cfa(model_combined, data = df_log, estimator = 'MLR')
summary(fit_combined, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

#Save results
## Parameter estimates
params_combined <- parameterEstimates(fit_combined, standardized = TRUE)
write.csv(params_combined, 'Results/CFA_params_combined_scaled.csv')

## Model implied covariance matrix
#fitted_combined <- fitted(fit_combined)$cov
#write.csv(fitted_combined, 'Results/CFA_modimplied_cov_combined.csv')

## Residual covariance matrix
#resid_combined <- lavResiduals(fit_combined)$cov
#resid_std_combined <- lavResiduals(fit_combined)$cov.z
#write.csv(resid_combined, 'Results/CFA_rescov_combined.csv')
#write.csv(resid_std_combined, 'Results/CFA_rescov_standard_combined.csv')

## Fit measures
fitmeasures_combined <- fitMeasures(fit_combined, output = 'matrix')
write.csv(fitmeasures_combined, 'Results/CFA_fitmeasures_combined_scaled.csv')

## Modindices
#modin_combined <- modindices(fit_combined, sort = TRUE, maximum.number = 25)
#write.csv(modin_combined, 'Results/CFA_modindices_combined.csv')

## R-square
#r2_combined <- inspect(fit_combined, 'r2', output = 'matrix')
#write.csv(r2_combined, 'Results/CFA_r2_combined.csv')

## Reliability
#reli_combined <- reliability(fit_combined)
#write.csv(reli_combined, 'Results/CFA_reliability_combined.csv')

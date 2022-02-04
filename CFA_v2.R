# ---------------------------------------
# Confirmatory factor analysis for WP3a
# Mark van der Plaat
# September 2021
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

# ---------------------------------------
# import data
#----------------------------------------

# Import csv
#file <- 'Data/df_sec_note_20112017.csv'
file <- 'Data/df_sec_note.csv'
df <- read.csv(file)

df$cr_serv_fees <- df$cr_serv_fees - min(df$cr_serv_fees)
df$cr_ls_income <- df$cr_ls_income - min(df$cr_ls_income)
df$cr_sec_income <- df$cr_sec_income - min(df$cr_sec_income)
df$cr_secveh_ta <- df$cr_secveh_ta - min(df$cr_secveh_ta)

mins <- sapply(df, min)

df_log <- log(df + 1)

# ---------------------------------------
# Improved model
#----------------------------------------
model_improved <-  'ABSCDO =~ cr_as_rmbs + cr_as_ccr + cr_as_auto + cr_as_abs_oth + hmda_sec_amount + cr_sec_income + cr_cds_purchased
                    ABCP =~ cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_sec_income + cr_cds_purchased

                    hmda_sec_amount ~~ cr_as_rmbs
                    cr_abcp_uc_own ~~ cr_abcp_ce_own'

fit_impr <- cfa(model_improved, data = df_log, estimator = 'MLR')
summary(fit_impr, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

#Save results
## Parameter estimates
params_impr <- parameterEstimates(fit_impr, standardized = TRUE)
write.csv(params_impr, 'Results/CFA_params_impr.csv')

## Model implied covariance matrix
fitted_impr <- fitted(fit_impr)$cov
write.csv(fitted_impr, 'Results/CFA_modimplied_cov_impr.csv')

## Residual covariance matrix
resid_impr <- lavResiduals(fit_impr)$cov
resid_std_impr <- lavResiduals(fit_impr)$cov.z
write.csv(resid_impr, 'Results/CFA_rescov_impr.csv')
write.csv(resid_std_impr, 'Results/CFA_rescov_standard_impr.csv')

## Fit measures
fitmeasures_impr <- fitMeasures(fit_impr, output = 'matrix')
write.csv(fitmeasures_impr, 'Results/CFA_fitmeasures_impr.csv')

## Modindices
modin_impr <- modindices(fit_impr, sort = TRUE, maximum.number = 25)
write.csv(modin_impr, 'Results/CFA_modindices_impr.csv')

## R-square
r2_impr <- inspect(fit_impr, 'r2', output = 'matrix')
write.csv(r2_impr, 'Results/CFA_r2_impr.csv')

# ---------------------------------------
# One sec factor 
#----------------------------------------
model_onesec <-  'SEC =~ cr_as_rmbs + cr_as_ccr + cr_as_auto + cr_as_abs_oth +  hmda_sec_amount + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_sec_income + cr_cds_purchased

                    hmda_sec_amount ~~ cr_as_rmbs
                    cr_abcp_uc_own ~~ cr_abcp_ce_own'


fit_onesec <- cfa(model_onesec, data = df_log, estimator = 'MLR')
summary(fit_onesec, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

#Save results
## Parameter estimates
params_onesec <- parameterEstimates(fit_onesec, standardized = TRUE)
write.csv(params_onesec, 'Results/CFA_params_onesec.csv')

## Model implied covariance matrix
fitted_onesec <- fitted(fit_onesec)$cov
write.csv(fitted_onesec, 'Results/CFA_modimplied_cov_onesec.csv')

## Residual covariance matrix
resid_onesec <- lavResiduals(fit_onesec)$cov
resid_std_onesec <- lavResiduals(fit_onesec)$cov.z
write.csv(resid_onesec, 'Results/CFA_rescov_onesec.csv')
write.csv(resid_std_onesec, 'Results/CFA_rescov_standard_onesec.csv')

## Fit measures
fitmeasures_onesec <- fitMeasures(fit_onesec, output = 'matrix')
write.csv(fitmeasures_onesec, 'Results/CFA_fitmeasures_onesec.csv')

## Modindices
modin_onesec <- modindices(fit_onesec, sort = TRUE, maximum.number = 25)
write.csv(modin_onesec, 'Results/CFA_modindices_onesec.csv')

## R-square
r2_onesec <- inspect(fit_onesec, 'r2', output = 'matrix')
write.csv(r2_onesec, 'Results/CFA_r2_onesec.csv')




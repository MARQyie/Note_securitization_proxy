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
# Setup factor model equations
#----------------------------------------
# Original model, not nested
model_or_nonest <-  'LS =~ cr_as_nsres + cr_as_nsoth + hmda_gse_amount + hmda_priv_amount + cr_ls_income 
                    ABS =~ cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_sec_income 
                    CDO =~ cr_cds_purchased + cr_trs_purchased + cr_co_purchased + cr_sec_income 
                    ABCP =~ cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_sec_income 
          
                    hmda_gse_amount ~~ hmda_priv_amount + hmda_sec_amount
                    hmda_sec_amount ~~ hmda_priv_amount + cr_as_rmbs
                    cr_as_nsres ~~ hmda_gse_amount + hmda_priv_amount
                    cr_abcp_uc_own ~~ cr_abcp_ce_own' # All factors are allowed to correlate

# Original model, nested
model_or_nest <-  'LS =~ cr_as_nsres + cr_as_nsoth + hmda_gse_amount + hmda_priv_amount + cr_ls_income
                    ABS =~ cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_as_sbo + cr_sec_income
                    CDO =~ cr_cds_purchased + cr_trs_purchased + cr_co_purchased + cr_sec_income
                    ABCP =~ cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_sec_income
                    SEC =~ ABS + CDO + ABCP
          
                    hmda_gse_amount ~~ hmda_priv_amount + hmda_sec_amount
                    hmda_sec_amount ~~ hmda_priv_amount + cr_as_rmbs
                    cr_as_nsres ~~ hmda_gse_amount + hmda_priv_amount
                    cr_abcp_uc_own ~~ cr_abcp_ce_own
          
                    LS ~~ SEC'

# ---------------------------------------
# Estimate factor model
#----------------------------------------
# Original model, not nested
## Fit the model
fit_nonest <- cfa(model_or_nonest, data = df_log, estimator = 'MLR')
#fit_nonest <- cfa(model_or_nonest, data = df_log, estimator = 'MLR', std.lv = TRUE) # TURN ON IF YOU WANT TO SET THE FACTOR VARIANCE TO 1 

## Summary of the fitted model
summary(fit_nonest, fit.measures=TRUE, standardized = TRUE)

## Fit measures
fitMeasures(fit_nonest)
# NOTE ABOUT FIT INDICES (USE THE ROBUST WHERE POSSIBLE)
# https://easystats.github.io/performance/reference/model_performance.lavaan.html
# GOOD: GFI (0.950), AGFI (0.924), RSMEA (0.061), SRMR (0.033), PNFI (0.672; not robust), IFI (0.916; not robust)
# ACCEPTABLE: CFI (0.928), TLI (0.901), NFI (0.913; not robust)
# BAD: Chi2 (497.796, p = 0.000)
# CONCLUSION: The fit of the model is rather good. 6 are good and 3 are acceptable. Only Chi2 is bad (as expected).
# No need to look for improvements to the model 

# Original model, nested
## Fit the model
fit_nest <- cfa(model_or_nest, data = df_log, estimator = 'MLR')

## Summary of the fitted model
summary(fit_nest, fit.measures=TRUE, standardized = TRUE)

## Fit measures
fitMeasures(fit_nest)
# NOTE ABOUT FIT INDICES (USE THE ROBUST WHERE POSSIBLE)
# https://easystats.github.io/performance/reference/model_performance.lavaan.html
# GOOD: RSMEA (0.069) SRMR (0.041), PNFI (0.668; not robust), AGFI (0.902)
# ACCEPTABLE: CFI (0.904)
# BAD: Chi2 (645.700, p = 0.000; WORSE THAN THE NOT NESTED MODEL), GFI (0.935), NFI (0.891; not robust), TLI (0.872), IFI (0.894; not robust) 
# CONCLUSION: The fit of the nested model is poorer than the non-nested model. The overall fit is acceptable, but not good enough. 
# Do not continue with the nested model

# ---------------------------------------
# save all results to csv
#----------------------------------------
# Original model, not nested
## Parameter estimates
params_nonest <- parameterEstimates(fit_nonest, standardized = TRUE)
write.csv(params_nonest, 'Results/CFA_params_nonest.csv')

## Model implied covariance matrix
fitted_nonest <- fitted(fit_nonest)$cov
write.csv(fitted_nonest, 'Results/CFA_modimplied_cov_nonest.csv')

## Residual covariance matrix
resid_nonest <- lavResiduals(fit_nonest)$cov
resid_std_nonest <- lavResiduals(fit_nonest)$cov.z
write.csv(resid_nonest, 'Results/CFA_rescov_nonest.csv')
write.csv(resid_std_nonest, 'Results/CFA_rescov_standard_nonest.csv')

## Fit measures
fitmeasures_nonest <- fitMeasures(fit_nonest, output = 'matrix')
write.csv(fitmeasures_nonest, 'Results/CFA_fitmeasures_nonest.csv')

## Modindices
modin_nonest <- modindices(fit_nonest, sort = TRUE, maximum.number = 25)
write.csv(modin_nonest, 'Results/CFA_modindices_nonest.csv')

#----------------------------------------
# Original model, nested
## Parameter estimates
params_nest <- parameterEstimates(fit_nest, standardized = TRUE)
write.csv(params_nest, 'Results/CFA_params_nest.csv')

## Model implied covariance matrix
fitted_nest <- fitted(fit_nest)$cov
write.csv(fitted_nest, 'Results/CFA_modimplied_cov_nest.csv')

## Residual covariance matrix
resid_nest <- lavResiduals(fit_nest)$cov
resid_std_nest <- lavResiduals(fit_nest)$cov.z
write.csv(resid_nest, 'Results/CFA_rescov_nest.csv')
write.csv(resid_std_nest, 'Results/CFA_rescov_standard_nest.csv')

## Fit measures
fitmeasures_nest <- fitMeasures(fit_nest, output = 'matrix')
write.csv(fitmeasures_nest, 'Results/CFA_fitmeasures_nest.csv')

## Modindices
modin_nest <- modindices(fit_nest, sort = TRUE, maximum.number = 25)
write.csv(modin_nest, 'Results/CFA_modindices_nest.csv')

# ---------------------------------------
# Improved model
#----------------------------------------
model_improved <-  'LS =~ cr_as_nsres + cr_as_nsoth + hmda_gse_amount + hmda_priv_amount + cr_ls_income
                    ABSCDO =~ cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_sec_income + cr_cds_purchased
                    CD =~ cr_cds_purchased + cr_trs_purchased + cr_co_purchased
                    ABCP =~ cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_sec_income + cr_cds_purchased
                     
                    hmda_gse_amount ~~ hmda_priv_amount + hmda_sec_amount
                    hmda_sec_amount ~~ hmda_priv_amount + cr_as_rmbs
                    cr_as_nsres ~~ hmda_gse_amount + hmda_priv_amount
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
# Improved model, with nesting
#----------------------------------------
model_improved_nest <-  'LS =~ cr_as_nsres + cr_as_nsoth + hmda_gse_amount + hmda_priv_amount + cr_ls_income 
                          ABSCDO =~ cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_sec_income + cr_cds_purchased
                          CD =~ cr_cds_purchased + cr_trs_purchased + cr_co_purchased
                          ABCP =~ cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_sec_income + cr_cds_purchased
                          SEC =~ ABSCDO + ABCP
                           
                          hmda_gse_amount ~~ hmda_priv_amount + hmda_sec_amount
                          hmda_sec_amount ~~ hmda_priv_amount + cr_as_rmbs
                          cr_as_nsres ~~ hmda_gse_amount + hmda_priv_amount
                          cr_abcp_uc_own ~~ cr_abcp_ce_own'


fit_impr_nest <- cfa(model_improved_nest, data = df_log, estimator = 'MLR')
summary(fit_impr_nest, fit.measures=TRUE, standardized = TRUE)

## Parameter estimates
params_impr_nest <- parameterEstimates(fit_impr_nest, standardized = TRUE)
write.csv(params_impr_nest, 'Results/CFA_params_impr_nest.csv')

## Model implied covariance matrix
fitted_impr_nest <- fitted(fit_impr_nest)$cov
write.csv(fitted_impr_nest, 'Results/CFA_modimplied_cov_impr_nest.csv')

## Residual covariance matrix
resid_impr_nest <- lavResiduals(fit_impr_nest)$cov
resid_std_impr_nest <- lavResiduals(fit_impr_nest)$cov.z
write.csv(resid_impr_nest, 'Results/CFA_rescov_impr_nest.csv')
write.csv(resid_std_impr_nest, 'Results/CFA_rescov_standard_impr_nest.csv')

## Fit measures
fitmeasures_impr_nest <- fitMeasures(fit_impr_nest, output = 'matrix')
write.csv(fitmeasures_impr_nest, 'Results/CFA_fitmeasures_impr_nest.csv')

## Modindices
modin_impr_nest <- modindices(fit_impr_nest, sort = TRUE, maximum.number = 25)
write.csv(modin_impr_nest, 'Results/CFA_modindices_impr_nest.csv')

# ---------------------------------------
# One sec factor 
#----------------------------------------
model_onesec <-  'LS =~ cr_as_nsres + cr_as_nsoth + hmda_gse_amount + hmda_priv_amount + cr_ls_income 
                  SEC =~ cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_sec_income + cr_cds_purchased
                  CD =~ cr_cds_purchased + cr_trs_purchased + cr_co_purchased
                  
                  hmda_gse_amount ~~ hmda_priv_amount + hmda_sec_amount
                  hmda_sec_amount ~~ hmda_priv_amount + cr_as_rmbs
                  cr_as_nsres ~~ hmda_gse_amount + hmda_priv_amount
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




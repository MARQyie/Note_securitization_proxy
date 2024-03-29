# ---------------------------------------
# Confirmatory factor analysis for WP3
# Balanced Data Only
# Mark van der Plaat
# February 2022
#----------------------------------------

# ---------------------------------------
# Import packages and set wd
#----------------------------------------

# Set wd
setwd('D:/RUG/PhD/Materials_papers/01-Note_on_securitization')

# Import CFA package
library(lavaan)
library(semTools)
library(MBESS)

# ---------------------------------------
# Import and log-transform data
#----------------------------------------

# Read files
file <- 'Data/df_sec_note_20112017_agg.csv'
df_agg <- read.csv(file)

# Log-transform
df_agg$cr_serv_fees <- df_agg$cr_serv_fees - min(df_agg$cr_serv_fees)
df_agg$cr_ls_income <- df_agg$cr_ls_income - min(df_agg$cr_ls_income)
df_agg$cr_sec_income <- df_agg$cr_sec_income - min(df_agg$cr_sec_income)

df_agg_log <- log(df_agg + 1)

# ---------------------------------------
# Two-factor theory model (aggregated
#----------------------------------------

# Set functional form model
model <-  '
# Measurement model
ABSCDO =~ NA*cr_as_sbo + 1*cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta + cr_sec_income + cr_serv_fees + cr_cds_purchased
ABCP =~ cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_sec_income + cr_serv_fees+ cr_cds_purchased

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
'

# Fit model
fit <- cfa(model, data = df_agg_log, estimator = 'MLR')
summary(fit, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

# Save results
## Parameter estimates
params <- parameterEstimates(fit, standardized = TRUE)
write.csv(params, 'Results/CFA_params_theory_agg.csv')

## Residual covariance matrix
resid_std <- lavResiduals(fit)$cov.z
write.csv(resid_std, 'Results/CFA_rescov_standard_theory_agg.csv')

## Fit measures
fitmeasures <- fitMeasures(fit, output = 'matrix')
write.csv(fitmeasures, 'Results/CFA_fitmeasures_theory_agg.csv')

## Modindices
modin <- modindices(fit, sort = TRUE)
write.csv(modin, 'Results/CFA_modindices_theory_agg.csv')

## R-square
r2 <- inspect(fit, 'r2', output = 'matrix')
write.csv(r2, 'Results/CFA_r2_theory_agg.csv')

## Reliability
rel <- reliability(fit, return.total = TRUE)
write.csv(rel, 'Results/CFA_reliability_theory_agg.csv')


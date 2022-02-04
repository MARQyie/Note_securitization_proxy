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

# Read file
file <- 'Data/df_sec_note_20112017_balanced.csv'
df <- read.csv(file)

# Log-transform
df$cr_serv_fees <- df$cr_serv_fees - min(df$cr_serv_fees)
df$cr_ls_income <- df$cr_ls_income - min(df$cr_ls_income)
df$cr_sec_income <- df$cr_sec_income - min(df$cr_sec_income)

df_log <- log(df + 1)

# Transform some variables back to originals
df_log$date <- df$date
df_log$percentiles_5 <- df$percentiles_5
df_log$IDRSSD <- df$IDRSSD
df_log$dum_ta95 <- df$dum_ta95
df_log$dum_ta99 <- df$dum_ta99

# Scale data with TA (used in the rolling window)
df_scaled <- df / df$ta * 100000
df_scaled_log <- log(df_scaled + 1)

df_scaled_log$date <- df$date
df_scaled_log$percentiles_5 <- df$percentiles_5
df_scaled_log$IDRSSD <- df$IDRSSD
df_scaled_log$dum_ta95 <- df$dum_ta95
df_scaled_log$dum_ta99 <- df$dum_ta99

# ---------------------------------------
# Two-factor theory model
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
fit <- cfa(model, data = df_log, estimator = 'MLR')
summary(fit, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

# Save results
## Parameter estimates
params <- parameterEstimates(fit, standardized = TRUE)
write.csv(params, 'Results/CFA_params_theory_balanced.csv')

## Residual covariance matrix
resid_std <- lavResiduals(fit)$cov.z
write.csv(resid_std, 'Results/CFA_rescov_standard_theory_balanced.csv')

## Fit measures
fitmeasures <- fitMeasures(fit, output = 'matrix')
write.csv(fitmeasures, 'Results/CFA_fitmeasures_theory_balanced.csv')

## Modindices
modin <- modindices(fit, sort = TRUE)
write.csv(modin, 'Results/CFA_modindices_theory_balanced.csv')

## R-square
r2 <- inspect(fit, 'r2', output = 'matrix')
write.csv(r2, 'Results/CFA_r2_theory_balanced.csv')

## Reliability
rel <- reliability(fit, return.total = TRUE)
write.csv(rel, 'Results/CFA_reliability_theory_balanced.csv')

# ---------------------------------------
# Panel Model: Intertemporal stability
#----------------------------------------

# Split the dataset into two periods
# Periods: 2011--2013; 2014-2017
df_1113 <- subset(df_log, date < 2014)
df_1417 <- subset(df_log, date > 2013)

# Fit models and check summaries
fit_1113 <- cfa(model, data = df_1113, estimator = 'MLR')
fit_1417 <- cfa(model, data = df_1417, estimator = 'MLR')

summary(fit_1113, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)
summary(fit_1417, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

# Rolling window with three years
# Set list of years
years <- c(2011, 2012, 2013, 2014, 2015)

for (y in years){
  # Set dataset
  df_year <-  subset(df_log, date >= y & date <= y + 2)

  # Fit model
  fit_year <- cfa(model, data = df_year, estimator = 'MLR')

  # Save fit indices and communalities
  ## Fit indices
  fitfile_year <- paste0('Results/CFA_fit_balanced_loopyear_',
                        paste0(y, '.csv'))
  fitmeasures_year <- fitMeasures(fit_year, output = 'matrix')
  write.csv(fitmeasures_year, fitfile_year)

  ## communalities
  r2file_year <- paste0('Results/CFA_r2_balanced_loopyear_',
                        paste0(y, '.csv'))
  r2_year <- inspect(fit_year, 'r2', output = 'matrix')
  write.csv(r2_year, r2file_year)
}

# Redo rolling window, but with the data scaled by total assets
model_alt <-  '
# Measurement model
ABSCDO =~ NA*cr_as_sbo + 1*cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta + cr_sec_income + cr_serv_fees + cr_cds_purchased
ABCP =~ cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_sec_income + cr_serv_fees+ cr_cds_purchased

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_ce_own
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_uc_own + cr_abcp_ce_own
cr_sec_income ~~ cr_serv_fees
'

for (y in years){
  # Set dataset
  df_year <-  subset(df_scaled_log, date >= y & date <= y + 2)

  # Fit model
  fit_year <- cfa(model_alt, data = df_year, estimator = 'MLR', control=list(iter.max=1000))

  # Save fit indices and communalities
  ## Fit indices
  fitfile_year <- paste0('Results/CFA_fit_balancedscaled_loopyear_',
                        paste0(y, '.csv'))
  fitmeasures_year <- fitMeasures(fit_year, output = 'matrix')
  write.csv(fitmeasures_year, fitfile_year)

  ## communalities
  r2file_year <- paste0('Results/CFA_r2_balancedscaled_loopyear_',
                        paste0(y, '.csv'))
  r2_year <- inspect(fit_year, 'r2', output = 'matrix')
  write.csv(r2_year, r2file_year)
}

# ---------------------------------------
# Panel Model: Inter-bank stability
#----------------------------------------

# Cross validation (leave-one-out)
# Set list of banks
lst_rssdid <- unique(df_log$IDRSSD)

# Loop over the list of banks and leave out one bank per Iteration
for (bank in lst_rssdid){
  # Subset the data
  df_bank <- subset(df_log, IDRSSD != bank)

  # Fit model
  fit_bank <- cfa(model, data = df_bank, estimator = 'MLR')

  # Save fit indices and communalities
  ## Fit indices
  fitfile_bank <- paste0('Results/Cross_validation/CFA_fit_crossvalidation_',
                        paste0(bank, '.csv'))
  fitmeasures_bank <- fitMeasures(fit_bank, output = 'matrix')
  write.csv(fitmeasures_bank, fitfile_bank)

  ## communalities
  r2file_bank <- paste0('Results/Cross_validation/CFA_r2_crossvalidation_',
                        paste0(bank, '.csv'))
  r2_bank <- inspect(fit_bank, 'r2', output = 'matrix')
  write.csv(r2_bank, r2file_bank)
}

# Leave 'mega banks' out of the sample (more homogeneous sample)
# NOTE: We run the sample without the .99 quantile banks based on ta
df_99 = subset(df_log, dum_ta99 == 1)
fit_bank99 <- cfa(model, data = df_99, estimator = 'MLR')
summary(fit_bank99, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

# -----------------------------------------
# Tests with total assets
# -----------------------------------------

model_alt1 <-  '
# Measurement model
ABSCDO =~ NA*cr_as_sbo + 1*cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta + cr_sec_income + cr_serv_fees + cr_cds_purchased
ABCP =~ cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_sec_income + cr_serv_fees+ cr_cds_purchased
TA =~ ta

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
'

# Fit model
fit <- cfa(model_alt1, data = df_log, estimator = 'MLR')
summary(fit, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

model_alt2 <-  '
# Measurement model
ABSCDO =~ NA*cr_as_sbo + 1*cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta + cr_sec_income + cr_serv_fees + cr_cds_purchased + ta
ABCP =~ cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_sec_income + cr_serv_fees+ cr_cds_purchased + ta

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
'

# Fit model
fit <- cfa(model_alt2, data = df_log, estimator = 'MLR')
summary(fit, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

years <- c(2011, 2012, 2013, 2014, 2015)

for (y in years){
  # Set dataset
  df_year <-  subset(df_log, date >= y & date <= y + 2)

  # Fit model
  fit_year <- cfa(model_alt2, data = df_year, estimator = 'MLR')

  # Save fit indices and communalities
  ## Fit indices
  fitfile_year <- paste0('Results/CFA_fit_balanced_loopyearalt_',
                        paste0(y, '.csv'))
  fitmeasures_year <- fitMeasures(fit_year, output = 'matrix')
  write.csv(fitmeasures_year, fitfile_year)

  ## communalities
  r2file_year <- paste0('Results/CFA_r2_balanced_loopyearalt_',
                        paste0(y, '.csv'))
  r2_year <- inspect(fit_year, 'r2', output = 'matrix')
  write.csv(r2_year, r2file_year)
}
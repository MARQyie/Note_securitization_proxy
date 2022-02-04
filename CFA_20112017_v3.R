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

df$cr_serv_fees <- df$cr_serv_fees - min(df$cr_serv_fees)
df$cr_ls_income <- df$cr_ls_income - min(df$cr_ls_income)
df$cr_sec_income <- df$cr_sec_income - min(df$cr_sec_income)

df_log <- log(df + 1)

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
modin_onesec <- modindices(fit_onesec, sort = TRUE)
write.csv(modin_onesec, 'Results/CFA_modindices_onesec.csv')

## R-square
r2_onesec <- inspect(fit_onesec, 'r2', output = 'matrix')
write.csv(r2_onesec, 'Results/CFA_r2_onesec.csv')

## Reliability
reli_onesec <- reliability(fit_onesec)
write.csv(reli_onesec, 'Results/CFA_reliability_onesec.csv')

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
write.csv(params_efa, 'Results/CFA_params_efa.csv')

## Model implied covariance matrix
fitted_efa <- fitted(fit_efa)$cov
write.csv(fitted_efa, 'Results/CFA_modimplied_cov_efa.csv')

## Residual covariance matrix
resid_efa <- lavResiduals(fit_efa)$cov
resid_std_efa <- lavResiduals(fit_efa)$cov.z
write.csv(resid_efa, 'Results/CFA_rescov_efa.csv')
write.csv(resid_std_efa, 'Results/CFA_rescov_standard_efa.csv')

## Fit measures
fitmeasures_efa <- fitMeasures(fit_efa, output = 'matrix')
write.csv(fitmeasures_efa, 'Results/CFA_fitmeasures_efa.csv')

## Modindices
modin_efa <- modindices(fit_efa, sort = TRUE)
write.csv(modin_efa, 'Results/CFA_modindices_efa.csv')

## R-square
r2_efa <- inspect(fit_efa, 'r2', output = 'matrix')
write.csv(r2_efa, 'Results/CFA_r2_efa.csv')

## Reliability
reli_efa <- reliability(fit_efa)
write.csv(reli_efa, 'Results/CFA_reliability_efa.csv')

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
write.csv(params_theory, 'Results/CFA_params_theory.csv')

## Model implied covariance matrix
fitted_theory <- fitted(fit_theory)$cov
write.csv(fitted_theory, 'Results/CFA_modimplied_cov_theory.csv')

## Residual covariance matrix
resid_theory <- lavResiduals(fit_theory)$cov
resid_std_theory <- lavResiduals(fit_theory)$cov.z
write.csv(resid_theory, 'Results/CFA_rescov_theory.csv')
write.csv(resid_std_theory, 'Results/CFA_rescov_standard_theory.csv')

## Fit measures
fitmeasures_theory <- fitMeasures(fit_theory, output = 'matrix')
write.csv(fitmeasures_theory, 'Results/CFA_fitmeasures_theory.csv')

## Modindices
modin_theory <- modindices(fit_theory, sort = TRUE)
write.csv(modin_theory, 'Results/CFA_modindices_theory.csv')

## R-square
r2_theory <- inspect(fit_theory, 'r2', output = 'matrix')
write.csv(r2_theory, 'Results/CFA_r2_theory.csv')

## Reliability
reli_theory <- reliability(fit_theory)
write.csv(reli_theory, 'Results/CFA_reliability_theory.csv')

# ---------------------------------------
# Calculate the reliability with confidence intervals

# Example
sds <- '1.15 1.20 1.57 2.82 1.31 1.24 1.33 1.29'

cors <- '
  1.000
  0.594  1.000
  0.607  0.613  1.000
  0.736  0.765  0.717  1.000
  0.378  0.321  0.360  0.414  1.000
  0.314  0.301  0.345  0.363  0.732  1.000
  0.310  0.262  0.323  0.337  0.665  0.583  1.000
  0.317  0.235  0.276  0.302  0.632  0.557  0.796  1.000'

covs <- getCov(cors, sds = sds, names = paste("Y", 1:8, sep = ""))


model <- '
  # main model.
  intrus =~ Y1 + l1*Y1 + l2*Y2 + l3*Y3 + l4*Y4
  avoid  =~ Y5 + l5*Y5 + l6*Y6 + l7*Y7 + l8*Y8
  # label the residual variances
  Y1 ~~ e1*Y1
  Y2 ~~ e2*Y2
  Y3 ~~ e3*Y3
  Y4 ~~ e4*Y4
  Y5 ~~ e5*Y5
  Y6 ~~ e6*Y6
  Y7 ~~ e7*Y7
  Y8 ~~ e8*Y8
  # covariance between Y7 and Y8
  Y7 ~~ e78*Y8
  # defined parameters
  intr.tru := (l1 + l2 + l3 + l4)^2
  intr.tot := (l1 + l2 + l3 + l4)^2 + e1 + e2 + e3 + e4
  intr.rel := intr.tru/intr.tot
  avoid.tru := (l5 + l6 + l7 + l8)^2
  avoid.tot := (l5 + l6 + l7 + l8)^2 + e5 + e6 + e7 + e8 + 2*e78
  avoid.rel := avoid.tru/avoid.tot
'

fit <- cfa(model, sample.cov = covs, sample.nobs = 500, mimic = "EQS", std.lv = TRUE)
summary(fit, standardized = TRUE, fit.measures = TRUE)

# 95% CI
parameterEstimates(fit)

#
model_theory_ci <-  '
# Measurement model
ABSCDO =~  cr_as_rmbs + l1*cr_as_sbo + l2*cr_as_abs + l3*hmda_sec_amount + l4*cr_secveh_ta + l5*cr_sec_income + l6*cr_serv_fees + l7*cr_cds_purchased
ABCP =~ cr_abcp_ta + l8*cr_abcp_uc_own + l9*cr_abcp_ce_own + l10*cr_abcp_uc_oth + l11*cr_sec_income + l12*cr_serv_fees + l13*cr_cds_purchased

# Error covariances
cr_abcp_uc_own ~~ ec1*cr_abcp_uc_oth
cr_abcp_ta ~~ ec2*cr_secveh_ta + ec3*cr_abcp_ce_own

# Error variances
cr_as_sbo ~~ e1*cr_as_sbo
cr_as_rmbs ~~ e2*cr_as_rmbs
cr_as_abs ~~ e3*cr_as_abs
hmda_sec_amount ~~ e4*hmda_sec_amount
cr_secveh_ta ~~ e5*cr_secveh_ta
cr_sec_income ~~ e6*cr_sec_income
cr_serv_fees ~~ e7*cr_serv_fees
cr_cds_purchased ~~ e8*cr_cds_purchased
cr_abcp_ta ~~ e9*cr_abcp_ta
cr_abcp_uc_own ~~ e10*cr_abcp_uc_own
cr_abcp_ce_own ~~ e11*cr_abcp_ce_own
cr_abcp_uc_oth ~~ e12*cr_abcp_uc_oth

# Calculate reliability scores (and confidence intervals)
ABSCDO.tru := (l1 + l2 + l3 + l4 + l5 + l6 + l7)^2
ABSCDO.tot := (l1 + l2 + l3 + l4 + l5 + l6 + l7)^2 + e1 + e2 + e3 + e4 + e5 + e6 + e7 + e8 + 2*ec2
ABSCDO.rel := ABSCDO.tru / ABSCDO.tot

ABCP.tru := (l8 + l9 + l10 + l11 + l12 + l13)^2
ABCP.tot := (l8 + l9 + l10 + l11 + l12 + l13)^2 + e6 + e7 + e8 + e9 + e10 + e11 + e12 + 2*(ec1 + ec2 + ec3)
ABCP.rel := ABCP.tru / ABCP.tot
'

fit_theory_ci <- cfa(model_theory_ci, data = df_log, estimator = 'MLR')
summary(fit_theory_ci, standardized = TRUE, fit.measures = TRUE)
parameterEstimates(fit_theory_ci)

# ---------------------------------------
# Combined
#----------------------------------------

model_combined <-  '
# Measurement model
ABSCDO =~ NA*cr_as_sbo + 1*cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_sec_8veh_ta + cr_sec_income + cr_serv_fees
ABCP =~ NA*cr_secveh_ta + 1*cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_cds_purchased

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
'

fit_combined <- cfa(model_combined, data = df_log, estimator = 'MLR')
summary(fit_combined, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

# Save results
## Parameter estimates
params_combined <- parameterEstimates(fit_combined, standardized = TRUE)
write.csv(params_combined, 'Results/CFA_params_combined.csv')

## Model implied covariance matrix
fitted_combined <- fitted(fit_combined)$cov
write.csv(fitted_combined, 'Results/CFA_modimplied_cov_combined.csv')

## Residual covariance matrix
resid_combined <- lavResiduals(fit_combined)$cov
resid_std_combined <- lavResiduals(fit_combined)$cov.z
write.csv(resid_combined, 'Results/CFA_rescov_combined.csv')
write.csv(resid_std_combined, 'Results/CFA_rescov_standard_combined.csv')

## Fit measures
fitmeasures_combined <- fitMeasures(fit_combined, output = 'matrix')
write.csv(fitmeasures_combined, 'Results/CFA_fitmeasures_combined.csv')

## Modindices
modin_combined <- modindices(fit_combined, sort = TRUE)
write.csv(modin_combined, 'Results/CFA_modindices_combined.csv')

## R-square
r2_combined <- inspect(fit_combined, 'r2', output = 'matrix')
write.csv(r2_combined, 'Results/CFA_r2_combined.csv')

## Reliability
reli_combined <- reliability(fit_combined)
write.csv(reli_combined, 'Results/CFA_reliability_combined.csv')

# ---------------------------------------
# Chi2 difference tests
#----------------------------------------

# Efa and theory
# NOTE: NOT POSSIBLE TO TEST BECAUSE fit_efa HAS FEWER PARAMETERS
# BE CAREFUL WITH COMPARING FIT INDICES

# Theory and combined
lavTestLRT(fit_theory, fit_combined) # Theory is better

# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------
# Extra models
#----------------------------------------

# NOTE: WE CHOOSE THE BEST FITTING MODEL OF THE TWO FACTOR MODELS AND USE THAT TO ESTIMATE THE REST

# Higher order model
#----------------------------------------
# NOTE: MODEL IS UNDERIDENTIFIED. ESTIMATES ARE NOT TRISTWORTHY

model_higherorder <-  '
# Measurement model
ABSCDO =~ NA*cr_as_sbo + 1*cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta + cr_sec_income + cr_serv_fees + cr_cds_purchased
ABCP =~ cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_sec_income + cr_serv_fees + cr_cds_purchased
SEC =~ NA*ABSCDO + start(1.2)*ABCP 

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own

SEC ~~ 1*SEC
'

fit_higherorder <- cfa(model_higherorder, data = df_log, estimator = 'MLR')
summary(fit_higherorder, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)
#NOTE: WITHOUT THE STARTING VALUE OF ABCP ON SEC, THE COV IS NOT POSITIVE DEFINITE (= -1.070024e-05)

#Save results
## Parameter estimates
params_higherorder <- parameterEstimates(fit_higherorder, standardized = TRUE)
write.csv(params_higherorder, 'Results/CFA_params_higherorder.csv')

## Model implied covariance matrix
fitted_higherorder <- fitted(fit_higherorder)$cov
write.csv(fitted_higherorder, 'Results/CFA_modimplied_cov_higherorder.csv')

## Residual covariance matrix
resid_higherorder <- lavResiduals(fit_higherorder)$cov
resid_std_higherorder <- lavResiduals(fit_higherorder)$cov.z
write.csv(resid_higherorder, 'Results/CFA_rescov_higherorder.csv')
write.csv(resid_std_higherorder, 'Results/CFA_rescov_standard_higherorder.csv')

## Fit measures
fitmeasures_higherorder <- fitMeasures(fit_higherorder, output = 'matrix')
write.csv(fitmeasures_higherorder, 'Results/CFA_fitmeasures_higherorder.csv')

## Modindices
#NOTE: could not compute modification indices; information matrix is singular
#modin_higherorder <- modindices(fit_higherorder, sort = TRUE, maximum.number = 25)
#write.csv(modin_higherorder, 'Results/CFA_modindices_higherorder.csv')

## R-square
r2_higherorder <- inspect(fit_higherorder, 'r2', output = 'matrix')
write.csv(r2_higherorder, 'Results/CFA_r2_higherorder.csv')

## Reliability
reli_higherorder <- reliability(fit_higherorder)
write.csv(reli_higherorder, 'Results/CFA_reliability_higherorder.csv')

## Chi2 difference test
#lavTestLRT(fit_theory, fit_higherorder) # THERE IS NO DIFFERENCE

# Higher order model -- ALTERNATIVE
#----------------------------------------

model_higherorder_alt <-  '
# Measurement model
ABSCDO =~ NA*cr_as_sbo + 1*cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta
ABCP =~ cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth
SEC =~ NA*ABSCDO + ABCP + 1*cr_sec_income + cr_serv_fees + cr_cds_purchased

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
'

fit_higherorder_alt <- cfa(model_higherorder_alt, data = df_log, estimator = 'MLR')
summary(fit_higherorder_alt, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

#Save results
## Parameter estimates
params_higherorder_alt <- parameterEstimates(fit_higherorder_alt, standardized = TRUE)
write.csv(params_higherorder_alt, 'Results/CFA_params_higherorder_alt.csv')

## Model implied covariance matrix
fitted_higherorder_alt <- fitted(fit_higherorder_alt)$cov
write.csv(fitted_higherorder_alt, 'Results/CFA_modimplied_cov_higherorder_alt.csv')

## Residual covariance matrix
resid_higherorder_alt <- lavResiduals(fit_higherorder_alt)$cov
resid_std_higherorder_alt <- lavResiduals(fit_higherorder_alt)$cov.z
write.csv(resid_higherorder_alt, 'Results/CFA_rescov_higherorder_alt.csv')
write.csv(resid_std_higherorder_alt, 'Results/CFA_rescov_standard_higherorder_alt.csv')

## Fit measures
fitmeasures_higherorder_alt <- fitMeasures(fit_higherorder_alt, output = 'matrix')
write.csv(fitmeasures_higherorder_alt, 'Results/CFA_fitmeasures_higherorder_alt.csv')

## Modindices
modin_higherorder_alt <- modindices(fit_higherorder_alt, sort = TRUE)
write.csv(modin_higherorder_alt, 'Results/CFA_modindices_higherorder_alt.csv')

## R-square
r2_higherorder_alt <- inspect(fit_higherorder_alt, 'r2', output = 'matrix')
write.csv(r2_higherorder_alt, 'Results/CFA_r2_higherorder_alt.csv')

## Reliability
reli_higherorder_alt <- reliability(fit_higherorder_alt)
write.csv(reli_higherorder_alt, 'Results/CFA_reliability_higherorder_alt.csv')

# Loan sales model
#----------------------------------------

model_loansales <-  '
# Measurement model
ABSCDO =~ NA*cr_as_sbo + 1*cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta + cr_sec_income + cr_serv_fees + cr_cds_purchased
ABCP =~ cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_sec_income + cr_serv_fees + cr_cds_purchased
#LS =~ cr_as_nsres + cr_as_nsoth + cr_as_sbo + cr_serv_fees

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
cr_as_sbo ~~ cr_serv_fees
'

fit_loansales <- cfa(model_loansales, data = df_log, estimator = 'MLR')
summary(fit_loansales, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

fitMeasures(fit_loansales, output = 'matrix')
reliability(fit_loansales)

# No Low Communalities model
#----------------------------------------
model_nlc <-  '
# Measurement model
ABSCDO =~ cr_as_rmbs + cr_as_abs + cr_secveh_ta + cr_sec_income + cr_cds_purchased
ABCP =~ cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_sec_income + cr_cds_purchased

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
'

fit_nlc <- cfa(model_nlc, data = df_log, estimator = 'MLR')
summary(fit_nlc, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)
fitMeasures(fit_nlc, output = 'matrix')
reliability(fit_nlc)

# SBO separate model
#----------------------------------------

model_sboseparate <-  '
# Measurement model
ABSCDO =~ + 1*cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta + cr_sec_income + cr_serv_fees+ cr_cds_purchased
ABCP =~ cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_sec_income + cr_serv_fees+ cr_cds_purchased
LS =~ cr_as_sbo

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own

# Covariances between SBO and factors
#ABSCDO ~~ cr_as_sbo
#ABCP ~~ cr_as_sbo
'

fit_sboseparate <- cfa(model_sboseparate, data = df_log, estimator = 'MLR')
summary(fit_sboseparate, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

#Save results
## Parameter estimates
params_sboseparate <- parameterEstimates(fit_sboseparate, standardized = TRUE)
write.csv(params_sboseparate, 'Results/CFA_params_sboseparate.csv')

## Residual covariance matrix
resid_std_sboseparate <- lavResiduals(fit_sboseparate)$cov.z
write.csv(resid_std_sboseparate, 'Results/CFA_rescov_standard_sboseparate.csv')

## Fit measures
fitmeasures_sboseparate <- fitMeasures(fit_sboseparate, output = 'matrix')
write.csv(fitmeasures_sboseparate_alt, 'Results/CFA_fitmeasures_sboseparate.csv')

## Modindices
modin_sboseparate <- modindices(fit_sboseparate)
write.csv(modin_sboseparate, 'Results/CFA_modindices_sboseparate.csv')

# ---------------------------------------
# Theory model with extra covariance sec. income -- TA sec. vehicles
#----------------------------------------

model_theory_extra <-  '
# Measurement model
ABSCDO =~ NA*cr_as_sbo + 1*cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta + cr_sec_income + cr_serv_fees+ cr_cds_purchased
ABCP =~ cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_sec_income + cr_serv_fees+ cr_cds_purchased

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
cr_sec_income ~~ cr_secveh_ta
cr_as_rmbs ~~ cr_abcp_uc_oth
'

fit_theory_extra <- cfa(model_theory_extra, data = df_log, estimator = 'MLR')
summary(fit_theory_extra, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

#Save results
## Parameter estimates
params_theory_extra <- parameterEstimates(fit_theory_extra, standardized = TRUE)
write.csv(params_theory_extra, 'Results/CFA_params_theory_extra.csv')

## Model implied covariance matrix
fitted_theory_extra <- fitted(fit_theory_extra)$cov
write.csv(fitted_theory_extra, 'Results/CFA_modimplied_cov_theory_extra.csv')

## Residual covariance matrix
resid_theory_extra <- lavResiduals(fit_theory_extra)$cov
resid_std_theory_extra <- lavResiduals(fit_theory_extra)$cov.z
write.csv(resid_theory_extra, 'Results/CFA_rescov_theory_extra.csv')
write.csv(resid_std_theory_extra, 'Results/CFA_rescov_standard_theory_extra.csv')

## Fit measures
fitmeasures_theory_extra <- fitMeasures(fit_theory_extra, output = 'matrix')
write.csv(fitmeasures_theory_extra, 'Results/CFA_fitmeasures_theory_extra.csv')

## Modindices
modin_theory_extra <- modindices(fit_theory_extra, sort = TRUE)
write.csv(modin_theory_extra, 'Results/CFA_modindices_theory_extra.csv')

## R-square
r2_theory_extra <- inspect(fit_theory_extra, 'r2', output = 'matrix')
write.csv(r2_theory_extra, 'Results/CFA_r2_theory_extra.csv')

## Reliability
reli_theory_extra <- reliability(fit_theory_extra)
write.csv(reli_theory_extra, 'Results/CFA_reliability_theory_extra.csv')



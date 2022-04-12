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
file <- 'Data/df_sec_note_binary_20112017.csv'
df_binary <- read.csv(file)

# ---------------------------------------
# Loop over estimators
#----------------------------------------

# Set model equations
model_sbo <-  '
# Measurement model
ABSCDO =~ NA*cr_as_sbo + 1*cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta  + cr_cds_purchased
ABCP =~ cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_cds_purchased

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
'

model_nosbo <-  '
# Measurement model
ABSCDO =~ cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta + cr_cds_purchased
ABCP =~ cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_cds_purchased

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
'

# Set list of estimators
# Note: we do not use ML with bootstrap, because BS-ML cannot handle extreme non-normality well
lst_estimators <- c('WLSMV','DWLS')

# Loop over estimators
for (estimator in lst_estimators){
  for (model in c(model_sbo,model_nosbo)){
    if (model == model_sbo){
      model_str <- 'SBO'
    } else {
      model_str <- 'NoSBO'
    }
    fit_binary <- cfa(model,
                      data = df_binary,
                      estimator = estimator,
                      ordered = TRUE)
    # summary(fit_binary, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

    #Save results
    ## Parameter estimates
    params_binary <- parameterEstimates(fit_binary, standardized = TRUE)
    write.csv(params_binary, sprintf('Results/CFA_params_binary_%s_%s.csv',model_str,estimator))

    ## Fit measures
    fitmeasures_binary <- fitMeasures(fit_binary, output = 'matrix')
    write.csv(fitmeasures_binary, sprintf('Results/CFA_fitmeasures_binary_%s_%s.csv',model_str,estimator))

    ## R-square
    r2_binary <- inspect(fit_binary, 'r2', output = 'matrix')
    write.csv(r2_binary, sprintf('Results/CFA_r2_binary_%s_%s.csv',model_str,estimator))

    ## Residual standardized covariance matrix
    resid_binary <- lavResiduals(fit_binary)$cov.z
    write.csv(resid_binary, sprintf('Results/CFA_rescov_standard_binary_%s_%s.csv',model_str,estimator))

    ## Polychoric Matrix
    pc_corr <- lavCor(fit_binary, ordered = TRUE, group = NULL, output = 'cor')
    write.csv(pc_corr, sprintf('Results/CFA_polychoriccorr_binary_%s_%s.csv',model_str,estimator))

    ## Modindices
    modin_binary <- modindices(fit_binary, sort = TRUE)
    write.csv(modin_binary, sprintf('Results/CFA_modindices_binary_%s_%s.csv',model_str,estimator))

    ## Reliability
    reli_binary <- reliability(fit_binary)
    write.csv(reli_binary, sprintf('Results/CFA_reliability_binary_%s_%s.csv',model_str,estimator))
  }
}

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

  ## Polychoric Matrix
  pc_corr <- lavCor(fit_binary, ordered = TRUE, group = NULL, output = 'cor')
  write.csv(pc_corr, sprintf('Results/CFA_polychoriccorr_binary_1f_%s.csv',estimator))

  ## Modindices
  modin_binary <- modindices(fit_binary, sort = TRUE)
  write.csv(modin_binary, sprintf('Results/CFA_modindices_binary_1f_%s.csv',estimator))

  ## Reliability
  reli_binary <- reliability(fit_binary)
  write.csv(reli_binary, sprintf('Results/CFA_reliability_binary_1f_%s.csv',estimator))
}

# ---------------------------------------
# Two-factor model after spec. search
#----------------------------------------

# Set model equations
model_1f_specsearch <-  '
# Measurement model
SEC =~ cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta  + cr_cds_purchased + cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
cr_abcp_uc_oth ~~ cr_as_rmbs
cr_secveh_ta ~~ cr_cds_purchased + cr_abcp_uc_own
'

fit_binary <- cfa(model_1f_specsearch,
                    data = df_binary,
                    estimator = 'WLSMV',
                    ordered = TRUE)
summary(fit_binary, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

# ---------------------------------------
# Visualization
#----------------------------------------
# Set labels
labels <- list(ABSCDO = 'F1',
              ABCP = 'F2',
              cr_as_sbo = 'SBO Transf.',
              cr_as_rmbs = 'Sec. Res. Loans',
              cr_as_abs = 'Sec. Other Assets',
              hmda_sec_amount = 'Sec. Res. Mort.',
              cr_secveh_ta = 'TA Sec. Vehicles',
              cr_cds_purchased = 'CDSs Pur.',
              cr_abcp_ta = 'TA ABCP Conduits',
              cr_abcp_uc_own = 'U.C. Own ABCP Conduits',
              cr_abcp_ce_own = 'C.E. Own ABCP Conduits',
              cr_abcp_uc_oth = 'U.C.  Other ABCP Conduits')

# Fit 2f with SBO
fit_binary <- cfa(model_sbo,
                      data = df_binary,
                      estimator = 'WLSMV',
                      ordered = TRUE)

# Visualize
library(lavaanPlot)
lavaanPlot(model = fit_binary,
           labels = labels,
           node_options = list(shape = "box", fontname = "Helvetica"),
           edge_options = list(color = "black"),
           coefs = TRUE,
           covs = TRUE,
           stand = TRUE,
           stars = list('latent','covs'))

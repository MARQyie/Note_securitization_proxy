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
file <- 'Data/df_sec_note_binary_20112017.csv'
df_binary <- read.csv(file)

# ---------------------------------------
# Estimate SEMS
#----------------------------------------

# Get estimates
model_hier1a <-  '
# Measurement model
ABSCDO =~ cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta  + cr_cds_purchased
ABCP =~ cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_cds_purchased
SEC =~ ABSCDO + ABCP

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
'
fit_hier1a <- cfa(model_hier1a,
                data = df_binary,
                estimator = 'WLSMV',
                ordered = TRUE)
summary(fit_hier1a, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)
# Warning in lav_model_vcov(lavmodel = lavmodel, lavsamplestats = lavsamplestats,  :
#   lavaan WARNING:
#     Could not compute standard errors! The information matrix could
#     not be inverted. This may be a symptom that the model is not
#     identified.
# Warning in lav_test_satorra_bentler(lavobject = NULL, lavsamplestats = lavsamplestats,  :
#   lavaan WARNING: could not invert information matrix needed for robust test statistic
#
# Warning in lav_object_post_check(object) :
#   lavaan WARNING: some estimated lv variances are negative

model_hier1b <-  '
# Measurement model
ABSCDO =~ cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta
ABCP =~ cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth
SEC =~ ABSCDO + ABCP + cr_cds_purchased

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
'
fit_hier1b <- cfa(model_hier1b,
                data = df_binary,
                estimator = 'WLSMV',
                ordered = TRUE)
summary(fit_hier1b, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)
# Warning in lav_object_post_check(object) :
#   lavaan WARNING: some estimated lv variances are negative

model_hier1c <-  '
# Measurement model
ABSCDO =~ cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta
ABCP =~ cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth
SEC =~ cr_cds_purchased + ABSCDO + ABCP

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
'
fit_hier1c <- cfa(model_hier1c,
                data = df_binary,
                estimator = 'WLSMV',
                ordered = TRUE)
summary(fit_hier1c, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)
# Warning in lav_object_post_check(object) :
#   lavaan WARNING: some estimated lv variances are negative

model_hier1d <-  '
# Measurement model
ABSCDO =~ cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta
ABCP =~ cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth
SEC =~ ABSCDO + ABCP + cr_cds_purchased

# Error covariances
cr_abcp_ta ~~ cr_abcp_ce_own
'
fit_hier1d <- cfa(model_hier1d,
                data = df_binary,
                estimator = 'WLSMV',
                ordered = TRUE)
summary(fit_hier1d, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)
# Warning in lav_object_post_check(object) :
#   lavaan WARNING: some estimated lv variances are negative

model_hier2a <-  '
# Measurement model
ABSCDO =~ cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta  + cr_cds_purchased
ABCP =~ cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_cds_purchased
SEC =~ ABCP + ABSCDO

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
'
fit_hier2a <- cfa(model_hier2a,
                data = df_binary,
                estimator = 'WLSMV',
                ordered = TRUE)
summary(fit_hier2a, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)
# Warning in lav_model_vcov(lavmodel = lavmodel, lavsamplestats = lavsamplestats,  :
#   lavaan WARNING:
#     Could not compute standard errors! The information matrix could
#     not be inverted. This may be a symptom that the model is not
#     identified.
# Warning in lav_test_satorra_bentler(lavobject = NULL, lavsamplestats = lavsamplestats,  :
#   lavaan WARNING: could not invert information matrix needed for robust test statistic

model_hier2b <-  '
# Measurement model
ABSCDO =~ cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta
ABCP =~ cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth
SEC =~ ABCP + ABSCDO + cr_cds_purchased

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
'
fit_hier2b <- cfa(model_hier2b,
                data = df_binary,
                estimator = 'WLSMV',
                ordered = TRUE)
summary(fit_hier2b, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)
# Warning in lav_object_post_check(object) :
#   lavaan WARNING: some estimated lv variances are negative

model_hier_stda <-  '
# Measurement model
ABSCDO =~ NA*cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta  + cr_cds_purchased
ABCP =~ NA*cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth + cr_cds_purchased
SEC =~ NA*ABSCDO + ABCP

# Set factor variances to one
ABSCDO ~~ 1*ABSCDO
ABCP ~~ 1*ABCP
SEC ~~ 1*SEC

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
'
fit_hier_stda <- cfa(model_hier_stda,
                data = df_binary,
                estimator = 'WLSMV',
                ordered = TRUE)
summary(fit_hier_stda, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)
# Warning in lav_model_vcov(lavmodel = lavmodel, lavsamplestats = lavsamplestats,  :
#   lavaan WARNING:
#     Could not compute standard errors! The information matrix could
#     not be inverted. This may be a symptom that the model is not
#     identified.
# Warning in lav_test_satorra_bentler(lavobject = NULL, lavsamplestats = lavsamplestats,  :
#   lavaan WARNING: could not invert information matrix needed for robust test statistic

model_hier_stdb <-  '
# Measurement model
ABSCDO =~ NA*cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta
ABCP =~ NA*cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth
SEC =~ NA*ABSCDO + ABCP + cr_cds_purchased

# Set factor variances to one
ABSCDO ~~ 1*ABSCDO
ABCP ~~ 1*ABCP
SEC ~~ 1*SEC

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
'
fit_hier_stdb <- cfa(model_hier_stdb,
                data = df_binary,
                estimator = 'WLSMV',
                ordered = TRUE)
summary(fit_hier_stdb, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)
# Warning in lav_model_vcov(lavmodel = lavmodel, lavsamplestats = lavsamplestats,  :
#   lavaan WARNING:
#     Could not compute standard errors! The information matrix could
#     not be inverted. This may be a symptom that the model is not
#     identified.
# Warning in lav_test_satorra_bentler(lavobject = NULL, lavsamplestats = lavsamplestats,  :
#   lavaan WARNING: could not invert information matrix needed for robust test statistic

model_hier_stdc <-  '
# Measurement model
ABSCDO =~ cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta
ABCP =~ cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth
SEC =~ NA*ABSCDO + ABCP + cr_cds_purchased

# Set factor variances to one
SEC ~~ 1*SEC

# Error covariances
cr_abcp_uc_own ~~ cr_abcp_uc_oth
cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
'
fit_hier_stdc <- cfa(model_hier_stdc,
                data = df_binary,
                estimator = 'WLSMV',
                ordered = TRUE)
summary(fit_hier_stdc, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

# Warning in lav_object_post_check(object) :
#   lavaan WARNING: some estimated lv variances are negative
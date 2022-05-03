# -------------------------------------------------
# Confirmatory factor analysis bootstrap correction
# Mark van der Plaat
# May 2022
# -------------------------------------------------

# This script performs a bootstrap correction for
# the zeros-problem of the one-factor model.

# -------------------------------------------------
# Load packages and other prelims
# -------------------------------------------------

# Set wd
setwd('D:/RUG/PhD/Materials_papers/01-Note_on_securitization')

# Import CFA package
library(lavaan)
library(semTools)

# other packages
library(rlist)
library(evaluate)
library(foreach)
library(doParallel)

# Set number of bootstraps
M <- 1000

# -------------------------------------------------
# Set functions
# -------------------------------------------------

# Block sampler (panel sampler)
blockSampler <- function(unique_banks,data) {
  # Function resamples the original dataframe but keeps the panel structure.
  #
  #     Input
  #     ----
  #     unique_banks : List of unique bank IDRSSDs
  #     data : Original dataframe (balanced)
  #
  #     Output
  #     ----
  #     data_resampled : resampled dataframe

  # Randomly sample bank IDRSSDs
  sample_banks <- sample(unique_banks, size=length(unique_banks), replace=T)

  # Fetch all years for each randomly picked bank and rbind
  data_resampled <- do.call(rbind, lapply(sample_banks, function(x)  data[data$IDRSSD==x,]))

  # Check if one of the columns is zero only. If true resample data.
  # Continue until statement is satisfied
  data_condition <- lapply(data_resampled[3:ncol(data_resampled)], function(x) all(x == 0))

  while (any(data_condition == T)){
    sample_banks <- sample(unique_banks, size=length(unique_banks), replace=T )
    data_resampled <- do.call(rbind, lapply(sample_banks, function(x)  data[data$IDRSSD==x,]))
    data_condition <- lapply(data_resampled[3:ncol(data_resampled)], function(x) all(x == 0))
  }

  # Return data_resampled
  return(data_resampled)
}

# CFA estimation
CFAWrapper <- function(data){
  # Function estimates the one-factor CFA model and returns the
  #     parameter estimates.
  #
  #     Input
  #     ----
  #     data : dataframe (balanced)
  #
  #     Output
  #     ----
  #     params : estimated parameters of the one-factor CFA model. Labels are not returned.

  # Set model
  model_1f <-  '
              # Measurement model
              SEC =~ cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta  + cr_cds_purchased + cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth

              # Error covariances
              cr_abcp_uc_own ~~ cr_abcp_uc_oth
              cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
'

  # Estimate parameters
  # NOTE includes all estimated parameters, not only the factor loadings.
  # Do not estimate the standard errors, not needed and takes too much time
  fit <- cfa(model_1f,
              data = data,
              estimator = 'WLSMV',
              ordered = TRUE,
              se = 'none')
  params <- parameterEstimates(fit)$est
  #params <- coef(fit)

  # Return parameter estimates
  return(params)
}

# Bootstrap loop function
bootstrapLooper <- function(unique_banks, data, M){
  # Set seed
  set.seed(42)

  # Set function to feed to the foreach loop
  foreachWrapper <- function(unique_banks, data){
    # Resample Data
    data_resampled <- blockSampler(unique_banks, data)

    # Get parameter estimates
    bootstrap_results <- CFAWrapper(data_resampled)

    return(bootstrap_results)
  }

  # Set parallel parameters
  # cores <- detectCores()
  # myCluster <- makeCluster(cores,
  #                          type = "PSOCK")
  # registerDoParallel(myCluster)

  # Bootstrap the parameters
  list_params <- foreach(i = 1:M, .combine = rbind) %do% {
    foreachWrapper(unique_banks, data)
  }
  # stopCluster(myCluster)

  # return results
  return(list_params)
}

# -------------------------------------------------
# Load data
# -------------------------------------------------

# Import csv
file <- 'Data/df_sec_note_binary_balance.csv'
df <- read.csv(file)

# Remove unneeded rows to speed up sampling
vars <- c('IDRSSD',
          'date',
          'cr_as_sbo',
          'cr_as_rmbs',
          'cr_as_abs',
          'hmda_sec_amount',
          'cr_secveh_ta',
          'cr_cds_purchased',
          'cr_abcp_ta',
          'cr_abcp_uc_own',
          'cr_abcp_ce_own',
          'cr_abcp_uc_oth')
df <- df[vars]

# Set unique banks list
unique_banks <- unique(df$IDRSSD)

# -------------------------------------------------
# Run Bootstrap Correction
# -------------------------------------------------

# Run bootstrap
# TODO Error and warning capturing

list_params <- bootstrapLooper(unique_banks, df, M)

# Calculate bias-corrected parameter estimates
params_boot_mean <- colMeans(list_params)
params_original <- CFAWrapper(df)
bias_correction <- params_boot_mean - params_original
params_corrected <- params_original - bias_correction

# Make table and save
table <- matrix(c(params_original, params_boot_mean, params_corrected, bias_correction), ncol = 4)
colnames(table) <- c('params_original','params_boot_mean','params_corrected','bias_correction')

write.csv(table, 'Bootstrap_correction/Results_bootstrap_corrections_with_warnings.csv')
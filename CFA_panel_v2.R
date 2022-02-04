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
file <- 'Data/df_sec_note_wide.csv'
df <- read.csv(file)

df_log <- log(df + 1)

# ---------------------------------------
# TSO model
#----------------------------------------
model <- '
# Repeated measurement model
ABSCDO2011 =~ cr_as_rmbs2011 + cr_as_abs2011 + cr_as_sbo2011 + hmda_sec_amount2011 + cr_secveh_ta2011 + cr_cds_purchased2011 + cr_sec_income2011 + cr_serv_fees2011
ABSCDO2012 =~ cr_as_rmbs2012 + cr_as_abs2012 + cr_as_sbo2012 + hmda_sec_amount2012 + cr_secveh_ta2012 + cr_cds_purchased2012 + cr_sec_income2012 + cr_serv_fees2012
ABSCDO2013 =~ cr_as_rmbs2013 + cr_as_abs2013 + cr_as_sbo2013 + hmda_sec_amount2013 + cr_secveh_ta2013 + cr_cds_purchased2013 + cr_sec_income2013 + cr_serv_fees2013
ABSCDO2014 =~ cr_as_rmbs2014 + cr_as_abs2014 + cr_as_sbo2014 + hmda_sec_amount2014 + cr_secveh_ta2014 + cr_cds_purchased2014 + cr_sec_income2014 + cr_serv_fees2014
ABSCDO2015 =~ cr_as_rmbs2015 + cr_as_abs2015 + cr_as_sbo2015 + hmda_sec_amount2015 + cr_secveh_ta2015 + cr_cds_purchased2015 + cr_sec_income2015 + cr_serv_fees2015
ABSCDO2016 =~ cr_as_rmbs2016 + cr_as_abs2016 + cr_as_sbo2016 + hmda_sec_amount2016 + cr_secveh_ta2016 + cr_cds_purchased2016 + cr_sec_income2016 + cr_serv_fees2016
ABSCDO2017 =~ cr_as_rmbs2017 + cr_as_abs2017 + cr_as_sbo2017 + hmda_sec_amount2017 + cr_secveh_ta2017 + cr_cds_purchased2017 + cr_sec_income2017 + cr_serv_fees2017

# Second order measurement model
ABSCDO =~ ABSCDO2011 + ABSCDO2012 + ABSCDO2013 + ABSCDO2014 + ABSCDO2015 + ABSCDO2016 + ABSCDO2017

# Second-order items factor
#MEASURES =~ cr_as_rmbs + cr_as_abs + cr_as_sbo + hmda_sec_amount + cr_secveh_ta + cr_cds_purchased + cr_sec_income + cr_serv_fees

# Items factors
cr_as_rmbs =~ cr_as_rmbs2011 + cr_as_rmbs2012 + cr_as_rmbs2013 + cr_as_rmbs2014 + cr_as_rmbs2015 + cr_as_rmbs2016 + cr_as_rmbs2017
cr_as_abs =~ cr_as_abs2011 + cr_as_abs2012 + cr_as_abs2013 + cr_as_abs2014 + cr_as_abs2015 + cr_as_abs2016 + cr_as_abs2017
cr_as_sbo =~ cr_as_sbo2011 + cr_as_sbo2012 + cr_as_sbo2013 + cr_as_sbo2014 + cr_as_sbo2015 + cr_as_sbo2016 + cr_as_sbo2017
hmda_sec_amount =~ hmda_sec_amount2011 + hmda_sec_amount2012 + hmda_sec_amount2013 + hmda_sec_amount2014 + hmda_sec_amount2015 + hmda_sec_amount2016 + hmda_sec_amount2017
cr_secveh_ta =~ cr_secveh_ta2011 + cr_secveh_ta2012 + cr_secveh_ta2013 + cr_secveh_ta2014 + cr_secveh_ta2015 + cr_secveh_ta2016 + cr_secveh_ta2017
cr_cds_purchased =~ cr_cds_purchased2011 + cr_cds_purchased2012 + cr_cds_purchased2013 + cr_cds_purchased2014 + cr_cds_purchased2015 + cr_cds_purchased2016 + cr_cds_purchased2017
cr_sec_income =~ cr_sec_income2011 + cr_sec_income2012 + cr_sec_income2013 + cr_sec_income2014 + cr_sec_income2015 + cr_sec_income2016 + cr_sec_income2017
cr_serv_fees =~ cr_serv_fees2011 + cr_serv_fees2012 + cr_serv_fees2013 + cr_serv_fees2014 + cr_serv_fees2015 + cr_serv_fees2016 + cr_serv_fees2017

# Fix covariances
ABSDCO ~~ 0*cr_as_rmbs + 0*cr_as_abs + 0*cr_as_sbo + 0*hmda_sec_amount + 0*cr_secveh_ta + 0*cr_cds_purchased + 0*cr_sec_income + 0*cr_serv_fees
cr_as_rmbs ~~ 0*cr_as_abs + 0*cr_as_sbo + 0*hmda_sec_amount + 0*cr_secveh_ta + 0*cr_cds_purchased + 0*cr_sec_income + 0*cr_serv_fees
cr_as_abs ~~ 0*cr_as_sbo + 0*hmda_sec_amount + 0*cr_secveh_ta + 0*cr_cds_purchased + 0*cr_sec_income + 0*cr_serv_fees
cr_as_sbo ~~ 0*hmda_sec_amount + 0*cr_secveh_ta + 0*cr_cds_purchased + 0*cr_sec_income + 0*cr_serv_fees
hmda_sec_amount ~~ 0*cr_secveh_ta + 0*cr_cds_purchased + 0*cr_sec_income + 0*cr_serv_fees
cr_secveh_ta ~~ 0*cr_cds_purchased + 0*cr_sec_income + 0*cr_serv_fees
cr_cds_purchased ~~ 0*cr_sec_income + 0*cr_serv_fees
cr_sec_income ~~ 0*cr_serv_fees
'

#fit <- sem(model, data = df_log, estimator = 'MLR', control=list(iter.max=1e5))
fit <- sem(model, data = df_log, missing = 'fiml', estimator = 'MLR', control=list(iter.max=1e5))
summary(fit, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)


# ---------------------------------------
# TSO model
#----------------------------------------
model <- '
# Repeated measurement model (trait factors)
ABSCDO2011 =~ 1*cr_as_rmbs2011 + a*cr_as_abs2011 + b*cr_as_sbo2011 + c*hmda_sec_amount2011 + d*cr_secveh_ta2011+ e*cr_cds_purchased2011 + f*cr_sec_income2011 + g*cr_serv_fees2011
ABSCDO2012 =~ 1*cr_as_rmbs2012 + a*cr_as_abs2012 + b*cr_as_sbo2012 + c*hmda_sec_amount2012 + d*cr_secveh_ta2012+ e*cr_cds_purchased2012 + f*cr_sec_income2012 + g*cr_serv_fees2012
ABSCDO2013 =~ 1*cr_as_rmbs2013 + a*cr_as_abs2013 + b*cr_as_sbo2013 + c*hmda_sec_amount2013 + d*cr_secveh_ta2013+ e*cr_cds_purchased2013 + f*cr_sec_income2013 + g*cr_serv_fees2013
ABSCDO2014 =~ 1*cr_as_rmbs2014 + a*cr_as_abs2014 + b*cr_as_sbo2014 + c*hmda_sec_amount2014 + d*cr_secveh_ta2014+ e*cr_cds_purchased2014 + f*cr_sec_income2014 + g*cr_serv_fees2014

# Second-order trait factor
ABSCDO =~ 1*ABSCDO2011 + 1*ABSCDO2012 + 1*ABSCDO2013 + 1*ABSCDO2014

# Remove all residual variables associated to latent variables
ABSCDO2011 ~~ 0*ABSCDO2011
ABSCDO2012 ~~ 0*ABSCDO2012
ABSCDO2013 ~~ 0*ABSCDO2013
ABSCDO2014 ~~ 0*ABSCDO2014

# Create occasion latent factors
occas2011 =~ 1*ABSCDO2011
occas2012 =~ 1*ABSCDO2012
occas2013 =~ 1*ABSCDO2013
occas2014 =~ 1*ABSCDO2014

# Create autoregressive paths
occas2012 ~ beta*occas2011
occas2013 ~ beta*occas2012
occas2014 ~ beta*occas2013

# Remove correlation latent exo factor
ABSCDO ~~ 0*occas2011

# Constrain residual variance
occas2011 ~~ occas2011
occas2012 ~~ zeta*occas2012
occas2013 ~~ zeta*occas2013
occas2014 ~~ zeta*occas2014

# Measurement error autocorrelations
cr_as_rmbs2011 ~~ cr_as_rmbs2012 + cr_as_rmbs2013 + cr_as_rmbs2014
cr_as_rmbs2012 ~~ cr_as_rmbs2013 + cr_as_rmbs2014
cr_as_rmbs2013 ~~ cr_as_rmbs2014

cr_as_abs2011 ~~ cr_as_abs2012 + cr_as_abs2013 + cr_as_abs2014
cr_as_abs2012 ~~ cr_as_abs2013 + cr_as_abs2014
cr_as_abs2013 ~~ cr_as_abs2014

cr_as_sbo2011 ~~ cr_as_sbo2012 + cr_as_sbo2013 + cr_as_sbo2014
cr_as_sbo2012 ~~ cr_as_sbo2013 + cr_as_sbo2014
cr_as_sbo2013 ~~ cr_as_sbo2014

hmda_sec_amount2011 ~~ hmda_sec_amount2012 + hmda_sec_amount2013 + hmda_sec_amount2014
hmda_sec_amount2012 ~~ hmda_sec_amount2013 + hmda_sec_amount2014
hmda_sec_amount2013 ~~ hmda_sec_amount2014

cr_secveh_ta2011 ~~ cr_secveh_ta2012 + cr_secveh_ta2013 + cr_secveh_ta2014
cr_secveh_ta2012 ~~ cr_secveh_ta2013 + cr_secveh_ta2014
cr_secveh_ta2013 ~~ cr_secveh_ta2014

cr_cds_purchased2011 ~~ cr_cds_purchased2012 + cr_cds_purchased2013 + cr_cds_purchased2014
cr_cds_purchased2012 ~~ cr_cds_purchased2013 + cr_cds_purchased2014
cr_cds_purchased2013 ~~ cr_cds_purchased2014

cr_sec_income2011 ~~ cr_sec_income2012 + cr_sec_income2013 + cr_sec_income2014
cr_sec_income2012 ~~ cr_sec_income2013 + cr_sec_income2014
cr_sec_income2013 ~~ cr_sec_income2014

cr_serv_fees2011 ~~ cr_serv_fees2012 + cr_serv_fees2013 + cr_serv_fees2014
cr_serv_fees2012 ~~ cr_serv_fees2013 + cr_serv_fees2014
cr_serv_fees2012cr_serv_fees2013 ~~ cr_serv_fees2014
'

fit <- sem(model, data = df_log, missing = 'fiml', estimator = 'MLR')
summary(fit, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)

# ---------------------------------------
# Latent Growth model
#----------------------------------------
model <-  '
# Measurement models where factor loadings are constraint across time
ABSCDO2006 =~ L1*cr_as_rmbs2006 + L2*cr_as_abs2006 + L3*hmda_sec_amount2006 + L4*cr_sec_income2006 + L5*cr_serv_fees2006 + L6*cr_cds_purchased2006
ABSCDO2007 =~ L1*cr_as_rmbs2007 + L2*cr_as_abs2007 + L3*hmda_sec_amount2007 + L4*cr_sec_income2007 + L5*cr_serv_fees2007 + L6*cr_cds_purchased2007
ABSCDO2008 =~ L1*cr_as_rmbs2008 + L2*cr_as_abs2008 + L3*hmda_sec_amount2008 + L4*cr_sec_income2008 + L5*cr_serv_fees2008 + L6*cr_cds_purchased2008

# Item intercepts, all freely estimated
cr_as_rmbs2006 ~ I1T1*1; cr_as_abs2006 ~ I2T1*1;  hmda_sec_amount2006 ~ I2T1*1;  cr_sec_income2006 ~ I2T1*1;  cr_serv_fees2006 ~ I2T1*1;  cr_cds_purchased2006 ~ I2T1*1;  
cr_as_rmbs2007 ~ I1T2*1; cr_as_abs2007 ~ I2T2*1;  hmda_sec_amount2007 ~ I2T2*1;  cr_sec_income2007 ~ I2T2*1;  cr_serv_fees2007 ~ I2T2*1;  cr_cds_purchased2007 ~ I2T2*1;
cr_as_rmbs2008 ~ I1T3*1; cr_as_abs2008 ~ I2T3*1;  hmda_sec_amount2008 ~ I2T3*1;  cr_sec_income2008 ~ I2T3*1;  cr_serv_fees2008 ~ I2T3*1;  cr_cds_purchased2008 ~ I2T3*1;

# Residual variances, all freely estimatable
cr_as_rmbs2006 ~~ E1T1*cr_as_rmbs2006; cr_as_abs2006 ~~ E2T1*cr_as_abs2006;  hmda_sec_amount2006 ~~ E2T1*hmda_sec_amount2006;  cr_sec_income2006 ~~ E2T1*cr_sec_income2006;  cr_serv_fees2006 ~~ E2T1*cr_serv_fees2006;  cr_cds_purchased2006 ~~ E2T1*cr_cds_purchased2006;
cr_as_rmbs2007 ~~ E1T2*cr_as_rmbs2007; cr_as_abs2007 ~~ E2T2*cr_as_abs2007;  hmda_sec_amount2007 ~~ E2T2*hmda_sec_amount2007;  cr_sec_income2007 ~~ E2T2*cr_sec_income2007;  cr_serv_fees2007 ~~ E2T2*cr_serv_fees2007;  cr_cds_purchased2007 ~~ E2T2*cr_cds_purchased2007;
cr_as_rmbs2008 ~~ E1T3*cr_as_rmbs2008; cr_as_abs2006 ~~ E2T3*cr_as_abs2008;  hmda_sec_amount2008* ~~ E2T3*hmda_sec_amount2008;  cr_sec_income2008 ~~ E2T3*cr_sec_income2008;  cr_serv_fees2008 ~~ E2T3*cr_serv_fees2008;  cr_cds_purchased2008 ~~ E2T3*cr_cds_purchased2008;

# Residual covariances, all freely estimatable
cr_as_rmbs2006 ~~ C1T12*cr_as_rmbs2007 + C1T13*cr_as_rmbs2008; cr_as_rmbs2007 ~~ C1T23*cr_as_rmbs2008;
cr_as_abs2006 ~~ C2T12*cr_as_abs2007 + C2T13*cr_as_abs2008; cr_as_abs2007 ~~ C2T23*cr_as_abs2008;
hmda_sec_amount2006 ~~ C3T12*hmda_sec_amount2007 + C3T13*hmda_sec_amount2008; hmda_sec_amount2007 ~~ C3T23*hmda_sec_amount2008;
cr_sec_income2006 ~~ C4T12*cr_sec_income2007 + C4T13*cr_sec_income2008; cr_sec_income2007 ~~ C4T23*cr_sec_income2008;
cr_serv_fees2006 ~~ C5T12*cr_serv_fees2007 + C5T13*cr_serv_fees2008; cr_serv_fees2007 ~~ C5T23*cr_serv_fees2008;
cr_cds_purchased2006 ~~ C6T12*cr_cds_purchased2007 + C6T13*cr_cds_purchased2008; cr_cds_purchased2007 ~~ C6T23*cr_cds_purchased2008;

# Fix factor variance
ABSCDO2006 ~~ 1* ABSCDO2006
ABSCDO2007 ~~ ABSCDO2007
ABSCDO2008 ~~ ABSCDO2008

# Fix factor mean
ABSCDO2006 ~ 0
ABSCDO2007 ~ 0 
ABSCDO2008 ~ 0

# Factor covariances, freely estimatable 
ABSCDO2006 ~~ ABSCDO2007 + ABSCDO2008
ABSCDO2007 ~~ ABSCDO2008
'

fit <- cfa(model, data = df_log, estimator = 'MLR', meanstructure = TRUE)
summary(fit, fit.measures=TRUE, standardized = TRUE, rsquare = TRUE)



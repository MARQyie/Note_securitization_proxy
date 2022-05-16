
library(lavaan)
library(rlist)
library(evaluate)
library(doMC)
registerDoMC(cores=24)
library(foreign)

# Set number of bootstraps
BB1=1000
BB2=100

col_idx=4:12

model_1f <-  '
              # Measurement model
              SEC =~ cr_as_rmbs + cr_as_abs + hmda_sec_amount + cr_secveh_ta  + cr_cds_purchased + cr_abcp_ta + cr_abcp_uc_own + cr_abcp_ce_own + cr_abcp_uc_oth

              # Error covariances
              cr_abcp_uc_own ~~ cr_abcp_uc_oth
              cr_abcp_ta ~~ cr_secveh_ta + cr_abcp_ce_own
'

data=read.csv("df_sec_note_binary_balance.csv")
unique_banks=unique(data$IDRSSD)


  fit <- cfa(model_1f,
              data = data,
              estimator = 'WLSMV',
              ordered = TRUE,
              se = 'none',
              warn = FALSE)
  params.original <- coef(fit)



boot.res<-foreach(b=1:BB1) %:% foreach(bb=1:BB2, .combine=rbind) %dopar% {

set.seed(BB1*b+bb)

print(c(b,bb))

# Randomly sample bank IDRSSDs
  sample_banks <- sample(unique_banks, size=length(unique_banks), replace=T)

# Fetch all years for each randomly picked bank and rbind
  data_resampled <- do.call(rbind, lapply(sample_banks, function(x)  data[data$IDRSSD==x,]))

# Check if one of the columns is zero only. If true resample data.
# Continue until statement is satisfied
  data_condition <- lapply(data_resampled[,col_idx], function(x) all(x == 0))

  counter=0
  while (any(data_condition == T) | sum(is.na(data_condition))!=0 | is.null(data_condition)){
  counter=counter+1
    sample_banks <- sample(unique_banks, size=length(unique_banks), replace=T )
    data_resampled <- do.call(rbind, lapply(sample_banks, function(x)  data[data$IDRSSD==x,]))
    data_condition <- lapply(data_resampled[,col_idx], function(x) all(x == 0))
  }

# Fit CFA


  fit <- try(cfa(model_1f,
              data = data_resampled,
              estimator = 'WLSMV',
              ordered = TRUE,
              se = 'none',
              warn = FALSE), silent=TRUE) 
  params.boot <- try(coef(fit), silent=TRUE)

if(!is.numeric(params.boot)){
params.boot=matrix(0,length(params.original),1)
}


c(params.boot,b,bb,counter==0)

}

save(list = ls(all=TRUE), file = paste("bootstrap_bias_corr.RData", sep=""))

#load("bootstrap_bias_corr.RData")

############################################

# check hoevaak coeff 0 (als fout gaat)

#bias.corrected=colMeans(boot.res[[1]])

#zeros.mat=matrix(0,BB1,1)

#for(b in 1:BB1){
#zeros.mat[b]=sum(rowMeans(boot.res[[b]][,1:(length(bias.corrected)-3)])==0)
#}

#sum(zeros.mat)

############################################

bias.corrected=colMeans(boot.res[[1]])
# bias corrected
# ignore last two numbers

# bias
bias=bias.corrected[1:(length(bias.corrected)-3)]-params.original

# relative bias (doen we nu niks mee)
bias.relative=100*(bias/params.original)

# zero condition info
# 100*mean(boot.res[[1]][,ncol(boot.res[[1]])]) # fraction > 1 sample necessary for 0 condition

# sd calculation
bias.corrected.mat=matrix(0,BB1,length(params.original))

for(b in 1:BB1){
bias.corrected.mat[b,]=colMeans(boot.res[[b]])[1:(length(bias.corrected)-3)]
}

sd.bias.corrected=apply(bias.corrected.mat,2,FUN="sd")

# conf. interval
q025.bias.corrected=apply(bias.corrected.mat,2,FUN="quantile",0.025)

q975.bias.corrected=apply(bias.corrected.mat,2,FUN="quantile",0.975)

write.csv(cbind(bias.corrected[1:(length(bias.corrected)-3)], sd.bias.corrected[1:(length(bias.corrected)-3)],
q025.bias.corrected[1:(length(bias.corrected)-3)], q975.bias.corrected[1:(length(bias.corrected)-3)]), 
'Results_panelbootstrap_correction_balanced.csv')












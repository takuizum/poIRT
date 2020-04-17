# generalized item response model
library(irtoys)
resp <- Scored
datastan <- list(N=nrow(resp), M=ncol(resp), y=resp)
model_G2PL <- stan_model("src/girtmodel.stan")
res_G2PL <- sampling(model_G2PL, data = datastan, iter = 1000, warmup = 200, init = 0, cores = 4)
res_G2PL@model_pars
all(summary(res_G2PL)$summary[,"Rhat"]<1.10)

# evaluation model fit
wbic <- function(log_lik) {
  wbic <- -mean(apply(log_lik, 1, sum))
  return(wbic)
}
log_lik <- rstan::extract(res_G2PL)$log_lik
wbic(log_lik)

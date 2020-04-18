# generalized item response model
library(irtoys)
library(rstan)
library(tidyverse)

resp <- Scored
datastan <- list(N=nrow(resp), M=ncol(resp), y=resp)
model_G2PL <- stan_model("src/girtmodel.stan")
res_G2PL <- sampling(model_G2PL, data = datastan, iter = 2000, warmup = 200, init = 0, cores = 4)
res_G2PL@model_pars
all(summary(res_G2PL)$summary[,"Rhat"]<1.10)

# evaluation model fit
wbic <- function(log_lik) {
    wbic <- -mean(apply(log_lik, 1, sum))
    return(wbic)
}
waic <- function(log_lik){
    log_lik <- apply(log_lik, c(1, 2), sum)
    T_n <- mean(-log(colMeans(exp(log_lik))))
    V_n <- mean(apply(log_lik, 2, var))
    T_n + V_n
}
log_lik <- rstan::extract(res_G2PL)$log_lik
wbic(log_lik)
waic(log_lik)

# Check by shiny intaractive module.
library(shinystan)
launch_shinystan(res_G2PL)

# Person outlier index
phi <- 
    rstan::extract(res_G2PL)$phi %>% 
    `colnames<-`(paste0("phi", 1:ncol(.))) %>% 
    as_tibble %>% 
    map_df(
        function(x) {
            m = mean(x)
            v = sd(x)
            tibble(eap = m, sd = v)
        }
    )

theta <- 
    rstan::extract(res_G2PL)$theta %>% 
    `colnames<-`(paste0("phi", 1:ncol(.))) %>% 
    as_tibble %>% 
    map_df(
        function(x) {
            m = mean(x)
            v = sd(x)
            tibble(eap = m, sd = v)
        }
    )

plot(theta$eap, phi$eap)

# item parameters
rstan::extract(res_G2PL)$b %>% apply(2, mean)
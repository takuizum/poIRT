using Statistics

LogLikelihood = LogLikeMCMC(data, θ, β, ω, ϕ, 2000)
@show waic(LogLikelihood);
@show wbic(LogLikelihood);

# fit analysis
# Marginal probability of outlier indicator
sum(mean(ω, dims = 1)[:] .> 1/2)
# EAP of person outlier parameter
sum(mean(ϕ, dims = 1) .> 1/2)

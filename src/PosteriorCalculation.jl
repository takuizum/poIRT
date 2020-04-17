using Statistics

LogLikelihood = LogLikeMCMC(data, θ, β, ω, ϕ)
waic(LogLikelihood)
@show wbic(LogLikelihood)

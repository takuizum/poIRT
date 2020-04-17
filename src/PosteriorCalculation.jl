using Statistics

LogLikelihood = LogLikeMCMC(data, θ, β, ω, ϕ)
@show waic(LogLikelihood);
@show wbic(LogLikelihood);

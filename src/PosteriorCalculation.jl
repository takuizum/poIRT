using Statistics

@code_warntype LogLik(data, θ[1,:], β[1,:], ω[1,:,:], ϕ[1,:])
LogLikeMCMC(data, θ, β, ω, ϕ)
LogLikelihood = LogLikeMCMC(data, θ, β, ω, ϕ)
sum(LogLikelihood, dims = 2)
using Plots
histogram(LogLikelihood)


waic(LogLikelihood)
@show wbic(LogLikelihood)
sum(data, dims = 2)


LogLikelihood[-Inf .< LogLikelihood .< Inf]
map(i -> -log(mean(exp.(LogLikelihood[i,:]))), 1:size(LogLikelihood,1))

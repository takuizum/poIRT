using Distributions, LinearAlgebra

# sample data

using RCall
R"""
library(irtoys)
data <- Scored
"""
@rget data
data = convert(Matrix{Int64}, data)

include("utils.jl")

y = copy(data)
N, J = size(y)
yast = zeros(Float64, N, J) # latent response variable
yast_long = reshape(yast, N*J, 1)
ϕ = rand(Beta(1, 5), N)
ω = zeros(Bool, N, J) # latent discrete outlier variable
z = rand(Gamma(2, 2), (N, J)) # latent variables to facilitate Gibbs sampling
V = diagm(fill(1/10^2, N + J))
X = convert(Matrix{Float64}, GenerateDummyX(y))
θ = rand(Normal(0, 100), N)
β = rand(Normal(0, 100), J)
ζ = zeros(Float64, N*J)

# Full
@code_warntype FC1(y, yast_long, θ, β, z, ω, ζ)
@code_warntype FC2(X, ζ, V, θ, β, yast_long)
z = rand(Gamma(2, 2), (N, J))
@code_warntype FC4(y, z, θ, β, ω)
@code_warntype FC5(ϕ, ω)

# Full GIbbs sampler
ϕ, ω, θ, β = GibbsSampler(data, S = 2000)


using Plots
plot(ϕ[:,1:10])
plot(ω[:,1:18])
plot(θ[101:end,1:10])
plot(β[101:end,1:18])

map(i -> var(θ[:,i]), 1:size(θ, 2))
map(i -> -1.702mean(β[:,i]), 1:size(β, 2))
map(i -> mean(ω[:,i]), 1:18)

using StatsPlots
density(θ[101:end,1:10], xlims = (-4, 4))
density(β[101:end,begin:end])
density(ϕ[101:end,1:10])

scatter(ω[:,1:18])

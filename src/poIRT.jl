# 必要なパッケージ類はtomlに書いてあります。
# 実行前にREPLで'] activate .'を実行してください。
# RCallの呼び出しには，別途Rのインストールと環境設定が必要になります。
# 下記を参照して，設定してください。
# http://juliainterop.github.io/RCall.jl/stable/installation.html


using Distributions, LinearAlgebra

# sample data

using RCall
R"""
# install.packages("irtoys")
library(irtoys)
data <- Scored
"""
@rget data;
data = convert(Matrix{Int64}, data);

include("src/utils.jl")

# Optimization 
@code_warntype FC1(y, yast_long, θ, β, z, ω, ζ)
@code_warntype FC2(X, ζ, V, θ, β, yast_long)
z = rand(Gamma(2, 2), (N, J))
@code_warntype FC4(y, z, θ, β, ω)
@code_warntype FC5(ϕ, ω)

# Full GIbbs sampler
ϕ, ω, θ, β= GibbsSampler(data, S = 2000);


using Plots
plot(ϕ[:,1:10])
plot(ω[:,1:18])
plot(θ[101:end,1:10])
plot(β[101:end,1:18])

map(i -> var(θ[:,i]), 1:size(θ, 2))
map(i -> -1.702mean(β[:,i]), 1:size(β, 2))
map(i -> mean(ω[:,i]), 1:18)

using StatsPlots
density(θ[201:end,1:10], xlims = (-4, 4))
density(β[201:end,begin:end])
density(ϕ[201:end,1:10])

scatter(ω[:,1:18])

"""
    Generate dummy covariate matrix X
"""
function GenerateDummyX(y)
    n, J = size(y)
    X = zeros(Int64, n*J, n+J)
    item_range = [1:1:J;]
    for i in 1:n
        X[item_range, i] .= 1
        for (j, v) in enumerate(item_range)
            jj = n + j
            X[v, jj] = y[i, j] == 1 ? -1 : 0
            X[v, jj] = -1
        end
        item_range .+= J
    end
    return X
end

# FC1 : Sampling latent response y* (and colculate ζ vector)
function FC1(y, yast_long, θ, β, z, ω, ζ)
    n, J = size(y)
    κ = 10^3
    t = 0
    for i in 1:n, j in 1:J
        t += 1
        ζ[t] = z[i, j]*κ^ω[i, j]
        if y[i, j] == 1
            yast_long[t] = rand(truncated(Normal(θ[i] - β[j], ζ[t]),  0, Inf))
        elseif y[i, j] == 0
            yast_long[t] = rand(truncated(Normal(θ[i] - β[j], ζ[t]), -Inf, 0))
        end
    end
end
# FC2 : Sampling θ and β from multivariate normal distribution widh diagonal Σ matrix
function FC2(X, ζ, V, θ, β, yast_long)
    n = length(θ);
    invΣ = Diagonal(1 ./ ζ);
    V′ = inv(X'invΣ*X + V)
    m = V′*(X'invΣ*yast_long[:]);
    θβ = rand(MvNormal(m, Symmetric(V′)));
    θ[:] = θβ[1:n];
    β[:] = θβ[n+1:end];
end
# FC4 : Sampling ω from item response function
function FC4(y, z, θ, β, ω)
    n, J = size(y)
    κ = 10^3
    for i in 1:n, j in 1:J
        if y[i,j] == 1
            p0 = cdf(Normal(0, z[i, j]), θ[i] - β[j])
            p1 = cdf(Normal(0, z[i, j]*κ), θ[i] - β[j])
        elseif y[i, j] == 0
            p0 = 1.0 - cdf(Normal(0, z[i, j]), θ[i] - β[j])
            p1 = 1.0 - cdf(Normal(0, z[i, j]*κ), θ[i] - β[j])
        end
        p = (p0^(1 - ω[i, j]) * p1^ω[i, j]) / (p0 + p1)
        if !isequal(p, NaN)
            ω[i, j] = rand(Bernoulli(p), 1)[1]
        else
            ω[i, j] = 0
            @warn("Nan was returned")
        end
    end
end
# FC5 : Sampling ϕ from posterior Beta distributution
function FC5(ϕ, ω)
    n, J = size(ω)
    for i in 1:n
        Ω = sum(ω[i, :])
        ϕ[i] = rand(Beta(1 + Ω, 5 + J - Ω), 1)[1]
    end
end
# full
function GibbsSampler(data; S = 1000)
    # init
    y = copy(data)
    N, J = size(y)
    yast = zeros(Float64, N, J) # latent response variable
    yast_long = reshape(yast, N*J, 1)
    ϕ = rand(Beta(1, 5), N)
    ω = zeros(Bool, N, J) # latent discrete outlier variable
    z = rand(Gamma(2, 2), (N, J)) # latent variables to facilitate Gibbs sampling
    V = Diagonal(fill(1/10^2, N + J)) # Prior (accuracy) of θ and β
    X = convert(Matrix{Float64}, GenerateDummyX(y))
    θ = rand(Normal(0, 100), N)
    β = rand(Normal(0, 100), J)
    ζ = zeros(Float64, N*J)
    # Ensure the matrix to save the sampled value
    ϕres = Matrix{Float64}(undef, S, N)
    θres = Matrix{Float64}(undef, S, N)
    βres = Matrix{Float64}(undef, S, J)
    ωres = Array{Bool, 3}(undef, S, N, J)
    # sample
    # Threads.@threads for s in 1:S
    for s in 1:S
        print("Sampling $s...\r")
        FC1(y, yast_long, θ, β, z, ω, ζ)
        FC2(X, ζ, V, θ, β, yast_long)
        z = rand(Gamma(2, 2), (N, J))
        FC4(y, z, θ, β, ω)
        FC5(ϕ, ω)
        # save
        ϕres[s, :] = ϕ[:]
        ωres[s, :, :] = ω
        μ = mean(θ[:])
        σ = sqrt(var(θ[:]))
        θ[:] = (θ[:] .- μ) ./ σ
        β[:] = (β[:] .- μ) ./ σ
        θres[s, :] = θ[:]
        βres[s, :] = β[:]
    end
    return (ϕres, ωres, θres, βres);
end
# Model likekihood
F(x, z) = cdf(Normal(0, z), x)
lnp = function(y, θ, β, ω, ϕ)
    κ = 10^3
    l = zero(Float64)
    l += y == 1 ? log(F(θ-β, κ^ω)) : log(1 - F(θ, κ^ω))
    l += ω ? log(ϕ) : log(1-ϕ)
    return l
end
function LogLik(y, θ, β, ω, ϕ)
    J = length(y)
    l = zero(typeof(θ))
    for j in 1:J
        l += lnp(y[j], θ, β[j], ω[j], ϕ)
    end
    return l
end

"""
    Calculate log likekihood via MCMC samples
"""
function LogLikeMCMC(y, θ, β, ω, ϕ, warmup = 100)
    S, N = size(θ)
    l = zeros(Float64, S-warmup, N)
    for (i, s) in enumerate(warmup+1:S)
        for n in 1:N
            l[i,n] = LogLik(y[n,:], θ[s,n], β[s,:], ω[s,n,:], ϕ[s,n])
        end
    end
    return l
end

function waic(log_lik)
    T_n = map(i -> -log(mean(exp.(log_lik[i,:]))), 1:size(log_lik,1))
    V_n = map(i -> var((log_lik[i, .!isnan.(log_lik[i,:])])), 1:size(log_lik,1))
    mean(T_n[.!isnan.(T_n)]) + mean(V_n[.!isnan.(V_n)])
end
wbic(log_lik) = -mean(map(i -> sum(log_lik[i, .!isnan.(log_lik[i,:])]), size(log_lik, 1)))

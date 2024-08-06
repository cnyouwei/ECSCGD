using LinearAlgebra
using Random
using Distributions

function cholesky_decomposition(Sigma::Matrix{Float64})
    """
    Computes the Cholesky decomposition of a positive definite matrix.
    """
    L = cholesky(Sigma).L
    return L
end

function multi_normal_gen(mu::Vector{Float64}, L::Matrix{Float64})
    """
    Generates a sample from a multivariate normal distribution.
    """
    return mu .+ L * randn(length(mu))
end

function project_to_simplex(v::Vector{Float64})
    """
    Projects a vector v onto the probability simplex.
    """
    n = length(v)
    if sum(v) == 1 && all(v .>= 0)
        return v
    end

    u = sort(v, rev=true)
    cumsum_u = cumsum(u)
    rho = findlast(x -> x > 0, u .* (1:n) .> (cumsum_u .- 1))
    theta = (cumsum_u[rho] - 1) / rho
    w = max.(v .- theta, 0)
    
    return w
end

function generate_random_positive_definite_matrix(n::Int)
    """
    Generates a random n x n positive definite matrix.
    """
    # Generate a random matrix
    A = rand(n, n)
    
    # Compute the product of A and its transpose
    positive_definite_matrix = A * A'
    
    return positive_definite_matrix
end

function generate_toeplitz_matrix(vals::Vector{Float64})
    """
    Generates a Toeplitz matrix given a vector of values.
    """
    n = length(vals)
    toeplitz_matrix = Matrix{Float64}(undef, n, n)
    for i in 1:n
        for j in 1:n
            toeplitz_matrix[i, j] = vals[abs(i - j) + 1]
        end
    end
    return toeplitz_matrix
end

function objective_function(x::Vector{Float64}, mu::Vector{Float64}, Sigma::Matrix{Float64}, gamma::Float64)
    """
    Computes the objective function value = mean - gamma * fourth_central_moment.
    """
    portfolio_mean = mu' * x
    portfolio_M4 = 3 * (x' * Sigma * x)^2
    return portfolio_mean - gamma * portfolio_M4
end

function gradient(x::Vector{Float64}, mu::Vector{Float64}, Sigma::Matrix{Float64}, gamma::Float64)
    """
    Computes the gradient of the objective function.
    """
    grad = -mu .+ 12 * gamma * (x' * Sigma * x) * (Sigma * x)
    return grad
end

function CVaR(x::Vector{Float64}, mu::Vector{Float64}, Sigma::Matrix{Float64}, delta::Float64=0.05)
    """
    Computes the Conditional Value at Risk (CVaR) of the negative mean return.
    """
    portfolio_loss = -mu' * x
    portfolio_std = sqrt(x' * Sigma * x)
    phi_alpha = pdf(Normal(0, 1), quantile(Normal(0, 1), delta))
    cvar = portfolio_loss + portfolio_std * phi_alpha / (1 - delta)
    return cvar
end

function CVaR_gradient(mu::Vector{Float64}, Sigma::Matrix{Float64}, x::Vector{Float64}, delta::Float64)
    """
    Computes the gradient of the CVaR function.
    """
    mu = reshape(mu, :, 1)
    x = reshape(x, :, 1)

    phi_delta = pdf(Normal(0, 1), quantile(Normal(0, 1), delta))
    c = phi_delta / (1 - delta)
    sqrt_term = sqrt(x' * Sigma * x)

    gradient = -mu .+ c .* (Sigma * x) / sqrt_term

    return reshape(gradient, :)
end

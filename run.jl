include("gradient_descent_solvers.jl")
include("EC_SCGD.jl")
using Dates
using Printf
using NPZ

NUM_ITER = 1

for i in 1:NUM_ITER

    n = 10  # dimension
    gamma = 0.5  # coefficient for the fourth central moment
    delta = 0.05  # confidence level of the CVaR

    # mu = rand(n) * 2 .- 1
    # Sigma = Matrix{Float64}(I(n))
    # case_name = "iid_"

    rho = 0.5
    mu = rand(n) * 2 .- 1
    vals = [rho^(min((i-1), n-(i-1))) for i in 1:n]
    # println(vals)
    Sigma = generate_toeplitz_matrix(vals)
    case_name = "toeplitz_"

    # mu = rand(n)
    # Sigma = generate_random_positive_definite_matrix(n)
    # case_name = "random_"

    # decompose the covariance matrix for generating correlated multivariate normal
    L = cholesky_decomposition(Matrix(Sigma))

    # parameter within 0,1 to adjust reduction of CVaR, eps = 1 means full reduction to the smallest possible CVaR
    eps = 0.4   

    num_iter_oracle = Int(1e7)   # max number of iterations for oracle solvers
    num_iter_EC_SCGD = Int(1e7)  # max number of iterations for EC-SCGD
    print_interval = 100         # number of prints when running EC-SCGD

    run_SCGD_alg = false
    run_EC_SCGD_alg = true

    time = Dates.format(Dates.now(), "yyyymmdd_HH_MM_SS")
    mkpath("./Data/")
    f_name = "./Data/" * case_name * "d" * string(n) * "_" * time

    # Run oracle solvers
    obj_unconstrained, cvar_unconstrained, x_unconstrained = run_unconstrained_oracleGD(mu, Sigma, gamma, delta, num_iterations = num_iter_oracle)

    obj_min_cvar, cvar_min, x_min_cvar = run_minimizeCVaR_oracleGD(mu, Sigma, delta, gamma, num_iterations = num_iter_oracle)

    xi = (1-eps) * cvar_unconstrained + eps * cvar_min

    obj_cvar_constrained, cvar_cvar_constrained, x_cvar_constrained = run_CVaRconstrained_oracleGD(mu, Sigma, gamma, delta, xi, num_iterations = num_iter_oracle)

    open(f_name * ".txt", "w") do f
        write(f, "n = $n, gamma = $gamma.\n")
        write(f, "mu = $(mu)\n")
        write(f, "Sigma = $(Sigma)\n\n")
        write(f, "Optimal value (unconstrained): $(obj_unconstrained)\n")
        write(f, "Optimal CVaR (unconstrained): $(cvar_unconstrained)\n")
        write(f, "Optimal x (unconstrained): $(x_unconstrained)\n\n")
        write(f, "delta = $delta.\n")
        write(f, "Min CVaR: $(cvar_min)\n")
        write(f, "Optimal x (minCVaR): $(x_min_cvar)\n\n")
        write(f, "Xi = $xi.\n")
        write(f, "Optimal value (CVaR constrained): $(obj_cvar_constrained)\n")
        write(f, "Optimal CVaR (CVaR constrained): $(cvar_cvar_constrained)\n")
        write(f, "Optimal x (CVaR constrained): $(x_cvar_constrained)\n")
        write(f, "Num iterations = $num_iter_EC_SCGD.\n")
    end

    # Run EC-SCGD
    if run_SCGD_alg
        SCGD_x, SCGD_value, SCGD_cvar, SCGD_recording_idx, SCGD_obj_path = run_SCGD(mu, Sigma, L, gamma, num_iterations = num_iter_EC_SCGD, print_interval = print_interval)
        plot_SCGD(SCGD_recording_idx, SCGD_obj_path, obj_unconstrained, num_iterations=num_iter_EC_SCGD, file_name = f_name)

        NPZ.npzwrite(f_name * "_SCGD.npz", Dict(
            "dim" => [n],
            "iterations" => [num_iter_EC_SCGD],
            "gamma" => [gamma],
            "mu" => convert(Vector{Float64}, mu),
            "Sigma" => convert(Matrix{Float64}, Sigma),
            "optimal_value" => [obj_unconstrained],
            "optimal_cvar" => [cvar_unconstrained],
            "optimal_x" => convert(Vector{Float64}, x_unconstrained),
            "SCGD_x" => convert(Vector{Float64}, SCGD_x),
            "SCGD_value" => [SCGD_value],
            "SCGD_obj_path" => convert(Vector{Float64}, SCGD_obj_path)))
    end
        
    if run_EC_SCGD_alg
        EC_SCGD_x, EC_SCGD_value, EC_SCGD_CVaR, EC_SCGD_recording_idx, EC_SCGD_obj_path, EC_SCGD_cvar_path = run_EC_SCGD(mu, Sigma, L, gamma, delta, xi, num_iterations = num_iter_EC_SCGD, print_interval = print_interval)
        plot_EC_SCGD(EC_SCGD_recording_idx, EC_SCGD_obj_path, EC_SCGD_cvar_path, obj_cvar_constrained, xi, num_iterations = num_iter_EC_SCGD, file_name = f_name)
        NPZ.npzwrite(f_name * ".npz", Dict(
            "dim" => [n],
            "iterations" => [num_iter_EC_SCGD],
            "gamma" => [gamma],
            "delta" => [delta],
            "xi" => [xi],
            "mu" => convert(Vector{Float64}, mu),
            "Sigma" => convert(Matrix{Float64}, Sigma),
            "optimal_value" => [obj_cvar_constrained],
            "optimal_cvar" => [cvar_cvar_constrained],
            "optimal_x" => convert(Vector{Float64}, x_cvar_constrained),
            "EC_SCGD_x" => convert(Vector{Float64}, EC_SCGD_x),
            "EC_SCGD_value" => [EC_SCGD_value],
            "EC_SCGD_CVaR" => [EC_SCGD_CVaR],
            "EC_SCGD_recording_idx" => convert(Vector{Float64}, EC_SCGD_recording_idx),
            "EC_SCGD_obj_path" => convert(Vector{Float64}, EC_SCGD_obj_path),
            "EC_SCGD_cvar_path" => convert(Vector{Float64}, EC_SCGD_cvar_path)))
    end
end
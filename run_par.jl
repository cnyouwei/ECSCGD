using Distributed
addprocs(Sys.CPU_THREADS - 1)  # Add workers (adjust as needed)

@everywhere begin
    include("gradient_descent_solvers.jl")
    include("EC_SCGD.jl")
    using NPZ
    using Dates
    using Printf
end

@everywhere function run_iteration(i, f_name_base, mu, Sigma, L, gamma, delta, xi, num_iter_EC_SCGD, print_interval, run_SCGD_alg, run_EC_SCGD_alg, obj_unconstrained, cvar_unconstrained, x_unconstrained, obj_cvar_constrained, cvar_cvar_constrained, x_cvar_constrained)
    f_name = f_name_base * "_iter" * string(i)
    if run_SCGD_alg
        SCGD_x, SCGD_value, SCGD_cvar, SCGD_recording_idx, SCGD_obj_path = run_SCGD(mu, Sigma, L, gamma, num_iterations = num_iter_EC_SCGD, print_interval = print_interval)
        plot_SCGD(SCGD_recording_idx, SCGD_obj_path, obj_unconstrained, num_iterations=num_iter_EC_SCGD, file_name = f_name)
        NPZ.npzwrite(f_name * "_SCGD.npz", Dict(
            "dim" => [length(mu)],
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
        EC_SCGD_x, EC_SCGD_value, EC_SCGD_CVaR, EC_SCGD_recording_idx, EC_SCGD_obj_path, EC_SCGD_cvar_path = run_EC_SCGD(mu, Sigma, L, gamma, delta, xi, num_iterations = num_iter_EC_SCGD, print_interval = print_interval, obj_cvar_constrained = obj_cvar_constrained, cvar_cvar_constrained = cvar_cvar_constrained)
        NPZ.npzwrite(f_name * ".npz", Dict(
            "dim" => [length(mu)],
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
    return nothing
end

# Setup parameters as in your code:
NUM_ITER = 10
n = 250
gamma = 0.5
delta = 0.05

mu = rand(n) * 2 .- 1
Sigma = Matrix{Float64}(I(n))
case_name = "iid_"

# rho = 0.5
# mu = rand(n) * 2 .- 1
# vals = [rho^(min((i-1), n-(i-1))) for i in 1:n]
# # println(vals)
# Sigma = generate_toeplitz_matrix(vals)
# case_name = "toeplitz_"

L = cholesky(Sigma).L
eps = 0.4

num_iter_oracle = Int(1e7)   # max number of iterations for oracle solvers
num_iter_EC_SCGD = Int(1e7)  # max number of iterations for EC-SCGD
print_interval = 1000         # number of prints when running EC-SCGD

run_SCGD_alg = false
run_EC_SCGD_alg = true

# Run oracle solvers
obj_unconstrained, cvar_unconstrained, x_unconstrained = run_unconstrained_oracleGD(mu, Sigma, gamma, delta, num_iterations = num_iter_oracle)

obj_min_cvar, cvar_min, x_min_cvar = run_minimizeCVaR_oracleGD(mu, Sigma, delta, gamma, num_iterations = num_iter_oracle)

xi = (1-eps) * cvar_unconstrained + eps * cvar_min

obj_cvar_constrained, cvar_cvar_constrained, x_cvar_constrained = run_CVaRconstrained_oracleGD(mu, Sigma, gamma, delta, xi, num_iterations = num_iter_oracle)

mkpath("./Data/")
time_str = Dates.format(Dates.now(), "yyyymmdd_HH_MM")
f_name_base = "./Data/$(case_name)d$(n)_$(time_str)"

open(f_name_base * ".txt", "w") do f
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


# Run iterations in parallel
pmap(i -> run_iteration(i, f_name_base, mu, Sigma, L, gamma, delta, xi, num_iter_EC_SCGD, print_interval,
                        run_SCGD_alg, run_EC_SCGD_alg, obj_unconstrained, cvar_unconstrained, x_unconstrained,
                        obj_cvar_constrained, cvar_cvar_constrained, x_cvar_constrained), 1:NUM_ITER)

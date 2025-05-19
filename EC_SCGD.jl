include("Utils.jl")

"""
    SCGD for unconstrained stochastic compositional optimization
"""
function SO_f2_unconstrained(x, mu, Sigma, L)
    w_2 = multi_normal_gen(mu, Matrix(L))
    z = dot(x, w_2)
    return vcat(x, [z])
end

function SO_f1_gradient_unconstrained(y, mu, Sigma, L, gamma)
    w_1 = multi_normal_gen(mu, Matrix(L))
    x = y[1:end-1]
    z = y[end]
    diff = dot(w_1, x) - z
    return vcat(4 * gamma * (diff)^3 .* w_1, [-1 - 4 * gamma * (diff)^3])
end

function SO_f2_gradient_unconstrained(mu, Sigma, L)
    I_d = I(length(mu))
    w_2 = multi_normal_gen(mu, Matrix(L))
    return hcat(I_d, w_2)
end

function iteration_SCGD(x, y, mu, Sigma, L, gamma, eta, tau)
    y_new = (SO_f2_unconstrained(x, mu, Sigma, L) + tau * y) / (1 + tau)
    f2_gradient = SO_f2_gradient_unconstrained(mu, Sigma, L)
    f1_gradient = SO_f1_gradient_unconstrained(y_new, mu, Sigma, L, gamma)
    x_new = project_to_simplex(x - f2_gradient * f1_gradient / eta)
    return x_new, y_new
end

function run_SCGD(mu, Sigma, L, gamma; eta=1e4, tau=1e4, num_iterations=Int(1e7), print_interval = 100, delta = 0.05)
    x = ones(length(mu)) / length(mu)
    y = zeros(length(mu) + 1)
    x_cumulative = zeros(length(mu))
    obj_path = []
    recording_idx = []

    record_log10_UB = log10(num_iterations)
    record_log10_LB = 0
    record_delta = 0.05
    record_counter = record_log10_LB
    
    for i in 1:num_iterations
        eta = max(5e2, 5*sqrt(i))
        tau = max(2e2, i/50)
        x, y = iteration_SCGD(x, y, mu, Sigma, L, gamma, eta, tau)
        x_cumulative += x
        if (log10(i) >= record_counter)
            push!(obj_path, objective_function(x_cumulative / i, mu, Sigma, gamma))
            push!(recording_idx, i)
            record_counter += record_delta
        end

        if i % Int(num_iterations / print_interval) == Int(num_iterations / print_interval) - 1
            println("$((i+1)/(num_iterations / 100))%:  Obj = $(round.(objective_function(x_cumulative / i, mu, Sigma, gamma), digits=6)), CVaR = $(round.(CVaR(x_cumulative / i, mu, Sigma, delta), digits=6))")
        end
    end

    x_average = x_cumulative / num_iterations
    return x_average, objective_function(x_average, mu, Sigma, gamma), CVaR(x_average, mu, Sigma, delta), recording_idx, obj_path
end

function plot_SCGD(recording_idx, obj_path, optimal_value; num_iterations=Int(1e8), file_name="Test")
    plt.figure()
    plt.loglog(recording_idx, (optimal_value .- obj_path), label="Obj. gap")
    plt.loglog(range(1.0, stop=num_iterations, length=100), 1 ./ sqrt.(range(1.0, stop=num_iterations, length=100)), "--", label="O(1/√t)")
    plt.ylim(1e-4, 1)
    plt.grid()
    plt.legend()
    plt.savefig(file_name * "_SCGD.png", dpi=300, bbox_inches="tight")

    plt.figure()
    plt.loglog(recording_idx, abs.(optimal_value .- obj_path), label="Abs. obj. gap")
    plt.loglog(range(1.0, stop=num_iterations, length=100), 1 ./ sqrt.(range(1.0, stop=num_iterations, length=100)), "--", label="O(1/√t)")
    plt.grid()
    plt.legend()
    plt.savefig(file_name * "_SCGD_abs.png", dpi=300, bbox_inches="tight")
end


"""
    EC-SCGD for expected value constrained stochastic compositional optimization
"""
function SO_f2(xu, mu, Sigma, L)
    x = xu[1:end-1]
    w_2 = multi_normal_gen(mu, Matrix(L))
    z = dot(x, w_2)
    y = vcat(x, xu[end], z)
    return y
end

function SO_g(xu, delta, xi, mu, Sigma, L)
    x = xu[1:end-1]
    u = xu[end]
    w_3 = multi_normal_gen(mu, Matrix(L))
    return u + 1/(1-delta) * max(0, -dot(w_3, x) - u) - xi
end

function SO_f1_gradient(y, mu, Sigma, L, gamma)
    w_1 = multi_normal_gen(mu, Matrix(L))
    x = y[1:end-2]
    u = y[end-1]
    z = y[end]
    diff = dot(w_1, x) - z
    return vcat(4 * gamma * (diff)^3 .* w_1, [-1 - 4 * gamma * (diff)^3])
end

function SO_f2_gradient(mu, Sigma, L)
    I_d = I(length(mu))
    w_2 = multi_normal_gen(mu, Matrix(L))
    zero = zeros(length(mu) + 1)
    return vcat(hcat(I_d, w_2), zero')
end

function SO_g_gradient(xu, delta, mu, Sigma, L)
    x = xu[1:end-1]
    u = xu[end]
    w_3 = multi_normal_gen(mu, Matrix(L))
    flag = (-dot(w_3, x) - u) >= 0
    return vcat(-1/(1-delta) .* w_3 .* flag, [1 - 1/(1-delta) * flag])
end

function iteration_EC_SCGD(xu, y, lam, mu, Sigma, L, gamma, delta, xi, alpha, eta, tau)
    f2 = SO_f2(xu, mu, Sigma, L)
    y_new = (f2 .+ tau .* y) / (1 + tau)
    f2_gradient = SO_f2_gradient(mu, Sigma, L)
    f1_gradient = SO_f1_gradient(y_new, mu, Sigma, L, gamma)
    g_gradient = SO_g_gradient(xu, delta, mu, Sigma, L)
    g = SO_g(xu, delta, xi, mu, Sigma, L)
    temp = xu .- (f2_gradient * f1_gradient .+ g_gradient * lam) / eta
    x_temp = project_to_simplex(temp[1:end-1])
    u_temp = temp[end]
    xu_new = vcat(x_temp, u_temp)
    lam_new = max(0, lam + g / alpha)
    return xu_new, y_new, lam_new
end

function run_EC_SCGD(mu, Sigma, L, gamma, delta, xi; eta=1e4, tau=1e4, num_iterations=Int(1e7), print_interval = 100, obj_cvar_constrained = 0, cvar_cvar_constrained = 0)
    x = ones(length(mu)) / length(mu)
    xu = vcat(x, 0.0)
    y = vcat(xu, 0.0)
    lam = 1.0

    xu_cumulative = zeros(length(mu) + 1)
    obj_path = []
    cvar_path = []
    recording_idx = []

    record_log10_UB = log10(num_iterations)
    record_log10_LB = 0
    record_delta = 0.05
    record_counter = record_log10_LB

    for i in 1:num_iterations
        
        eta = max(3e4, sqrt(i)*300)
        tau = max(1e3, i/50)
        alpha = max(20*length(mu),sqrt(i)*length(mu)/50)
        
        # eta = max(1e4,sqrt(i)*100)
        # tau = max(1e3,i/100)
        # alpha = max(1e3,sqrt(i))

        xu, y, lam = iteration_EC_SCGD(xu, y, lam, mu, Sigma, L, gamma, delta, xi, alpha, eta, tau)
        xu_cumulative += xu

        if (log10(i) >= record_counter)
            push!(obj_path, objective_function(xu_cumulative[1:end-1] / i, mu, Sigma, gamma))
            push!(cvar_path, CVaR(xu_cumulative[1:end-1] / i, mu, Sigma, delta))
            push!(recording_idx, i)
            record_counter += record_delta
        end

        if i % Int(num_iterations / print_interval) == Int(num_iterations / print_interval) - 1
            println("$((i+1)/(num_iterations / 100))%:  Obj* - Obj = $(round.(obj_cvar_constrained - objective_function(xu_cumulative[1:end-1] / (i+1), mu, Sigma, gamma), digits=6)),\tCVaR_UB - CVaR = $(round.(cvar_cvar_constrained - CVaR(xu_cumulative[1:end-1] / (i+1), mu, Sigma, delta), digits=6)),\tlam = $(round.(lam, digits=6))")
        end
    end

    x_average = xu_cumulative[1:end-1] / num_iterations
    println("Finished! Obj = $(round.(objective_function(x_average, mu, Sigma, gamma), digits=6)), CVaR = $(round.(CVaR(x_average, mu, Sigma, delta), digits=6)), x = $(round.(x_average, digits=6))")

    return x_average, objective_function(x_average, mu, Sigma, gamma), CVaR(x_average, mu, Sigma, delta), recording_idx, obj_path, cvar_path
end

function plot_EC_SCGD(recording_idx, obj_path, cvar_path, optimal_value, optimal_cvar; num_iterations=Int(1e8), file_name="Test")
    plt.figure()
    plt.loglog(recording_idx, (optimal_value .- obj_path), label="F(x*) - F(x)")
    plt.loglog(recording_idx, max.(0, cvar_path .- optimal_cvar), label="(CVaR - ξ)_+")
    plt.loglog(range(1.0, stop=num_iterations, length=100), 1 ./ sqrt.(range(1.0, stop=num_iterations, length=100)), "--", label="O(1/√t)")
    plt.xlim(1, num_iterations)
    plt.ylim(1/sqrt(num_iterations), 1)
    plt.grid()
    plt.legend()
    plt.xlabel("Iteration t")
    plt.ylabel("Gap")
    plt.savefig(file_name * "_EC_SCGD.png", dpi=300, bbox_inches="tight")

    # plt.figure()
    # plt.loglog(recording_idx, abs.(optimal_value .- obj_path), label="Abs. obj. gap")
    # plt.loglog(recording_idx, abs.(cvar_path .- optimal_cvar), label="|CVaR - ξ|")
    # plt.loglog(range(1.0, stop=num_iterations, length=100), 1 ./ sqrt.(range(1.0, stop=num_iterations, length=100)), "--", label="O(1/√t)")
    # plt.grid()
    # plt.legend()
    # plt.xlabel("Iteration t")
    # plt.ylabel("Gap")
    # plt.savefig(file_name * "_EC_SCGD_abs.png", dpi=300, bbox_inches="tight")
end

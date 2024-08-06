include("Utils.jl")

"""
    Gradient descent methods for the unconstrained oracle problem with knowledge of mu and Sigma
    Maximize:
        mu^T x - 3 gamma CVaR_{delta}(x)
    Constraint:
        x in Delta
"""

function iteration_unconstrained_oracle(x, mu, Sigma, gamma, eta)
    x -= gradient(x, mu, Sigma, gamma) / eta
    x = project_to_simplex(x)
    return x
end

function run_unconstrained_oracleGD(mu, Sigma, gamma, delta; eta = 1e4, num_iterations = Int(1e4))
    println("Running unconstrained oracle GD...")
    
    x = ones(length(mu)) / length(mu)
    
    prev_obj_val = 0.0
    obj_val = objective_function(x, mu, Sigma, gamma)

    i = 0
    while abs(prev_obj_val - obj_val) > 1e-15 && i < num_iterations
        eta = max(1e2,sqrt(i+1)*5)
        x = iteration_unconstrained_oracle(x, mu, Sigma, gamma, eta)

        prev_obj_val = obj_val
        obj_val = objective_function(x, mu, Sigma, gamma)
        i += 1
    end
    
    println("\tObj = $(round(objective_function(x, mu, Sigma, gamma), digits=5)), CVaR = $(round(CVaR(x, mu, Sigma, delta), digits=5)), x = $(round.(x, digits=5)).\tTerminated in $i iterations.")
    
    return objective_function(x, mu, Sigma, gamma), CVaR(x, mu, Sigma, delta), x
end

"""
    Gradient descent methods for the minimizing CVaR with knowledge of mu and Sigma
    Minimize:
        CVaR_{delta}(x)
    Constraint:
        x in Delta
"""

function iteration_minimizeCVaR_oracleGD(x, mu, Sigma, delta, eta)
    x = project_to_simplex(x - CVaR_gradient(mu, Sigma, x, delta) / eta)
    return x
end

function run_minimizeCVaR_oracleGD(mu, Sigma, delta, gamma; eta = 1e4, num_iterations = Int(1e4))
    println("Minimizing CVaR...")
    x = ones(length(mu)) / length(mu)

    prev_obj_val = 0.0
    obj_val = CVaR(x, mu, Sigma, delta)

    i = 0
    while abs(prev_obj_val - obj_val) > 1e-15 && i < num_iterations
        eta = max(1e2,sqrt(i+1)*5)
        x = iteration_minimizeCVaR_oracleGD(x, mu, Sigma, delta, eta)

        prev_obj_val = obj_val
        obj_val = CVaR(x, mu, Sigma, delta)
        i += 1
    end
    
    println("\tObj = $(round(objective_function(x, mu, Sigma, gamma), digits=5)), CVaR = $(round(CVaR(x, mu, Sigma, delta), digits=5)), x = $(round.(x, digits=5)).\tTerminated in $i iterations.")
    
    return objective_function(x, mu, Sigma, gamma), CVaR(x, mu, Sigma, delta), x
end

"""
    Gradient descent methods for the CVaR-constrained oracle problem
    Maximize:
        mu^T x - 3 gamma CVaR_{delta}(x)
    Constraint:
        CVaR_{delta}(x) leq xi
"""

function iteration_CVaRconstrained_oracle(x, lam, mu, Sigma, gamma, delta, xi, alpha, eta)
    grad = gradient(x, mu, Sigma, gamma) + lam * CVaR_gradient(mu, Sigma, x, delta)
    x = project_to_simplex(x - grad / eta)
    lam += alpha * (CVaR(x, mu, Sigma, delta) - xi)
    return x, lam
end

function run_CVaRconstrained_oracleGD(mu, Sigma, gamma, delta, xi; eta = 1e4, num_iterations = Int(1e4))
    println("Running CVaR-constrained oracle GD with xi = $xi...")
    
    x = ones(length(mu)) / length(mu)
    lam = 0.0

    prev_obj_val = 0.0
    obj_val = objective_function(x, mu, Sigma, gamma)
    g = abs(CVaR(x, mu, Sigma, delta) - xi)

    i = 0
    while (abs(prev_obj_val - obj_val) > 1e-15 || g > 1e-15) && i < num_iterations
        eta = max(1e2,sqrt(i+1)*5)
        alpha = max(1e2, sqrt(i+1)/5)
        x, lam = iteration_CVaRconstrained_oracle(x, lam, mu, Sigma, gamma, delta, xi, alpha, eta)

        prev_obj_val = obj_val
        obj_val = objective_function(x, mu, Sigma, gamma)
        g = abs(CVaR(x, mu, Sigma, delta) - xi)
        i += 1

        if i % Int(num_iterations/100) == Int(num_iterations/100)-1
            println("CVaR gap = $g")
        end
    end
    
    println("\tObj = $(round(objective_function(x, mu, Sigma, gamma), digits=5)), CVaR = $(round(CVaR(x, mu, Sigma, delta), digits=5)), x = $(round.(x, digits=5)).\tTerminated in $i iterations.")
    
    if abs(xi - CVaR(x, mu, Sigma, delta)) > 1e-8
        println("\tWARNING! Not converged. Gap = $(xi - CVaR(x, mu, Sigma, delta))")
    end
    
    return objective_function(x, mu, Sigma, gamma), CVaR(x, mu, Sigma, delta), x
end

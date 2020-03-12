using Printf

function build_master_cutting_plane(X, model, xMin, xMax)
    nVariables, nCassures = size(X)
    x_ = @variable(model, xMin[i] <= x_[i in 1:nVariables] <= xMax[i]);
    α_ = @variable(model, α_>=-1e10)
    @objective(model, :Min, α_)
    return x_, α_
end

function launch_cutting_plane(X, xMin, xMax)
    master = Model(solver=CplexSolver(CPX_PARAM_THREADS=1, CPX_PARAM_LPMETHOD=2, CPXPARAM_ScreenOutput=0));
    # master = Model(solver=ClpSolver(LogLevel=0));
    x, α = build_master_cutting_plane(X, master, xMin, xMax)
    stop = false
    lb = -1e20
    ub = +1e20
    n_ite=0

    best_ub = ub

    while ! stop
        n_ite+=1
        solve(master)
        lb = getvalue(α)
        x_value, rhs, sub_gradient = build_cut(X, x)
        ub = rhs
        best_ub = min(ub, best_ub)
        if lb >= best_ub - EPS
            stop = true
        else
            # println(rhs)
            # println(sub_gradient)
            # println(x_value)
            @constraint(master, α>=rhs+sum( sub_gradient[i] * (x[i]-x_value[i]) for i in 1:nVariables))
        end
        @printf("%10d%20.10E%20.10E%20.10E\n", n_ite, lb, ub, best_ub)
    end
end

# objective is
function build_cut(X_, x_)
    nVariables, nCassures = size(X_)
    sub_gradient_ = Base.zeros( nVariables);
    rhs_ = 0
    x_tmp = getvalue(x_)

    for i in 1:nVariables
        rhs_ += x_tmp[i]/nVariables
        # println("i = ", i," : ", x_tmp[i]/nVariables)

        sub_gradient_[i]=1.0/nVariables
        for j in 1:nCassures
            if x_tmp[i] < X_[i,j]
                rhs_+= j * 2 * abs(x_tmp[i]-X[i,j]) / nCassures / nVariables
                # println("i = ", i, " j = ", j, " : ", j * 2 * abs(x_tmp[i]-X[i,j]) / nCassures / nVariables)

                sub_gradient_[i]+=-j*2.0/nCassures/nVariables
            elseif x_tmp[i]>X_[i,j]
                rhs_+= j * 3 * abs(x_tmp[i]-X[i,j]) / nCassures / nVariables
                # println("i = ", i, " j = ", j, " : ", j * 3 * abs(x_tmp[i]-X[i,j]) / nCassures / nVariables)

                sub_gradient_[i]+=j*3.0/nCassures/nVariables
            else
                # continue
            end
        end
    end
    return x_tmp, rhs_, sub_gradient_
end

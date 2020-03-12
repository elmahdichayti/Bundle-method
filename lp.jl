
function full_jump_model(X, model, xMin, xMax)
    nVariables, nCassures = size(X)
    x_ = @variable(model, xMin[i] <= x_[i in 1:nVariables] <= xMax[i]);
    @variable(model, x_left[i in 1:nVariables, j in 1:nCassures] >= 0);
    @variable(model, x_right[i in 1:nVariables, j in 1:nCassures]>= 0 );
    @constraint(model, [i in 1:nVariables, j in 1:nCassures], x_[i] == X[i,j]-x_left[i, j]+x_right[i,j]);
    @objective(model, :Min,
        +sum(x_[i]/nVariables for i in 1:nVariables)
        +sum(+j*2*x_left[i,j]/nVariables/nCassures for i in 1:nVariables, j in 1:nCassures)
        +sum(+j*3*x_right[i,j]/nVariables/nCassures for i in 1:nVariables, j in 1:nCassures)
        );
    # b_left = @variable(model, b_left[i in 1:nVariables, j in 1:nCassures], Bin);
    # @constraint(model, [i in 1:nVariables, j in 1:nCassures], x_left[i, j] <= b_left[i,j] * 10);
    # @constraint(model, [i in 1:nVariables, j in 1:nCassures], x_right[i, j] <= (1-b_left[i,j]) * 10);
    return x_, x_left, x_right
end

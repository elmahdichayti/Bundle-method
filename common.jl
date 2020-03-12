using JuMP, CPLEX
using Random
# srand(1234)
Random.seed!(1234);
# common part of the TP, define the criterion

EPS = 1e-6;

function build_X(nVariables, nCassures, xMin, xMax)
    return [ xMin[i] + rand()*(xMax[i]-xMin[i]) for i in 1:nVariables, j in 1:nCassures]
end

function build_bounds(nVariables)
    return [-10*i for i in 1:nVariables],  [+10*i for i in 1:nVariables];
end
# let{i in 1..nVariables, j in 1..nCassures} X[i,j] := xMin[i]+Uniform01()*(xMax[i]-xMin[i]);

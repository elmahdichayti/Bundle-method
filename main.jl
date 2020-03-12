include("common.jl")
include("cutting_plane.jl")
include("lp.jl")
include("proximal.jl")

###
# general setup
###
nVariables = 50;
nCassures = 50;
xMin, xMax = build_bounds(nVariables)
X = build_X(nVariables, nCassures, xMin, xMax)

# println("-----------------------------------")
# println("-------- CUTTING PLANE ------------")
# println("-----------------------------------")
# launch_cutting_plane(X, xMin, xMax)

println("-----------------------------------")
println("-------- PROXIMAL -----------------")
println("-----------------------------------")

launch_proximal(X, xMin, xMax)

using Reactant

x = reshape([1.0, 2.0, 3.0, 4.0], (2,2))

@show x

y = Reactant.ConcreteRArray(x)

y2 = convert(Array{Float64, 2}, y)
@show y2

@assert x == y2

@show [y[1,1], y[1,2], y[2, 1], y[2, 2]]

@assert y[1,1] == x[1,1]
@assert y[1,2] == x[1,2]
@assert y[2,1] == x[2,1]
@assert y[2,2] == x[2,2]

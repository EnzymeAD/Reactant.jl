g(x::Float64, y) = 2x + y
display(g)

# g(x, y::Float64) = x + 2y
# display(g)

# println(g(2.0, 3))

# println(g(2, 3.0))

# println(g(2.0, 3.0))

g(x::Number, y) = 2x + y
println(g(2.0, 3))

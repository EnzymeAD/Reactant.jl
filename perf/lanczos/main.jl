using Reactant
using LinearAlgebra
using Random

Random.seed!(0)

A = rand(ComplexF64, 512, 512)
A = A' * A # make it hermitian
@assert ishermitian(A)

b = normalize!(rand(ComplexF64, 512))

Are = Reactant.to_rarray(A)
bre = Reactant.to_rarray(b)

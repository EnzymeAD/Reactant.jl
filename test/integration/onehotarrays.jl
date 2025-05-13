using Reactant
using Test
using OneHotArrays
using Random

function oh_mul(a, b)
  return a * b
end

@testset "OneHotArrays" begin
	m = onehotbatch([10, 20, 30, 10, 10], 10:10:40)
	r_m = Reactant.to_rarray(m)
	a = rand(100,4)
	r_a = Reactant.to_rarray(a)
	@show @code_hlo oh_mul(r_a, r_m)
	r_res = @jit oh_mul(r_a, r_m)
	res = oh_mul(a, m)
	@test convert(Array, r_res) ≈ res 
end

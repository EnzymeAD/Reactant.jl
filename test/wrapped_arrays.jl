using Reactant, Test, Statistics, NNlib

function view_getindex_1(x)
    x = view(x, 2:3, 1:2, :)
    return x[2, 1, 1]
end

function view_getindex_2(x)
    x = view(x, 2:3, 1:2, :)
    return x[1:1, 1, :]
end

function view_getindex_3(x)
    x = view(x, 2:3, 1:2, :)
    x2 = view(x, 1:1, 2:2, 1:2)
    return x2[1, 1, 1:1]
end

@testset "view getindex" begin
    x = rand(4, 4, 3)
    x_ra = Reactant.to_rarray(x)

    @test @allowscalar(@jit(view_getindex_1(x_ra))) ≈ view_getindex_1(x)
    @test @jit(view_getindex_2(x_ra)) ≈ view_getindex_2(x)
    @test @jit(view_getindex_3(x_ra)) ≈ view_getindex_3(x)
end

function reshape_wrapper(x)
    x = view(x, 2:3, 1:2, :)
    return reshape(x, 4, :)
end

@testset "reshape wrapper" begin
    x = rand(4, 4, 3)
    x_ra = Reactant.to_rarray(x)
    @test @jit(reshape_wrapper(x_ra)) ≈ reshape_wrapper(x)
end

function permutedims_wrapper(x)
    x = view(x, 2:3, 1:2, :)
    return permutedims(x, (2, 1, 3))
end

@testset "permutedims wrapper" begin
    x = rand(4, 4, 3)
    x_ra = Reactant.to_rarray(x)
    @test @jit(permutedims_wrapper(x_ra)) ≈ permutedims_wrapper(x)
end

function bcast_wrapper(f::F, x) where {F}
    x = view(x, 2:3, :)
    return f.(x)
end

@testset "Broadcasting on wrapped arrays" begin
    x = rand(4, 3)
    x_ra = Reactant.to_rarray(x)

    for op in (-, tanh, sin)
        @test @jit(bcast_wrapper(op, x_ra)) ≈ bcast_wrapper(op, x)
    end
end

function mean_var(x)
    x = view(x, 2:3, :)
    return mean(x; dims=1), var(x; dims=1)
end

@testset "mean/var" begin
    x = rand(4, 3)
    x_ra = Reactant.to_rarray(x)

    m1, v1 = mean_var(x)
    m2, v2 = @jit(mean_var(x_ra))

    @test m1 ≈ m2
    @test v1 ≈ v2
end

function btranspose_badjoint(x)
    x1 = NNlib.batched_transpose(x)
    x2 = NNlib.batched_adjoint(x)
    return x1 .+ x2
end

@testset "batched transpose/adjoint" begin
    x = rand(4, 2, 3)
    x_ra = Reactant.to_rarray(x)
    @test @jit(btranspose_badjoint(x_ra)) ≈ btranspose_badjoint(x)
end

function bypass_permutedims(x)
    x = PermutedDimsArray(x, (2, 1, 3))  # Don't use permutedims here
    return view(x, 2:3, 1:2, :)
end

@testset "PermutedDimsArray" begin
    x = rand(4, 4, 3)
    x_ra = Reactant.to_rarray(x)
    y_ra = @jit(bypass_permutedims(x_ra))
    @test @allowscalar(Array(y_ra)) ≈ bypass_permutedims(x)
end

function writeto_reshaped_array!(x)
    z1 = similar(x)
    z2 = reshape(z1, 1, 2, 3, 1)
    @. z2 = 1.0
    return z1
end

@testset "writeto_reshaped_array!" begin
    x = ConcreteRArray(rand(3, 2))
    y = @jit writeto_reshaped_array!(x)
    @test all(isone, Array(y))
end

function write_to_transposed_array!(x)
    z1 = similar(x)
    z2 = transpose(z1)
    @. z2 = 1.0
    return z1
end

@testset "write_to_transposed_array!" begin
    x = ConcreteRArray(rand(3, 2))
    y = @jit write_to_transposed_array!(x)
    @test all(isone, Array(y))
end

function write_to_adjoint_array!(x)
    z1 = similar(x)
    z2 = adjoint(z1)
    @. z2 = 1.0
    return z1
end

@testset "write_to_adjoint_array!" begin
    x = ConcreteRArray(rand(3, 2))
    y = @jit write_to_adjoint_array!(x)
    @test all(isone, Array(y))
end

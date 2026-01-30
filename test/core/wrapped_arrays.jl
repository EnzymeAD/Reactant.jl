using Reactant, Test, Statistics, NNlib, LinearAlgebra

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
    x = Reactant.TestUtils.construct_test_array(Float64, 4, 4, 3)
    x_ra = Reactant.to_rarray(x)

    @test @allowscalar(@jit(view_getindex_1(x_ra))) ≈ view_getindex_1(x)
    @test Array(@jit(view_getindex_2(x_ra))) ≈ view_getindex_2(x)
    @test Array(@jit(view_getindex_3(x_ra))) ≈ view_getindex_3(x)
end

function reshape_wrapper(x)
    x = view(x, 2:3, 1:2, :)
    return reshape(x, 4, :)
end

@testset "reshape wrapper" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 4, 4, 3)
    x_ra = Reactant.to_rarray(x)
    @test @jit(reshape_wrapper(x_ra)) ≈ reshape_wrapper(x)
end

function permutedims_wrapper(x)
    x = view(x, 2:3, 1:2, :)
    return permutedims(x, (2, 1, 3))
end

@testset "permutedims wrapper" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 4, 4, 3)
    x_ra = Reactant.to_rarray(x)
    @test @jit(permutedims_wrapper(x_ra)) ≈ permutedims_wrapper(x)
end

function bcast_wrapper(f::F, x) where {F}
    x = view(x, 2:3, :)
    return f.(x)
end

@testset "Broadcasting on wrapped arrays" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 4, 3)
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
    x = Reactant.TestUtils.construct_test_array(Float64, 4, 3)
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
    x = Reactant.TestUtils.construct_test_array(Float64, 4, 2, 3)
    x_ra = Reactant.to_rarray(x)
    @test @jit(btranspose_badjoint(x_ra)) ≈ btranspose_badjoint(x)
end

function bypass_permutedims(x)
    x = PermutedDimsArray(x, (2, 1, 3))  # Don't use permutedims here
    return view(x, 2:3, 1:2, :)
end

add_perm_dims(x) = x .+ PermutedDimsArray(x, (2, 1))

@testset "PermutedDimsArray" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 4, 4, 3)
    x_ra = Reactant.to_rarray(x)
    y_ra = @jit(bypass_permutedims(x_ra))
    @test @allowscalar(Array(y_ra)) ≈ bypass_permutedims(x)

    x = Reactant.TestUtils.construct_test_array(Float64, 4, 4)
    x_ra = Reactant.to_rarray(x)

    @test @jit(add_perm_dims(x_ra)) ≈ add_perm_dims(x)
end

function writeto_reshaped_array!(x)
    z1 = similar(x)
    z2 = reshape(z1, 1, 2, 3, 1)
    @. z2 = 1.0
    return z1
end

function write_to_transposed_array!(x)
    z1 = similar(x)
    z2 = transpose(z1)
    @. z2 = 1.0
    return z1
end

function write_to_adjoint_array!(x)
    z1 = similar(x)
    z2 = adjoint(z1)
    @. z2 = 1.0
    return z1
end

function write_to_permuted_dims_array!(x)
    z1 = similar(x)
    z2 = PermutedDimsArray(z1, (2, 1))
    @. z2 = 1.0
    return z1
end

function write_to_diagonal_array!(x)
    z = Diagonal(x)
    @. z = 1.0
    return z
end

@testset "Preserve Aliasing with Parent" begin
    @testset "$(aType)" for (aType, fn) in [
        ("ReshapedArray", writeto_reshaped_array!),
        ("Transpose", write_to_transposed_array!),
        ("Adjoint", write_to_adjoint_array!),
    ]
        x = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float64, 3, 2))
        y = @jit fn(x)
        @test all(isone, Array(y))
    end

    @testset "PermutedDimsArray" begin
        x = Reactant.TestUtils.construct_test_array(Float64, 4, 4)
        x_ra = Reactant.to_rarray(x)
        @test @jit(write_to_permuted_dims_array!(x_ra)) ≈ write_to_permuted_dims_array!(x)
    end

    @testset "Diagonal" begin
        x = Reactant.TestUtils.construct_test_array(Float64, 4, 4)
        x_ra = Reactant.to_rarray(x)
        y_ra = copy(x_ra)

        y = @jit(write_to_diagonal_array!(x_ra))
        y_res = @allowscalar Array(y)
        @test x_ra ≈ y_ra
        @test all(isone, diag(y_res))
        y_res[diagind(y_res)] .= 0
        @test all(iszero, y_res)
    end
end

function lower_triangular_write(x)
    y = LowerTriangular(copy(x))
    @. y *= 2
    return y
end

function upper_triangular_write(x)
    y = UpperTriangular(copy(x))
    @. y *= 2
    return y
end

function tridiagonal_write(x)
    y = Tridiagonal(copy(x))
    @. y *= 2
    return y
end

@testset "Broadcasted Multiply and Allocate" begin
    @testset "$(aType)" for (aType, fn) in [
        ("LowerTriangular", lower_triangular_write),
        ("UpperTriangular", upper_triangular_write),
        ("Tridiagonal", tridiagonal_write),
    ]
        x = Reactant.TestUtils.construct_test_array(Float64, 4, 4)
        x_ra = Reactant.to_rarray(x)
        @test @jit(fn(x_ra)) ≈ fn(x)
    end
end

function broadcast_reshaped_array(x, idx1, idx2)
    y = reshape(x, 20, 2)
    return y[idx1, idx2] .+ 1
end

function broadcast_reshaped_array(x, idx1, idx2::Number)
    y = reshape(x, 20, 2)
    return y[idx1, idx2] .+ 1
end

function broadcast_reshaped_array(x, idx1)
    y = reshape(x, 20, 2)
    return y[idx1, :] .+ 1
end

@testset "Broadcast reshaped array" begin
    x_ra = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float64, 5, 4, 2))
    idx1_ra = Reactant.to_rarray([3, 7, 15, 19])
    idx2_ra = Reactant.to_rarray([2, 1])

    @test broadcast_reshaped_array(Array(x_ra), Array(idx1_ra), Array(idx2_ra)) ≈
        @jit(broadcast_reshaped_array(x_ra, idx1_ra, idx2_ra)) ≈
        @jit(broadcast_reshaped_array(x_ra, Array(idx1_ra), Array(idx2_ra)))

    idx3 = Reactant.to_rarray(2; track_numbers=true)

    @test broadcast_reshaped_array(Array(x_ra), Array(idx1_ra), Int64(idx3)) ≈
        @jit(broadcast_reshaped_array(x_ra, idx1_ra, idx3)) ≈
        @jit(broadcast_reshaped_array(x_ra, Array(idx1_ra), Int64(idx3)))

    @test broadcast_reshaped_array(Array(x_ra), Array(idx1_ra)) ≈
        @jit(broadcast_reshaped_array(x_ra, idx1_ra)) ≈
        @jit(broadcast_reshaped_array(x_ra, Array(idx1_ra)))
end

@testset "reshaped subarray indexing" begin
    fn(x) = view(x, 1:2) .+ 1
    x_ra = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float64, 3, 4, 3))
    @test @jit(fn(x_ra)) ≈ fn(Array(x_ra))
end

function reshape_getindex(x)
    x = reshape(x, 2, 4, 3)
    return x[1, :, :]
end

function permutedims_getindex(x)
    x = PermutedDimsArray(x, (2, 1))
    return x[1, :]
end

@testset "no gather getindex" begin
    x = ones(8, 3)
    x_ra = Reactant.to_rarray(x)

    hlo = repr(@code_hlo(reshape_getindex(x_ra)))
    @test !occursin("stablehlo.gather", hlo)

    hlo = repr(@code_hlo(permutedims_getindex(x_ra)))
    @test !occursin("stablehlo.gather", hlo)
end

function view_adjoint(x)
    y = view(x, 1:2, 1:2)
    return adjoint(y) .+ y
end

function view_transpose(x)
    y = view(x, 1:2, 1:2)
    return transpose(y) .+ y
end

function view_diagonal(x)
    y = view(x, 1:2, 1:2)
    return Diagonal(y) .+ y
end

@testset "2 levels of wrapping" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 2, 4)
    x_ra = Reactant.to_rarray(x)

    @test @jit(view_adjoint(x_ra)) ≈ view_adjoint(x)
    @test @jit(view_transpose(x_ra)) ≈ view_transpose(x)
    @test @jit(view_diagonal(x_ra)) ≈ view_diagonal(x)
end

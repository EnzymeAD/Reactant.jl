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

    view_getindex_1_compiled = @compile view_getindex_1(x_ra)

    @test view_getindex_1_compiled(x_ra) ≈ view_getindex_1(x)

    view_getindex_2_compiled = @compile view_getindex_2(x_ra)

    @test view_getindex_2_compiled(x_ra) ≈ view_getindex_2(x)

    view_getindex_3_compiled = @compile view_getindex_3(x_ra)

    @test view_getindex_3_compiled(x_ra) ≈ view_getindex_3(x)
end

function reshape_wrapper(x)
    x = view(x, 2:3, 1:2, :)
    return reshape(x, 4, :)
end

@testset "reshape wrapper" begin
    x = rand(4, 4, 3)
    x_ra = Reactant.to_rarray(x)

    reshape_wrapper(x)

    reshape_wrapper_compiled = @compile reshape_wrapper(x_ra)

    @test reshape_wrapper_compiled(x_ra) ≈ reshape_wrapper(x)
end

function permutedims_wrapper(x)
    x = view(x, 2:3, 1:2, :)
    return permutedims(x, (2, 1, 3))
end

@testset "permutedims wrapper" begin
    x = rand(4, 4, 3)
    x_ra = Reactant.to_rarray(x)

    permutedims_wrapper(x)

    permutedims_wrapper_compiled = @compile permutedims_wrapper(x_ra)

    @test permutedims_wrapper_compiled(x_ra) ≈ permutedims_wrapper(x)
end

function bcast_wrapper(f::F, x) where {F}
    x = view(x, 2:3, :)
    return f.(x)
end

@testset "Broadcasting on wrapped arrays" begin
    x = rand(4, 3)
    x_ra = Reactant.to_rarray(x)

    for op in (-, tanh, sin)
        bcast_compiled = @compile bcast_wrapper(op, x_ra)

        @test bcast_compiled(op, x_ra) ≈ bcast_wrapper(op, x)
    end
end

function mean_var(x)
    x = view(x, 2:3, :)
    return mean(x; dims=1), var(x; dims=1)
end

@testset "mean/var" begin
    x = rand(4, 3)
    x_ra = Reactant.to_rarray(x)

    mean_var_compiled = @compile mean_var(x_ra)

    m1, v1 = mean_var(x)
    m2, v2 = mean_var_compiled(x_ra)

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

    btranspose_badjoint_compiled = @compile btranspose_badjoint(x_ra)

    @test btranspose_badjoint_compiled(x_ra) ≈ btranspose_badjoint(x)
end

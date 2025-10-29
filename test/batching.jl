using Reactant, Test

f1(x::AbstractMatrix) = sum(x; dims=1)

f2(x::AbstractMatrix, y::Int) = x .+ y

function f3(x::AbstractArray{T,3}, y::Int) where {T}
    return Reactant.Ops.batch(Base.Fix2(f2, y), x, [1, 2])
end

@testset "mapslices" begin
    A = collect(reshape(1:30, (2, 5, 3)))
    A_ra = Reactant.to_rarray(A)

    B = mapslices(f1, A; dims=[1, 2])
    B_ra = @jit mapslices(f1, A_ra; dims=[1, 2])

    @test B ≈ B_ra

    B = mapslices(sum, A; dims=[1, 3])
    B_ra = @jit mapslices(sum, A_ra; dims=[1, 3])

    @test B ≈ B_ra
end

@testset "closure" begin
    A = collect(reshape(1:30, (2, 5, 3)))
    A_ra = Reactant.to_rarray(A)

    @test @jit(f3(A_ra, 1)) ≈ A .+ 1
end

# Auto-Batching
function run_auto_batching_tests(f::F, args...) where {F}
    @testset "$(nameof(F))" begin
        @testset "Correctness" begin
            res1 = @jit f(args...)
            res2 = @jit compile_options = CompileOptions(;
                disable_auto_batching_passes=true
            ) f(args...)
            @test res1 ≈ res2
        end

        @testset "No while loops" begin
            hlo = repr(
                @code_hlo compile_options = CompileOptions(;
                    disable_auto_batching_passes=true
                ) f(args...)
            )
            @test occursin("stablehlo.while", hlo)

            hlo = repr(@code_hlo f(args...))
            @test !occursin("stablehlo.while", hlo)
        end
    end
end

function looped_reduction(y, x)
    z = copy(y)
    @trace for i in 1:size(x, 2)
        z[:, i, :] = dropdims(sum(abs2, x[:, i, :, :]; dims=3); dims=3)
    end
    return z
end

@testset "Loop of Reduces => Single Reduction" begin
    x = Reactant.to_rarray(rand(Float32, 3, 256, 5, 7))
    y = Reactant.to_rarray(rand(Float32, 3, 260, 5))

    run_auto_batching_tests(looped_reduction, y, x)
end

function naive_batched_matmul(x, y)
    @assert size(x, 3) == size(y, 3)
    z = similar(x, size(x, 1), size(y, 2), size(x, 3))
    @trace for i in 1:size(x, 3)
        z[:, :, i] = x[:, :, i] * y[:, :, i]
    end
    return z
end

@testset "Naive Batched Matmul => Single Dot General" begin
    x = Reactant.to_rarray(rand(Float32, 3, 256, 5))
    y = Reactant.to_rarray(rand(Float32, 256, 7, 5))

    run_auto_batching_tests(naive_batched_matmul, x, y)
end

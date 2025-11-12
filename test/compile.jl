using Reactant, Test

Base.sum(x::NamedTuple{(:a,),Tuple{T}}) where {T<:Reactant.TracedRArray} = (; a=sum(x.a))

@testset "compile" begin
    @testset "create_result" begin
        @testset "NamedTuple" begin
            x = (; a=Reactant.TestUtils.construct_test_array(Float64, 4, 3))
            x2 = Reactant.to_rarray(x)

            res = @jit sum(x2)
            @test res isa NamedTuple
            @test res.a isa ConcreteRNumber{Float64}
            @test isapprox(res.a, sum(x.a))
        end

        @testset "Array" begin
            x = [1 2; 3 4; 5 6]
            f = Reactant.compile(() -> x, ())
            @test f() ≈ x
        end
    end

    @testset "world-age" begin
        a = ones(2, 10)
        b = ones(10, 2)
        a_ra = Reactant.to_rarray(a)
        b_ra = Reactant.to_rarray(b)

        fworld(x, y) = @jit(x * y)

        @test fworld(a_ra, b_ra) ≈ ones(2, 2) * 10
    end

    @testset "type casting & optimized out returns" begin
        a = ones(2, 10)
        a_ra = Reactant.to_rarray(a)

        ftype1(x) = Float64.(x)
        ftype2(x) = Float32.(x)

        y1 = @jit ftype1(a_ra)
        y2 = @jit ftype2(a_ra)

        @test y1 isa Reactant.ConcreteRArray{Float64,2}
        @test y2 isa Reactant.ConcreteRArray{Float32,2}

        @test y1 ≈ Float64.(a)
        @test y2 ≈ Float32.(a)
    end

    @testset "no variable name collisions in compile macros (#237)" begin
        f(x) = x
        g(x) = f(x)
        x = Reactant.TestUtils.construct_test_array(Float64, 2, 2)
        y = Reactant.to_rarray(x)
        @test (@jit g(y); true)
    end

    # disabled due to long test time (core tests go from 2m to 7m just with this test)
    # @testset "resource exhaustation bug (#190)" begin
    #     x = rand(2, 2)
    #     y = Reactant.to_rarray(x)
    #     @test try
    #         for _ in 1:10_000
    #             f = @compile sum(y)
    #         end
    #         true
    #     catch e
    #         false
    #     end
    # end
end

@testset "Module export" begin
    f(x) = sin.(cos.(x))
    x_ra = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float64, 3))

    hlo_code = @code_hlo f(x_ra)
    @test !startswith(string(hlo_code), "Module")
    @test startswith(string(hlo_code), "module")
end

@testset "Bool attributes" begin
    x_ra = Reactant.to_rarray(false; track_numbers=Number)
    @test @jit(iszero(x_ra)) == true
    x_ra = Reactant.to_rarray(true; track_numbers=Number)
    @test @jit(iszero(x_ra)) == false
end

@testset "Vararg compilation: Issue #293" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 2, 2)
    x_ra = Reactant.to_rarray(x)

    @test @allowscalar(x_ra[1]) ≈ x[1]
    @test @allowscalar(x_ra[1:1]) ≈ x[1:1]
end

@testset "no_nan passes" begin
    x_ra = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 4, 16))
    y_ra = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 4, 16))

    fn(x) = x .- x

    hlo = @code_hlo fn(x_ra)
    @test occursin("subtract", repr(hlo))
    @test !occursin("constant", repr(hlo))
    hlo = @code_hlo no_nan = true fn(x_ra)
    @test !occursin("subtract", repr(hlo))
    @test occursin("constant", repr(hlo))

    fn(x, y) = begin
        c = x .+ y
        return c .- y
    end

    hlo = @code_hlo fn(x_ra, y_ra)
    @test occursin("subtract", repr(hlo))
    @test occursin("add", repr(hlo))
    hlo = @code_hlo no_nan = true fn(x_ra, y_ra)
    @test !occursin("subtract", repr(hlo))
    @test !occursin("add", repr(hlo))
end

# While a bit specific, the following is used to check for a bug in `should_rewrite_call`
function sinusoidal_embedding(
    x::AbstractArray{T,4}, min_freq, max_freq, embedding_dims::Int
) where {T}
    if size(x)[1:3] != (1, 1, 1)
        throw(DimensionMismatch("Input shape must be (1, 1, 1, batch)"))
    end

    lower, upper = log(T(min_freq)), log(T(max_freq))
    n = embedding_dims ÷ 2
    x_ = 2 .* x .* exp.(reshape(range(lower, upper; length=n), 1, 1, n, 1))
    return cat(sinpi.(x_), cospi.(x_); dims=Val(3))
end

@testset "sinusoidal_embedding" begin
    x_ra = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 1, 1, 1, 4))
    hlo = @code_hlo sinusoidal_embedding(x_ra, 0.1, 10.0, 4)
end

# test #493
@testset "unique(::Vector{Symbol}) (#493)" begin
    x = [:a, :b, :a]
    @test @jit(unique(x)) == [:a, :b]
end

@testset "custom trace path" begin
    struct MockTestCustomPath{T}
        x::T
    end

    function Reactant.Compiler.make_tracer(
        seen, prev::MockTestCustomPath, path, mode; kwargs...
    )
        custom_path = Reactant.append_path(path, (; custom_id=1))
        traced_x = Reactant.make_tracer(seen, prev.x, custom_path, mode; kwargs...)
        return MockTestCustomPath(traced_x)
    end

    function Reactant.traced_getfield(
        x::MockTestCustomPath, fld::@NamedTuple{custom_id::Int}
    )
        return if fld.custom_id == 1
            x.x
        else
            error("this is awkward... shouldn't have reach here")
        end
    end

    function Reactant.Compiler.create_result(
        tocopy::MockTestCustomPath,
        path,
        result_stores,
        path_to_shard_info,
        to_unreshard_results,
        unresharded_code::Vector{Expr},
        unresharded_arrays_cache,
        used_shardinfo,
        result_cache,
        var_idx,
        resultgen_code,
    )
        custom_path = Reactant.append_path(path, (; custom_id=1))

        args = (
            result_stores,
            path_to_shard_info,
            to_unreshard_results,
            unresharded_code::Vector{Expr},
            unresharded_arrays_cache,
            used_shardinfo,
            result_cache,
            var_idx,
            resultgen_code,
        )

        if !haskey(result_cache, tocopy)
            ar = Reactant.Compiler.create_result(tocopy.x, custom_path, args...)
            sym = Symbol("result", var_idx[])
            var_idx[] += 1

            push!(
                resultgen_code,
                quote
                    $sym = ($MockTestCustomPath)($ar)
                end,
            )
            result_cache[tocopy] = sym
        end

        return quote
            $(result_cache[tocopy])
        end
    end

    fcustom_path(x) = MockTestCustomPath(x.x)

    x = MockTestCustomPath(ones(Int))
    xre = MockTestCustomPath(Reactant.to_rarray(x.x))

    y = @jit fcustom_path(xre)
    @test y isa MockTestCustomPath
    @test y.x isa Reactant.RArray
    @test y.x == fcustom_path(x).x
end

# CHLO legalize options
# test that we are running some mhlo passes first before legalizing, else we will end up
# decomposing some necessary ops
function fn_test(x)
    y = Reactant.Ops.top_k(x, 16).values
    y_complex = Complex.(y, -y .+ 1)
    conj!(y_complex)
    return y_complex
end

@testset "chlo legalize" begin
    x_ra = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 128))
    hlo = @code_hlo legalize_chlo_to_stablehlo = true fn_test(x_ra)
    @test occursin("mhlo.topk", repr(hlo))
end

function fn_test_for_synchronize(x)
    return x .+ 1
end

@testset "synchronize" begin
    @test isnothing(Reactant.synchronize(1))
    @test isnothing(Reactant.synchronize([1, 2, 3]))

    x = Reactant.TestUtils.construct_test_array(Float32, 10)

    @test isnothing(Reactant.synchronize(x))

    xr = Reactant.to_rarray(x)
    fsyncfalse = @compile sync = false fn_test_for_synchronize(xr)
    fsynctrue = @compile sync = true fn_test_for_synchronize(xr)

    ysyncfalse = fsyncfalse(xr)
    @test isnothing(Reactant.synchronize(ysyncfalse))

    ysynctrue = fsynctrue(xr)
    @test isnothing(Reactant.synchronize(ysynctrue))

    @test ysyncfalse == ysynctrue

    @test Reactant.synchronize((ysyncfalse, ysynctrue)) == nothing
end

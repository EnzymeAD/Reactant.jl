using Reactant, Test, Enzyme, InteractiveUtils, FileCheck

const RunningOnTPU = contains(string(Reactant.devices()[1]), "TPU")

Base.sum(x::NamedTuple{(:a,),Tuple{T}}) where {T<:Reactant.TracedRArray} = (; a=sum(x.a))

@inline intout(vis::AbstractArray{T}) where {T<:Real} = similar(vis, T)
intout_caller(vis) = @noinline intout(vis)

@testset "compile" begin
    vis = rand(Float64, 64)
    visr = Reactant.to_rarray(vis)
    @test_throws MethodError @compile intout_caller(visr)
end

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

    @testset "Enum" begin
        @enum MyEnum begin
            MyEnumA = 1
            MyEnumB = 2
        end

        x = MyEnumA
        f = @compile identity(x)
        @test f(x) == x
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
    @test @filecheck begin
        @check_not "constant"
        @check "subtract"
        repr(hlo)
    end
    hlo = @code_hlo no_nan = true fn(x_ra)
    @test @filecheck begin
        @check_not "subtract"
        @check "constant"
        repr(hlo)
    end

    fn(x, y) = begin
        c = x .+ y
        return c .- y
    end

    hlo = @code_hlo fn(x_ra, y_ra)
    @test @filecheck begin
        @check_dag "subtract"
        @check_dag "add"
        repr(hlo)
    end
    hlo = @code_hlo no_nan = true fn(x_ra, y_ra)
    @test @filecheck begin
        @check_not "subtract"
        @check_not "add"
        repr(hlo)
    end
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
        path::Tuple,
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
    @test @filecheck begin
        @check "mhlo.topk"
        repr(hlo)
    end
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

@testset "Julia Compilation cache" begin
    x = @compile -(Reactant.to_rarray(ones(2)))
    y = @compile -(Reactant.to_rarray(ones(2)))

    @test typeof(x) == typeof(y)
end

f_var(args...) = sum(args)

@testset "Vararg" begin
    x = Reactant.to_rarray(ones(3))
    y = Reactant.to_rarray(3 * ones(3))
    z = Reactant.to_rarray(2.6 * ones(3))

    @test @jit(f_var(x, y, z)) ≈ [6.6, 6.6, 6.6]
end

sumcos(x) = sum(cos.(x))

function grad_ip(x)
    dx = Enzyme.make_zero(x)
    Enzyme.autodiff(Reverse, sumcos, Active, Duplicated(x, dx))
    return dx
end

function resgrad_ip(x)
    dx = Enzyme.make_zero(x)
    res = Enzyme.autodiff(ReverseWithPrimal, sumcos, Active, Duplicated(x, dx))
    return (res, dx)
end

@testset "Basic grad cos" begin
    c = Reactant.to_rarray(ones(3, 2))

    @test @jit(grad_ip(c)) ≈ -sin.(ones(3, 2))

    orig, r = @jit(resgrad_ip(c))

    @test orig[2] ≈ sum(cos.(ones(3, 2)))
    @test r ≈ -sin.(ones(3, 2))
end

@testset "matmul" begin
    c = Reactant.to_rarray(ones(50, 70))
    d = Reactant.to_rarray(ones(70, 30))

    @test @jit(*(c, d)) ≈ *(ones(50, 70), ones(70, 30))
end

@testset "@code_hlo" begin
    W = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 10, 20))
    x = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 20, 5))
    res = @code_hlo W * x

    @test @filecheck begin
        @check "stablehlo.dot_general"
        sprint(show, res)
    end
end

@testset "@code_hlo broadcasting" begin
    x = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 2, 2))
    y = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 2, 2))
    res = @code_hlo (.+)(x, y)

    @test @filecheck begin
        @check "stablehlo.add"
        sprint(show, res)
    end
end

@testset "@code_xla" begin
    x_ra = Reactant.to_rarray(ones(Float32, 4))
    hlo = repr(@code_xla(sin.(x_ra)))
    @test @filecheck begin
        @check_dag "HloModule"
        @check_dag "sine"
        hlo
    end
end

@testset "Raise keyword" begin
    v = Reactant.TestUtils.construct_test_array(Float32, 16)
    rv = Reactant.to_rarray(v)
    @test sin.(v) ≈ @jit raise = true sin.(rv)
    @test cos.(v) ≈ @jit raise = false cos.(rv)
    @test exp.(v) ≈ @jit raise = "canonicalize" exp.(rv)
    @test_throws Reactant.MLIR.IR.AddPipelineException @jit raise = "this_pass-does_not_ExisT" exp.(
        rv
    )
end

@testset "Broadcasting with Range" begin
    x = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float64, 10))
    fn(x) = x .+ (1:length(x))

    @test @jit(fn(x)) ≈ fn(Array(x))
end

@testset "don't expand ranges by default" begin
    fn(x) = Reactant.broadcast_to_size(x, (length(x),))

    hlo = repr(@code_hlo(fn(1:10000)))
    @test @filecheck begin
        @check_dag "stablehlo.iota"
        @check_dag "stablehlo.add"
        hlo
    end
    @test Array(@jit(fn(1:10000))) ≈ collect(1:10000)

    hlo = repr(@code_hlo(fn(32:10000)))
    @test @filecheck begin
        @check_dag "stablehlo.iota"
        @check_dag "stablehlo.add"
        hlo
    end
    @test Array(@jit(fn(32:10000))) ≈ collect(32:10000)

    hlo = repr(@code_hlo(fn(0:10000)))
    @test @filecheck begin
        @check_not "stablehlo.add"
        @check "stablehlo.iota"
        hlo
    end
    @test Array(@jit(fn(0:10000))) ≈ collect(0:10000)

    hlo = repr(@code_hlo(fn(Base.OneTo(10000))))
    @test @filecheck begin
        @check_dag "stablehlo.iota"
        @check_dag "stablehlo.add"
        hlo
    end
    @test Array(@jit(fn(Base.OneTo(10000)))) ≈ collect(Base.OneTo(10000))
end

function dip!(x)
    x[:a] = x[:a] .* x[:b]
    return nothing
end

@testset "Dict" begin
    x = Dict{Symbol,Vector{Float32}}()
    x[:a] = 2.7 * ones(4)
    x[:b] = 3.1 * ones(4)

    ra = Reactant.to_rarray(x)
    @jit dip!(ra)
    @test ra[:a] ≈ (2.7 * 3.1) * ones(4)
end

@testset "ConcreteRArray inplace broadcast" begin
    x = Reactant.to_rarray(zeros(Float32, 2, 3))
    y = Reactant.to_rarray(reshape(collect(Float32, 1:6), 2, 3))

    x .= y ./ 2

    @test Array(x) ≈ Array(y) ./ 2

    x = zeros(Float32, 2, 3)
    x .= y ./ 2

    @test Array(x) ≈ Array(y) ./ 2

    x = view(zeros(Float32, 2, 5), :, 1:3)
    x .= y ./ 2

    @test Array(x) ≈ Array(y) ./ 2
end

@testset "HLO Cost Analysis" begin
    x_ra = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float64, 4, 4))
    mul_comp = @compile x_ra * x_ra
    @test begin
        Reactant.XLA.cost_analysis(mul_comp) isa
        Reactant.MLIR.API.JLHloCostAnalysisProperties
    end broken = RunningOnTPU
end

function fractional_idx(times, t)
    n₂ = searchsortedfirst(times, t)
    n₁ = max(1, n₂ - 1)
    Nt = length(times)
    n₂ = min(Nt, n₂)

    t₁ = times[n₁]
    t₂ = times[n₂]

    ñ = (t - t₁) / (t₂ - t₁)

    return ñ, n₁, n₂
end

@testset "Fractional index" begin
    times = 0:0.01:4.5
    @test times isa Base.StepRangeLen
    res = @jit fractional_idx(times, ConcreteRNumber(2.143))
    @test res[1] ≈ 0.29999999999997334
    @test res[2] == 215
    @test res[3] == 216
end

@testset "Traced fractional index" begin
    times = Reactant.to_rarray(0:0.01:4.5; track_numbers=Number)
    @test times isa Reactant.TracedStepRangeLen
    res = @jit fractional_idx(times, ConcreteRNumber(2.143))
    @test res[1] ≈ 0.29999999999997334
    @test res[2] == 215
    @test res[3] == 216
end

@testset "Unitrange" begin
    x = 2:10
    @test (@jit getindex(x, 3)) == 4
    @test (@jit getindex(x, Reactant.ConcreteRNumber(4))) == 5

    x = Reactant.to_rarray(2:10; track_numbers=Number)
    @test (@jit getindex(x, 3)) == 4
    @test (@jit getindex(x, Reactant.ConcreteRNumber(4))) == 5
end

linrange_mat(x1, x2) = Reactant.materialize_traced_array(LinRange(x1, x2, 10024))

@testset "LinRange" begin
    x1 = 0.0f0
    x2 = 1.0f0
    x1_ra = Reactant.to_rarray(x1; track_numbers=Number)
    x2_ra = Reactant.to_rarray(x2; track_numbers=Number)

    @test @jit(linrange_mat(x1_ra, x2_ra)) ≈ collect(LinRange(x1, x2, 10024))
    hlo = repr(@code_hlo(linrange_mat(x1_ra, x2_ra)))
    @test @filecheck begin
        @check "stablehlo.iota"
        hlo
    end
end

@testset "chlo legalize to stablehlo" begin
    x = Reactant.TestUtils.construct_test_array(ComplexF32, 4, 4)
    x_ra = Reactant.to_rarray(x)

    hlo1 = repr(@code_hlo Reactant.Ops.conj(x_ra))
    hlo2 = repr(@code_hlo legalize_chlo_to_stablehlo = true Reactant.Ops.conj(x_ra))

    @test @filecheck begin
        @check "chlo.conj"
        hlo1
    end
    @test @filecheck begin
        @check_not "chlo"
        hlo2
    end
end

@testset "Module printing" begin
    for opt in (true, false, :before_jit), debug in (true, false)
        v = collect(Float32(1):Float32(64))
        vr = Reactant.to_rarray(v)
        mod = @code_hlo optimize = opt log.(vr)

        io = IOBuffer()
        show(IOContext(io, :debug => debug), mod)
        mod_string = String(take!(io))

        res = @jit(Reactant.Ops.hlo_call(mod_string, vr))[1]
        @test res ≈ log.(v)
    end
end

@testset "Dump MLIR modules" begin
    always_old = Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[]
    dir_old = Reactant.MLIR.IR.DUMP_MLIR_DIR[]

    mktempdir() do dir
        Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
        Reactant.MLIR.IR.DUMP_MLIR_DIR[] = dir
        @compile sin.(Reactant.to_rarray(Float32[1.0]))
        for mod in readdir(dir; join=true)
            @test @filecheck begin
                @check "hlo.sine"
                read(mod, String)
            end
        end
    end

    mktempdir() do dir
        Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = false
        Reactant.MLIR.IR.DUMP_MLIR_DIR[] = dir
        @compile exp.(Reactant.to_rarray(Float32[1.0]))
        @test isempty(readdir(dir; join=true))
    end

    Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = always_old
    Reactant.MLIR.IR.DUMP_MLIR_DIR[] = dir_old
end

@testset "Allocator Stats" begin
    platform_name = lowercase(Reactant.XLA.platform_name(Reactant.XLA.default_backend()))
    if platform_name != "cpu"
        @test Reactant.XLA.allocatorstats() isa Reactant.XLA.AllocatorStats
    else
        @test_throws Reactant.XLA.ReactantInternalError Reactant.XLA.allocatorstats()
    end
end

# FIXME: this has too many intermittent failures. Re-enable once fixed
# @testset "compilation cache" begin
#     if Reactant.PersistentCompileCache.autotune_cache_enabled() &&
#         contains(string(Reactant.devices()[1]), "CUDA")
#         A = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 2, 5))
#         B = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 5, 1000))
#         @jit A * B

#         @test any(
#             endswith(".textproto"),
#             readdir(Reactant.PersistentCompileCache.get_autotune_cache_directory()),
#         )
#     end
# end

@testset "call through inference barrier" begin
    points = [
        Reactant.TestUtils.construct_test_array(Float32, 2),
        Reactant.TestUtils.construct_test_array(Float32, 2),
    ]
    params = Reactant.TestUtils.construct_test_array(Float32, 4, 2)
    points_ra = Reactant.to_rarray(points)
    params_ra = Reactant.to_rarray(params)

    f(params, points) = mapreduce(Base.Fix1(*, params), +, points)

    @test @jit(f(params_ra, points_ra)) ≈ f(params, points)
end

@testset "Dict with Symbol Keys" begin
    f(x) = Dict(:Mhalo => x, :x => x .+ 1)
    x = Reactant.to_rarray([1.0f0, 2.0f0])
    y = @jit(f(x))

    @test y isa Dict
    @test haskey(y, :Mhalo)
    @test haskey(y, :x)
    @test Array(y[:Mhalo]) ≈ [1.0f0, 2.0f0]
    @test Array(y[:x]) ≈ [2.0f0, 3.0f0]
end

@testset "debug = true" begin
    x = Reactant.to_rarray([1, 2, 3])
    ir = sprint(show, @code_hlo debug = true sum(x))
    if !RunningOnTPU
        @test @filecheck begin
            @check_dag "loc(\"arg1 (path=(:args, 1))\")"
            ir
        end
    end
end

@testset "code_hlo throws when passed a thunk" begin
    x = Reactant.to_rarray([1, 2, 3])
    rsum = @compile sync = true sum(x)
    @test_throws "`@code_hlo` expects the original function, not a compiled `Thunk`." (@code_hlo rsum(
        x
    ))
end

struct ThunkOutlineLayer{A,B}
    w::A
    b::B
end

function thunk_outline_inc_nt(nt::NamedTuple{K}) where {K}
    return NamedTuple{K}(map(thunk_outline_inc_nt, values(nt)))
end
thunk_outline_inc_nt(x::Reactant.TracedRArray{T}) where {T} = x .+ T(2)

@testset "Large buffer count thunk outlining (#3511)" begin
    # 1. In-place Vector mutation & deterministic thunk body caching
    N = 200
    xs_host = [Float32[i, i + 1] for i in 1:N]
    xs = [Reactant.to_rarray(copy(v)) for v in xs_host]
    function step_many!(arrs)
        for a in arrs
            a .+= 1.0f0
        end
        return nothing
    end
    f1 = @compile step_many!(xs)
    f2 = @compile step_many!(xs)
    @test typeof(f1) == typeof(f2)
    @test f1(xs) === nothing
    @test f1(xs) === nothing
    @test all(Array(xs[i]) ≈ xs_host[i] .+ 2.0f0 for i in 1:N)

    # 2. Returning a new Vector of N arrays
    map_many(arrs) = map(a -> a .* 2.0f0 .+ 1.0f0, arrs)
    f_map = @compile map_many(xs)
    ys = f_map(xs)
    @test ys isa Vector{<:Reactant.ConcreteRArray{Float32,1}}
    @test length(ys) == N
    @test all(Array(ys[i]) ≈ 2.0f0 .* (xs_host[i] .+ 2.0f0) .+ 1.0f0 for i in 1:N)

    # 3. Periodic Vector of multi-field structs mixing preserved & computed buffers
    L = 64
    layers_host = [(rand(Float32, 2, 2), rand(Float32, 2)) for _ in 1:L]
    layers = [
        ThunkOutlineLayer(Reactant.to_rarray(copy(w)), Reactant.to_rarray(copy(b))) for
        (w, b) in layers_host
    ]
    update_layers(ls) = map(l -> ThunkOutlineLayer(l.w, l.b .+ 1.0f0), ls)
    f_layers = @compile update_layers(layers)
    out_layers = f_layers(layers)
    @test length(out_layers) == L
    @test all(
        Array(out_layers[i].w) ≈ layers_host[i][1] &&
        Array(out_layers[i].b) ≈ layers_host[i][2] .+ 1.0f0 for i in 1:L
    )

    # 4. Cross-slot in-place aliasing (`reverse!`)
    rev_host = [Float32[k, 2k] for k in 1:32]
    rev_xs = [Reactant.to_rarray(copy(v)) for v in rev_host]
    f_rev = @compile reverse!(rev_xs)
    f_rev(rev_xs)
    @test all(Array(rev_xs[k]) ≈ rev_host[33 - k] for k in 1:32)

    # 5. Lux-style homogeneous NamedTuple of repeated layers
    hom_nt_host = (;
        (
            Symbol(:layer_, k) => (w=Float32[k k+1; k+2 k+3], b=Float32[k, -k]) for
            k in 1:32
        )...,
    )
    hom_nt_r = Reactant.to_rarray(hom_nt_host)
    step_hom_nt(nt::NamedTuple{K}) where {K} =
        NamedTuple{K}(map(l -> (w=l.w .* 2.0f0, b=l.b .+ 1.0f0), values(nt)))
    f_hom_nt = @compile step_hom_nt(hom_nt_r)
    res_hom_nt = f_hom_nt(hom_nt_r)
    @test typeof(res_hom_nt) === typeof(hom_nt_r)
    @test Array(res_hom_nt.layer_1.w) ≈ hom_nt_host.layer_1.w .* 2.0f0
    @test Array(res_hom_nt.layer_32.b) ≈ hom_nt_host.layer_32.b .+ 1.0f0

    # 6. Larger heterogeneous nested NamedTuple (exercises `unpack_selected_fields`,
    #    `pack_vec_to_tuple_or_nt`, and binary-tree `ThunkChunk{depth, tag}` outlining)
    het_blocks = Pair{Symbol,Any}[]
    for b in 1:12
        fields = Pair{Symbol,Any}[
            Symbol(:a_, b) => Float32[b, b + 1],
            Symbol(:b_, b) => Float64[b 2b; 3b 4b],
            Symbol(:c_, b) => Float32[b, b + 1, b + 2],
            Symbol(:d_, b) => Float64[10b],
            Symbol(:e_, b) => Float32[b 2b],
        ]
        push!(het_blocks, Symbol(:block_, b) => (; fields...))
    end
    nt_host = (; het_blocks...)
    nt_r = Reactant.to_rarray(nt_host)
    f_nt = @compile thunk_outline_inc_nt(nt_r)
    res_nt = f_nt(nt_r)
    @test typeof(res_nt) === typeof(nt_r)
    @test Array(res_nt.block_1.a_1) ≈ nt_host.block_1.a_1 .+ 2.0f0
    @test Array(res_nt.block_1.b_1) ≈ nt_host.block_1.b_1 .+ 2.0
    @test Array(res_nt.block_12.c_12) ≈ nt_host.block_12.c_12 .+ 2.0f0
    @test Array(res_nt.block_12.e_12) ≈ nt_host.block_12.e_12 .+ 2.0f0
end

using StaticArrays, Reactant, Enzyme, Test

@testset "broadcast with a traced number" begin
    f(p) = SVector(1.0, 2.0) .+ p
    g(p) = SVector(p, 2p) .* p
    h(p) = SMatrix{2,2}(1.0, 2.0, 3.0, 4.0) .- p

    p = ConcreteRNumber(2.0)
    @test @jit(f(p)) ≈ f(2.0)
    @test @jit(g(p)) ≈ g(2.0)
    @test @jit(h(p)) ≈ h(2.0)
end

struct Point3 <: FieldVector{3,Float64}
    x::Float64
    y::Float64
    z::Float64
end

@testset "map with mixed static arrays (#3393)" begin
    t = ConcreteRNumber(2.0)

    # `StaticArrays` implements `+`/`-` via `map`, which Reactant overlays. A static array
    # of traced numbers can't be materialized into a `TracedRArray`, so mixed inputs used to
    # error with `MethodError: no method matching +(::SVector{...}, ::TracedRNumber)`.
    works(t) = SVector(t, t) + SVector(one(t), zero(t))
    fails(t) = SVector(t, t) + SVector(1.0, 0.0)
    fails_int(t) = SVector(t, t) + SVector(1, 0)
    fails_rev(t) = SVector(1.0, 0.0) - SVector(t, t)
    fails_bcast(t) = SVector(t, t) .+ SVector(1.0, 0.0)
    fails_smatrix(t) = SMatrix{2,2}(t, t, t, t) + SMatrix{2,2}(1.0, 2.0, 3.0, 4.0)
    fails_fv(t) = SVector(t, t, t) + Point3(1.0, 2.0, 3.0)
    fails_fv_rev(t) = Point3(1.0, 2.0, 3.0) + SVector(t, t, t)
    function fails_map3(t)
        return map((a, b, c) -> a * b + c, SVector(t, t), SVector(1.0, 2.0), SVector(3, 4))
    end
    fails_vec(t) = map(+, SVector(t, t), [1.0, 2.0])
    fails_vec_first(t) = map(+, [1.0, 2.0], SVector(t, t))
    closure(t) = map(x -> x * t, SVector(1.0, 2.0))
    closure_mixed(t) = map((x, y) -> x * t + y, SVector(1.0, 2.0), [3.0, 4.0])

    for fn in (
        works,
        fails,
        fails_int,
        fails_rev,
        fails_bcast,
        fails_smatrix,
        fails_fv,
        fails_fv_rev,
        fails_map3,
        fails_vec,
        fails_vec_first,
        closure,
        closure_mixed,
    )
        @testset "$fn" begin
            res = @jit fn(t)
            ref = fn(2.0)
            @test res isa StaticArray
            @test size(res) == size(ref)
            @test Array(collect(res)) ≈ Array(collect(ref))
        end
    end

    # Mixing a `StaticArray` with a traced `ConcreteRArray`
    v = ConcreteRArray([3.0, 4.0])
    for (res, ref) in (
        (@jit((v -> SVector(1.0, 2.0) + v)(v)), SVector(4.0, 6.0)),
        (@jit((v -> v + SVector(1.0, 2.0))(v)), SVector(4.0, 6.0)),
        (@jit(((t, v) -> SVector(t, t) + v)(t, v)), SVector(5.0, 6.0)),
        (
            @jit((w -> w + Point3(1.0, 2.0, 3.0))(ConcreteRArray([3.0, 4.0, 5.0]))),
            SVector(4.0, 6.0, 8.0),
        ),
    )
        @test res isa StaticArray
        @test size(res) == size(ref)
        @test Array(collect(res)) ≈ Array(collect(ref))
    end
    @test_throws DimensionMismatch @jit((v -> SVector(1.0, 2.0, 3.0) + v)(v))

    # Non-array arguments follow the usual path and return a (traced) array
    res = @jit map(+, SVector(t, t), (1.0, 2.0))
    @test res isa Vector
    @test Array(collect(res)) ≈ [3.0, 4.0]

    # Differentiation through mixed static-array arithmetic
    loss(t) = sum(SVector(t, 2t) + SVector(1.0, 3.0))
    @test only(@jit(Enzyme.gradient(Reverse, loss, t))) ≈ 3.0
end

@testset "mapreduce with static arrays" begin
    t = ConcreteRNumber(2.0)

    @test @jit(sum(SVector(t, t))) ≈ 4.0
    @test @jit(mapreduce(abs2, +, SVector(t, t))) ≈ 8.0
    @test @jit((t -> mapreduce(x -> sum([x, 1.0]), +, SVector(t, 2t)))(t)) ≈ 8.0

    sum_dims(t) = sum(SMatrix{2,2}(t, 2t, 3t, 4t); dims=1)
    res = @jit sum_dims(t)
    @test res isa StaticArray
    @test Array(collect(res)) ≈ Array(collect(sum_dims(2.0)))

    # Plain `StaticArray` with a traced closure preserves static return type
    closure_dims(t) = mapreduce(x -> x * t, +, SMatrix{2,2}(1.0, 2.0, 3.0, 4.0); dims=1)
    res_closure = @jit closure_dims(t)
    @test res_closure isa StaticArray
    @test Array(collect(res_closure)) ≈ Array(collect(closure_dims(2.0)))
end

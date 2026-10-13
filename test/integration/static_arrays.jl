using StaticArrays, Reactant, Test

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

    # Non-static arguments follow the usual path and return a (traced) array
    res = @jit map(+, SVector(t, t), (1.0, 2.0))
    @test res isa Vector
    @test Array(collect(res)) ≈ [3.0, 4.0]
end

@testset "mapreduce with static arrays" begin
    t = ConcreteRNumber(2.0)

    @test @jit(sum(SVector(t, t))) ≈ 4.0
    @test @jit(mapreduce(abs2, +, SVector(t, t))) ≈ 8.0

    sum_dims(t) = sum(SMatrix{2,2}(t, 2t, 3t, 4t); dims=1)
    res = @jit sum_dims(t)
    @test res isa StaticArray
    @test Array(collect(res)) ≈ Array(collect(sum_dims(2.0)))
end

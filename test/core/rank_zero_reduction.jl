using Reactant, Test

function rank_zero_reductions(x)
    return (
        sum(x),
        sum(x; dims=()),
        sum(abs2, x),
        sum(abs2, x; dims=()),
        sum(x; init=10),
        sum(y -> y == 1, x),
        prod(y -> y == 1, x),
    )
end

function rank_zero_bool_reductions(x)
    return (
        sum(x),
        sum(x; dims=()),
        sum(x; init=10),
        sum(y -> y == 1, x),
        prod(y -> y == 1, x),
    )
end

function check_rank_zero_reductions(x, changed, fn=rank_zero_reductions)
    a = Reactant.to_rarray(fill(x))
    b = Reactant.to_rarray(fill(changed))
    a_before, b_before = Array(a), Array(b)
    compiled = @compile fn(a)

    for (value, input) in ((x, a), (changed, b))
        expected = fn(fill(value))
        actual = compiled(input)
        @test length(actual) == length(expected)
        for (got, want) in zip(actual, expected)
            if want isa AbstractArray
                @test size(got) == size(want) == ()
                @test Array(got) == want
                @test eltype(Array(got)) === eltype(want)
            else
                @test got == want
                @test Reactant.unwrapped_eltype(got) === typeof(want)
            end
        end
    end
    @test Array(a) == a_before
    @test Array(b) == b_before
end

@testset "rank-zero reductions keep mapped and accumulator types" begin
    check_rank_zero_reductions(0.2, -0.7)
    check_rank_zero_reductions(0.2f0, -0.7f0)
    check_rank_zero_reductions(Int8(1), Int8(3))
    check_rank_zero_reductions(true, false, rank_zero_bool_reductions)

    # Adjacent array paths still reduce without scalarization.
    @test @jit(sum(Reactant.to_rarray(Int[]))) == sum(Int[])
    matrix = reshape(collect(1:4), 2, 2)
    @test @jit(sum(Reactant.to_rarray(matrix))) == sum(matrix)
end

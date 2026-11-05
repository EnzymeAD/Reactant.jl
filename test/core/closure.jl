using Test
using Reactant

muler(x) = y -> x * y

function boxed_muler(x)
    let b
        f = y -> b * y
        b = x
        return f
    end
end

function mapslices_boxed_capture(A)
    let c
        f = a -> a .+ c
        c = sum(A)
        return mapslices(f, A; dims=1)
    end
end

function mapslices_boxed_in_for(A, n)
    @trace track_numbers = false for _ in 1:n
        c = sum(A)
        A = mapslices(a -> a .+ c, A; dims=1)
    end
    return A
end

@testset "closure" begin
    x = ones(2, 2)
    y = ones(2, 2)
    x_ra = Reactant.to_rarray(x)
    y_ra = Reactant.to_rarray(y)

    f = muler(x_ra)
    @test @jit(f(y_ra)) ≈ x * y

    bf = boxed_muler(x_ra)
    @test Core.Box in fieldtypes(typeof(bf))
    @test @jit(bf(y_ra)) ≈ x * y

    A = ones(3, 2)
    A_ra = Reactant.to_rarray(copy(A))
    @test Array(@jit(mapslices_boxed_capture(A_ra))) ≈ mapslices_boxed_capture(A)

    A_ra = Reactant.to_rarray(copy(A))
    @test Array(@jit(mapslices_boxed_in_for(A_ra, 2))) ≈ mapslices_boxed_in_for(copy(A), 2)
end

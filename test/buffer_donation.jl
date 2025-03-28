using Test
using Reactant

# TODO try again with `2` to check automatic conversion from int to float
function donate_fill_x_with_2(x, y)
    x .= 2.0
    return nothing
end

function donate_inplace_mul(x, y)
    x .*= y
    return nothing
end

@testset "buffer_donation" begin
    a = Reactant.to_rarray(ones(2, 2))
    b = Reactant.to_rarray(3 * ones(2, 2))
    @jit(donate_fill_x_with_2(a, b))
    @test convert(Array, a) == 2 * ones(2, 2)
    hlo = @code_hlo(donate_fill_x_with_2(a, b))
    @test length(findall("reactant.donated", repr(hlo))) == 1

    (; preserved_args) = Reactant.Compiler.compile_xla(donate_fill_x_with_2, (a, b))[3]
    preserved_args_idx = last.(preserved_args)
    @test preserved_args_idx == [1] # only `y`(i.e. `b`) is preserved

    a = Reactant.to_rarray(2 * ones(2, 2))
    b = Reactant.to_rarray(3 * ones(2, 2))
    @jit(donate_inplace_mul(a, b))
    @test convert(Array, a) == 6 * ones(2, 2)
    hlo = @code_hlo(donate_inplace_mul(a, b))
    @test length(findall("reactant.donated", repr(hlo))) == 1

    (; preserved_args) = Reactant.Compiler.compile_xla(donate_inplace_mul, (a, b))[3]
    preserved_args_idx = last.(preserved_args)
    @test preserved_args_idx == [1] # only `y`(i.e. `b`) is preserved
end

function update_inplace!(x, y)
    x .+= y
    return nothing
end

function update_inplace_bad!(x, y)
    old = x[]
    x[] = old .+ y
    return old
end

@testset "buffer_donation" begin
    x = Reactant.to_rarray(ones(3))
    y = Reactant.to_rarray(ones(3))

    @code_hlo assert_nonallocating = true update_inplace!(x, y)

    @test_throws AssertionError @code_hlo assert_nonallocating = true update_inplace_bad!(
        Ref(x), y
    )
end
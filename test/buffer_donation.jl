using Reactant, Test

function donate_fill_x_with_2(x, y)
    x .= 2
    return nothing
end

function donate_inplace_mul(x, y)
    x .*= y
    return nothing
end

function multiple_donated_args(x, y, z)
    x .= 2.0
    y .= 3.0
    z .= 4.0
    return x, z, y
end

@testset "buffer_donation" begin
    a = Reactant.to_rarray(ones(2, 2))
    b = Reactant.to_rarray(3 * ones(2, 2))
    @jit(donate_fill_x_with_2(a, b))
    @test convert(Array, a) == 2 * ones(2, 2)
    hlo = @code_hlo(donate_fill_x_with_2(a, b))
    @test length(findall("tf.aliasing_output = 0", repr(hlo))) == 1

    (; preserved_args) = Reactant.Compiler.compile_xla(donate_fill_x_with_2, (a, b))[4]
    preserved_args_idx = last.(preserved_args)
    @test preserved_args_idx == [1] # only `y`(i.e. `b`) is preserved

    a = Reactant.to_rarray(2 * ones(2, 2))
    b = Reactant.to_rarray(3 * ones(2, 2))
    @jit(donate_inplace_mul(a, b))
    @test convert(Array, a) == 6 * ones(2, 2)
    hlo = @code_hlo(donate_inplace_mul(a, b))
    @test length(findall("tf.aliasing_output = 0", repr(hlo))) == 1

    (; preserved_args) = Reactant.Compiler.compile_xla(donate_inplace_mul, (a, b))[4]
    preserved_args_idx = last.(preserved_args)
    @test preserved_args_idx == [1] # only `y`(i.e. `b`) is preserved

    a = Reactant.to_rarray(ones(2, 2))
    b = Reactant.to_rarray(ones(3, 4))
    c = Reactant.to_rarray(ones(2, 2))
    @jit(multiple_donated_args(a, b, c))
    @test convert(Array, a) == 2 * ones(2, 2)
    @test convert(Array, b) == 3 * ones(3, 4)
    @test convert(Array, c) == 4 * ones(2, 2)
    hlo = @code_hlo(multiple_donated_args(a, b, c))
    @test contains(
        repr(hlo),
        "@main(%arg0: tensor<2x2xf64> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}, %arg1: tensor<4x3xf64> {enzymexla.memory_effects = [], tf.aliasing_output = 2 : i32}, %arg2: tensor<2x2xf64> {enzymexla.memory_effects = [], tf.aliasing_output = 1 : i32}) -> (tensor<2x2xf64>, tensor<2x2xf64>, tensor<4x3xf64>) attributes {enzymexla.memory_effects = []} {",
    )
end

function update_inplace!(x, y, z)
    x .+= y .* z
    return nothing
end

function update_inplace_bad!(x, y, z)
    old = x[]
    x[] = old .+ y .* z
    return old
end

@testset "buffer_donation" begin
    x = Reactant.to_rarray(ones(3))
    y = Reactant.to_rarray(ones(3))
    z = Reactant.to_rarray(ones(3))

    @code_hlo assert_nonallocating = true update_inplace!(x, y, z)
    (; preserved_args) = Reactant.Compiler.compile_xla(update_inplace!, (x, y, z))[4]
    preserved_args_idx = last.(preserved_args)
    @test preserved_args_idx == [1, 2] # y and z are both preserved (preserved_args is 0-indexed)

    @test_throws AssertionError @code_hlo assert_nonallocating = true update_inplace_bad!(
        Ref(x), y, z
    )
end

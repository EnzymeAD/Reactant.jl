using Reactant, Test
using LinearAlgebra
using Reactant.ReactantCore

function condition1(x)
    y = sum(x)
    @trace if y > 0
        z = y + 1
    else
        z = y - 1
    end
    return z
end

@testset "condition1" begin
    x = rand(2, 10)
    x_ra = Reactant.to_rarray(x)

    @test @jit(condition1(x_ra)) ≈ condition1(x)

    x = -rand(2, 10)
    x_ra = Reactant.to_rarray(x)

    @test @jit(condition1(x_ra)) ≈ condition1(x)
end

function condition1_missing_var(x)
    y = sum(x)
    @trace if y > 0
        z = y + 1
        p = -1
    else
        z = y - 1
    end
    return z
end

@testset "condition1_missing_var" begin
    x = rand(2, 10)
    x_ra = Reactant.to_rarray(x)

    @test @jit(condition1_missing_var(x_ra)) ≈ condition1_missing_var(x)

    x = -rand(2, 10)
    x_ra = Reactant.to_rarray(x)

    @test @jit(condition1_missing_var(x_ra)) ≈ condition1_missing_var(x)
end

@testset "return not supported" begin
    @test_throws LoadError @eval @trace if x > 0
        return 1
    end
end

function condition2_nested_if(x, y)
    x_sum = sum(x)
    @trace if x_sum > 0
        y_sum = sum(y)
        if y_sum > 0
            z = x_sum + y_sum
        else
            z = x_sum - y_sum
        end
    else
        y_sum = sum(y)
        z = x_sum - y_sum
    end
    return z
end

function condition2_if_else_if(x, y)
    x_sum = sum(x)
    y_sum = sum(y)
    @trace if x_sum > 0 && y_sum > 0
        z = x_sum + y_sum
    elseif x_sum > 0
        z = x_sum - y_sum
    else
        z = y_sum - x_sum
    end
    return z
end

@testset "condition2: multiple conditions" begin
    x = rand(2, 10)
    y = rand(2, 10)
    x_ra = Reactant.to_rarray(x)
    y_ra = Reactant.to_rarray(y)

    @test @jit(condition2_nested_if(x_ra, y_ra)) ≈ condition2_nested_if(x, y)
    @test @jit(condition2_if_else_if(x_ra, y_ra)) ≈ condition2_if_else_if(x, y)

    y = -rand(2, 10)
    y_ra = Reactant.to_rarray(y)

    @test @jit(condition2_nested_if(x_ra, y_ra)) ≈ condition2_nested_if(x, y)
    @test @jit(condition2_if_else_if(x_ra, y_ra)) ≈ condition2_if_else_if(x, y)

    x = -rand(2, 10)
    y = -rand(2, 10)
    x_ra = Reactant.to_rarray(x)
    y_ra = Reactant.to_rarray(y)

    @test @jit(condition2_nested_if(x_ra, y_ra)) ≈ condition2_nested_if(x, y)
    @test @jit(condition2_if_else_if(x_ra, y_ra)) ≈ condition2_if_else_if(x, y)
end

function condition3_mixed_conditions(x, y)
    x_sum = sum(x)
    y_sum = sum(y)
    @trace if x_sum > 0 && y_sum > 0
        z = x_sum + y_sum
    else
        z = -(x_sum + y_sum)
    end
    return z
end

@testset "condition3: mixed conditions" begin
    x = rand(2, 10)
    y = rand(2, 10)
    x_ra = Reactant.to_rarray(x)
    y_ra = Reactant.to_rarray(y)

    @test @jit(condition3_mixed_conditions(x_ra, y_ra)) ≈ condition3_mixed_conditions(x, y)

    x = -rand(2, 10)
    y = -rand(2, 10)
    x_ra = Reactant.to_rarray(x)
    y_ra = Reactant.to_rarray(y)

    @test @jit(condition3_mixed_conditions(x_ra, y_ra)) ≈ condition3_mixed_conditions(x, y)

    x = rand(2, 10)
    y = -rand(2, 10)
    x_ra = Reactant.to_rarray(x)
    y_ra = Reactant.to_rarray(y)
    @test @jit(condition3_mixed_conditions(x_ra, y_ra)) ≈ condition3_mixed_conditions(x, y)

    y = rand(2, 10)
    z = -rand(2, 10)
    y_ra = Reactant.to_rarray(y)
    z_ra = Reactant.to_rarray(z)
    @test @jit(condition3_mixed_conditions(x_ra, y_ra)) ≈ condition3_mixed_conditions(x, y)
end

function condition4_mixed_conditions(x, y)
    x_sum = sum(x)
    y_sum = sum(y)
    @trace if x_sum > 0 || y_sum > 0 && !(y_sum > 0)
        z = x_sum + y_sum
        p = 1
    else
        z = -(x_sum + y_sum)
        p = -1
    end
    return z
end

@testset "condition4: mixed conditions" begin
    x = rand(2, 10)
    y = rand(2, 10)
    x_ra = Reactant.to_rarray(x)
    y_ra = Reactant.to_rarray(y)

    @test @jit(condition4_mixed_conditions(x_ra, y_ra)) ≈ condition4_mixed_conditions(x, y)

    x = -rand(2, 10)
    y = -rand(2, 10)
    x_ra = Reactant.to_rarray(x)
    y_ra = Reactant.to_rarray(y)

    @test @jit(condition4_mixed_conditions(x_ra, y_ra)) ≈ condition4_mixed_conditions(x, y)

    x = rand(2, 10)
    y = -rand(2, 10)
    x_ra = Reactant.to_rarray(x)
    y_ra = Reactant.to_rarray(y)
    @test @jit(condition4_mixed_conditions(x_ra, y_ra)) ≈ condition4_mixed_conditions(x, y)

    y = rand(2, 10)
    z = -rand(2, 10)
    y_ra = Reactant.to_rarray(y)
    z_ra = Reactant.to_rarray(z)
    @test @jit(condition4_mixed_conditions(x_ra, y_ra)) ≈ condition4_mixed_conditions(x, y)
end

function condition5_multiple_returns(x, y)
    x_sum = sum(x)
    y_sum = sum(y)
    @trace if x_sum > 0
        z = x_sum + y_sum
        p = 1
    else
        z = -(x_sum + y_sum)
        p = -1
    end
    return z, p
end

@testset "condition5: multiple returns" begin
    x = rand(2, 10)
    y = rand(2, 10)
    x_ra = Reactant.to_rarray(x)
    y_ra = Reactant.to_rarray(y)

    res_ra = @jit(condition5_multiple_returns(x_ra, y_ra))
    res = condition5_multiple_returns(x, y)
    @test res_ra[1] ≈ res[1]
    @test res_ra[2] ≈ res[2]
end

function condition6_bareif_relu(x)
    @trace if x < 0
        x = 0.0
    end
    return x
end

@testset "condition6: bareif relu" begin
    x = 2.0
    x_ra = Reactant.to_rarray(x; track_numbers=Number)

    res_ra = @jit(condition6_bareif_relu(x_ra))
    res = condition6_bareif_relu(x)
    @test res_ra ≈ res

    x = -2.0
    x_ra = Reactant.to_rarray(x; track_numbers=Number)

    res_ra = @jit(condition6_bareif_relu(x_ra))
    res = condition6_bareif_relu(x)
    @test res_ra ≈ res
end

function condition7_bare_elseif(x)
    @trace if x > 0
        x = x + 1
    elseif x < 0
        x = x - 1
    elseif x == 0
        x = x
    end
    return x
end

@testset "condition7: bare elseif" begin
    x = 2.0
    x_ra = Reactant.to_rarray(x; track_numbers=Number)

    res_ra = @jit(condition7_bare_elseif(x_ra))
    res = condition7_bare_elseif(x)
    @test res_ra ≈ res

    x = -2.0
    x_ra = Reactant.to_rarray(x; track_numbers=Number)

    res_ra = @jit(condition7_bare_elseif(x_ra))
    res = condition7_bare_elseif(x)
    @test res_ra ≈ res

    x = 0.0
    x_ra = Reactant.to_rarray(x; track_numbers=Number)

    res_ra = @jit(condition7_bare_elseif(x_ra))
    res = condition7_bare_elseif(x)
    @test res_ra ≈ res
end

function condition8_return_if(x)
    @trace (y, z) = if sum(x) > 0
        -1, 2.0
    elseif sum(x) < 0
        1, -2.0
    else
        0, 0.0
    end
    return y, z
end

@testset "condition8: return if" begin
    x = rand(2, 10)
    x_ra = Reactant.to_rarray(x)

    res_ra = @jit(condition8_return_if(x_ra))
    res = condition8_return_if(x)
    @test res_ra[1] ≈ res[1]
    @test res_ra[2] ≈ res[2]

    x = -rand(2, 10)
    x_ra = Reactant.to_rarray(x)

    res_ra = @jit(condition8_return_if(x_ra))
    res = condition8_return_if(x)
    @test res_ra[1] ≈ res[1]
    @test res_ra[2] ≈ res[2]

    x = zeros(2, 10)
    x_ra = Reactant.to_rarray(x)

    res_ra = @jit(condition8_return_if(x_ra))
    res = condition8_return_if(x)
    @test res_ra[1] ≈ res[1]
    @test res_ra[2] ≈ res[2]
end

function condition9_if_ends_with_nothing(x)
    @trace if sum(x) > 0
        y = 1.0
        nothing
    else
        y = 2.0
    end
    return y
end

@testset "condition9: if ends with nothing" begin
    x = rand(2, 10)
    x_ra = Reactant.to_rarray(x)

    res_ra = @jit(condition9_if_ends_with_nothing(x_ra))
    res = condition9_if_ends_with_nothing(x)
    @test res_ra ≈ res

    x = -rand(2, 10)
    x_ra = Reactant.to_rarray(x)

    res_ra = @jit(condition9_if_ends_with_nothing(x_ra))
    res = condition9_if_ends_with_nothing(x)
    @test res_ra ≈ res
end

function condition9_if_ends_with_pathological_nothing(x)
    @trace if sum(x) > 0
        y = 1.0
        nothing = 2.0
    else
        y = 2.0
        nothing = 3.0
    end
    return y, nothing
end

@testset "condition9: if ends with pathological nothing" begin
    x = rand(2, 10)
    x_ra = Reactant.to_rarray(x)

    res_ra = @jit(condition9_if_ends_with_pathological_nothing(x_ra))
    res = condition9_if_ends_with_pathological_nothing(x)
    @test res_ra[1] ≈ res[1]
    @test res_ra[2] ≈ res[2]
end

function condition10_condition_with_setindex(x)
    @trace if sum(x) > 0
        x[:, 1] = -1.0
    else
        @allowscalar x[1, 1] = 1.0
    end
    return x
end

@testset "condition10: condition with setindex!" begin
    x = rand(2, 10)
    x_ra = Reactant.to_rarray(x)

    res_ra = @jit(condition10_condition_with_setindex(x_ra))
    @test @allowscalar(res_ra[1, 1]) == -1.0
    @test @allowscalar(res_ra[2, 1]) == -1.0
    @test @allowscalar(x_ra[1, 1]) == -1.0
    @test @allowscalar(x_ra[2, 1]) == -1.0

    x = -rand(2, 10)
    x[2, 1] = 0.0
    x_ra = Reactant.to_rarray(x)

    res_ra = @jit(condition10_condition_with_setindex(x_ra))
    @test @allowscalar(res_ra[1, 1]) == 1.0
    @test @allowscalar(res_ra[2, 1]) == 0.0
    @test @allowscalar(x_ra[1, 1]) == 1.0
    @test @allowscalar(x_ra[2, 1]) == 0.0
end

function condition11_nested_ifff(x, y, z)
    x_sum = sum(x)
    @trace if x_sum > 0
        y_sum = sum(y)
        if y_sum > 0
            if sum(z) > 0
                z = x_sum + y_sum + sum(z)
            else
                z = x_sum + y_sum
            end
        else
            z = x_sum - y_sum
        end
    else
        y_sum = sum(y)
        z = x_sum - y_sum
    end
    return z
end

@testset "condition11: nested if 3 levels deep" begin
    x = rand(2, 10)
    y = rand(2, 10)
    z = rand(2, 10)
    x_ra = Reactant.to_rarray(x)
    y_ra = Reactant.to_rarray(y)
    z_ra = Reactant.to_rarray(z)

    @test @jit(condition11_nested_ifff(x_ra, y_ra, z_ra)) ≈ condition11_nested_ifff(x, y, z)

    x = -rand(2, 10)
    y = -rand(2, 10)
    z = -rand(2, 10)
    x_ra = Reactant.to_rarray(x)
    y_ra = Reactant.to_rarray(y)
    z_ra = Reactant.to_rarray(z)

    @test @jit(condition11_nested_ifff(x_ra, y_ra, z_ra)) ≈ condition11_nested_ifff(x, y, z)
end

function condition12_compile_test(x, y, z)
    x_sum = sum(x)
    @trace if x_sum > 0
        y_sum = sum(y)
        z = x_sum + y_sum + sum(z)
    else
        y_sum = sum(y)
        z = x_sum - y_sum
    end
    return z
end

@testset "condition12: compile test" begin
    x = rand(2, 10)
    y = rand(2, 10)
    z = rand(2, 10)
    x_ra = Reactant.to_rarray(x)
    y_ra = Reactant.to_rarray(y)
    z_ra = Reactant.to_rarray(z)

    @test @jit(condition12_compile_test(x_ra, y_ra, z_ra)) ≈
        condition12_compile_test(x, y, z)

    x = -rand(2, 10)
    y = -rand(2, 10)
    z = -rand(2, 10)
    x_ra = Reactant.to_rarray(x)
    y_ra = Reactant.to_rarray(y)
    z_ra = Reactant.to_rarray(z)

    @test @jit(condition12_compile_test(x_ra, y_ra, z_ra)) ≈
        condition12_compile_test(x, y, z)
end

function condition_with_structure(x)
    y = x .+ 1
    @trace if sum(y) > 0
        z = (; a=y, b=(y .- 1, y))
    else
        z = (; a=(-y), b=(y, y .+ 1))
    end
    return z
end

@testset "condition with structure" begin
    x = rand(2, 10)
    x_ra = Reactant.to_rarray(x)

    res_ra = @jit condition_with_structure(x_ra)
    res = condition_with_structure(x)
    @test res_ra.a ≈ res.a
    @test res_ra.b[1] ≈ res.b[1]
    @test res_ra.b[2] ≈ res.b[2]

    x = -rand(2, 10)
    x_ra = Reactant.to_rarray(x)

    res_ra = @jit condition_with_structure(x_ra)
    res = condition_with_structure(x)
    @test res_ra.a ≈ res.a
    @test res_ra.b[1] ≈ res.b[1]
    @test res_ra.b[2] ≈ res.b[2]
end

function for_with_step(x)
    @trace for i in 10:3:22
        @allowscalar x[i] = i * i
    end
    return x
end

@testset "for: for with step" begin
    x = rand(1:100, 22)
    x_ra = Reactant.to_rarray(x)

    @test @jit(for_with_step(x_ra)) == for_with_step(x)
end

function nnorm(x, n)
    @trace for i in 1:n
        x = x * i ./ sum(x)
    end
    return x
end

@testset "for: induction" begin
    x = randn(Float32, 10)
    x_ra = Reactant.to_rarray(x)

    n = 10

    @test @jit(nnorm(x_ra, n)) ≈ nnorm(x, n)
end

function sinkhorn(μ, ν, C)
    λ = eltype(C)(0.8)
    K = @. exp(-C / λ)

    u = fill!(similar(μ), one(eltype(μ)))
    v = similar(ν)

    @trace for _ in 1:10
        v = ν ./ (K' * u)
        u = μ ./ (K * v)
    end

    return Diagonal(u) * K * Diagonal(v)
end

@testset "for: sinkhorn" begin
    Nμ = 10
    Nν = 5

    μ = ones(Float32, Nμ) ./ Nμ
    ν = ones(Float32, Nν) ./ Nν
    C = randn(Float32, Nμ, Nν)

    μ_ra = Reactant.to_rarray(μ)
    ν_ra = Reactant.to_rarray(ν)
    C_ra = Reactant.to_rarray(C)

    @test @jit(sinkhorn(μ_ra, ν_ra, C_ra)) ≈ sinkhorn(μ, ν, C)
end

@testset "for: forbidden syntax" begin
    @test_throws "break" @eval function f_with_break()
        @trace for i in 1:10
            break
        end
    end

    @test_throws "continue" @eval function f_with_continue()
        @trace for i in 1:10
            continue
        end
    end

    @test_throws "return" @eval function f_with_return()
        @trace for i in 1:10
            return nothing
        end
    end
end

function cumsum!(x)
    v = zero(eltype(x))
    @trace for i in 1:length(x)
        v += @allowscalar x[i]
        @allowscalar x[i] = v
    end
    return x
end

@testset "for: mutation within loop" begin
    x = rand(1:100, 10)
    x_ra = Reactant.to_rarray(x)

    @test @jit(cumsum!(x_ra)) == cumsum!(x)
end

function for_ref_outer(x)
    i = sum(x)
    @trace for i in 1:length(x)
        x .+= i
    end
    return x / i
end

@testset "for: outer reference" begin
    x = randn(Float64, 10)
    x_ra = Reactant.to_rarray(x)

    @test @jit(for_ref_outer(x_ra)) ≈ for_ref_outer(x)
end

function for_inner_scope(x)
    @trace for i in 1:10
        s = sum(x)
        x = x / s
    end
    return x
end

@testset "for: inner scope" begin
    x = randn(Float64, 10)
    x_ra = Reactant.to_rarray(x)

    @test @jit(for_inner_scope(x_ra)) ≈ for_inner_scope(x)
end

function for_with_named_tuple(x)
    st = (; x)
    res = x
    @trace for i in 1:10
        res .= res .+ st.x
    end
    return res
end

@testset "for: named tuple" begin
    x = randn(Float64, 10)
    x_ra = Reactant.to_rarray(x)

    @test @jit(for_with_named_tuple(x_ra)) ≈ for_with_named_tuple(x)
end

mutable struct Container{A,B,C}
    a::A
    b::B
    c::C
end

function for_in_container(ctr)
    dt = copy(ctr.a)
    @trace for i in 1:10
        ctr.b .+= ctr.c * dt
    end
end

@testset "for: container" begin
    x = Container(3.1, [1.4], [2.7])
    x_ra = Reactant.to_rarray(x)

    @jit(for_in_container(x_ra))
    for_in_container(x)

    @test x.a ≈ x_ra.a
    @test x.b ≈ x_ra.b
    @test x.c ≈ x_ra.c
end

_call1(a, b) = a
function call1(a, b)
    x = @trace _call1(a, b)
    y = @trace _call1(a, b)
    return @trace _call1(x, y)
end

@testset "call: basic" begin
    a = rand(2, 3)
    b = rand(2, 3)
    a_ra = Reactant.to_rarray(a)
    b_ra = Reactant.to_rarray(b)

    @test @jit(call1(a_ra, b_ra)) ≈ call1(a, b)

    # check whether the func for _call1 was only generated once:
    ir = @code_hlo optimize = false call1(a_ra, b_ra)
    ops = [op for op in Reactant.MLIR.IR.OperationIterator(Reactant.MLIR.IR.body(ir))]
    @test length(ops) == 2 # call1, _call1

    # With different operand sizes, different functions need to be generated:
    c = rand(4, 5)
    c_ra = Reactant.to_rarray(c)

    @test @jit(call1(a_ra, c_ra)) ≈ call1(a, c)
    ir = @code_hlo optimize = false call1(a_ra, c_ra)
    ops = [op for op in Reactant.MLIR.IR.OperationIterator(Reactant.MLIR.IR.body(ir))]
    @test length(ops) == 3
end

_call2(a) = a + a
function call2(a)
    return @trace _call2(a)
end

@testset "call: rnumber" begin
    a = 10
    a_rn = ConcreteRNumber(a)

    @test @jit(call2(a_rn)) == call2(a)
end

function _call3(x::Int, y)
    if x > 10
        return y .+ y
    else
        return y .* y
    end
end

function call3(y)
    z = @trace _call3(1, y)
    @trace _call3(1, z) # doesn't generate new function because y.shape == z.shape
    @trace _call3(11, y) # new function because x changed.
end

@testset "call: caching for Julia operands" begin
    y = rand(3)
    y_ra = Reactant.to_rarray(y)

    ir = @code_hlo optimize = false call3(y_ra)
    ops = [op for op in Reactant.MLIR.IR.OperationIterator(Reactant.MLIR.IR.body(ir))]
    @test length(ops) == 5 # call3, .+, .*, _call3 (2X)
end

struct Foo
    x
end
struct Bar
    x
end

_call4(foobar::Union{Foo,Bar}) = foobar.x
function call4(foo, foo2, bar)
    @trace _call4(foo)
    @trace _call4(foo2)
    @trace _call4(bar)
end

@testset "call: Caching struct arguments" begin
    a = rand(10)
    b = rand(10)
    foo = Foo(Reactant.to_rarray(a))
    foo2 = Foo(Reactant.to_rarray(b))
    bar = Foo(Bar(Reactant.to_rarray(b))) # typeof(foo) == typeof(bar), but these don't match!
    ir = @code_hlo optimize = false call4(foo, foo2, bar)
    ops = [op for op in Reactant.MLIR.IR.OperationIterator(Reactant.MLIR.IR.body(ir))]
    @test length(ops) == 3 # call4, _call4 for {foo, foo2}, and _call4 for bar
end

function _call5!(a, b)
    @allowscalar a[1] = zero(eltype(a))
    return b
end

function call5!(a, b)
    @trace _call5!(a, b)
    return a
end

@testset "call: argument mutation" begin
    a = ones(3)
    b = ones(3)
    a_ra = Reactant.to_rarray(a)
    b_ra = Reactant.to_rarray(b)
    @jit call5!(a_ra, b_ra)
    call5!(a, b)
    @test a_ra == a
end

mutable struct TestClock{I}
    iteration::I
end

mutable struct TestSimulation{C,I,B}
    clock::C
    stop_iteration::I
    running::B
end

function step!(sim)
    @trace if sim.clock.iteration >= sim.stop_iteration
        sim.running = false
    else
        sim.clock.iteration += 1 # time step
    end
    return nothing
end

function simulate!(sim)
    return ReactantCore.traced_while(sim -> sim.running, step!, (sim,))
end

@testset "simulation loop" begin
    clock = TestClock(ConcreteRNumber(0))
    simulation = TestSimulation(clock, ConcreteRNumber(3), ConcreteRNumber(true))

    f! = @compile sync = true simulate!(simulation)
    result = f!(simulation)

    @test result == [3, 3, false]
    @test simulation.running == false
    @test simulation.clock.iteration == 3
    @test simulation.stop_iteration == 3
end

function ternary_max(x, y)
    @trace result = x > y ? x : y
    return result
end

@testset "ternary operator return value" begin
    a, b = ConcreteRNumber(1), ConcreteRNumber(2)

    @test (@jit ternary_max(a, b)) == 2
end

mutable struct MaybeTraced
    x
end

@testset "is_traced of struct" begin
    containstraced = MaybeTraced(
        MaybeTraced(Reactant.TracedRArray{Float64,1}((), nothing, (3,)))
    )
    @test Reactant.ReactantCore.is_traced(containstraced)

    doesnotcontaintraced = MaybeTraced(MaybeTraced(3))
    @test !Reactant.ReactantCore.is_traced(doesnotcontaintraced)

    recursivetraced = MaybeTraced((
        1,
        "string",
        MaybeTraced(nothing),
        MaybeTraced(Reactant.TracedRArray{Float64,1}((), nothing, (3,))),
    ))
    recursivetraced.x[3].x = recursivetraced
    @test Reactant.ReactantCore.is_traced(recursivetraced)

    recursivenottraced = MaybeTraced((1, "string", MaybeTraced(nothing)))
    recursivenottraced.x[3].x = recursivenottraced
    @test !Reactant.ReactantCore.is_traced(recursivenottraced)
end

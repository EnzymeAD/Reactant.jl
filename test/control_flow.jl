using Reactant, Test
using LinearAlgebra

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
    x_ra = Reactant.to_rarray(x; track_numbers=(Number,))

    res_ra = @jit(condition6_bareif_relu(x_ra))
    res = condition6_bareif_relu(x)
    @test res_ra ≈ res

    x = -2.0
    x_ra = Reactant.to_rarray(x; track_numbers=(Number,))

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
    x_ra = Reactant.to_rarray(x; track_numbers=(Number,))

    res_ra = @jit(condition7_bare_elseif(x_ra))
    res = condition7_bare_elseif(x)
    @test res_ra ≈ res

    x = -2.0
    x_ra = Reactant.to_rarray(x; track_numbers=(Number,))

    res_ra = @jit(condition7_bare_elseif(x_ra))
    res = condition7_bare_elseif(x)
    @test res_ra ≈ res

    x = 0.0
    x_ra = Reactant.to_rarray(x; track_numbers=(Number,))

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
        x[1, 1] = 1.0
    end
    return x
end

@testset "condition10: condition with setindex!" begin
    x = rand(2, 10)
    x_ra = Reactant.to_rarray(x)

    res_ra = @jit(condition10_condition_with_setindex(x_ra))
    @test res_ra[1, 1] == -1.0
    @test res_ra[2, 1] == -1.0
    @test x_ra[1, 1] == -1.0 broken = true
    @test x_ra[2, 1] == -1.0 broken = true

    x = -rand(2, 10)
    x[2, 1] = 0.0
    x_ra = Reactant.to_rarray(x)

    res_ra = @jit(condition10_condition_with_setindex(x_ra))
    @test res_ra[1, 1] == 1.0
    @test res_ra[2, 1] == 0.0
    @test x_ra[1, 1] == 1.0 broken = true
    @test x_ra[2, 1] == 0.0
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

function nnorm(x, n)
    @trace for i in 1:n
       x = x * i ./ sum(x)
   end
   x
end

@testset "for: induction" begin
    x = randn(Float32, 10)
    x_ra = Reactant.to_rarray(x);

    n = 10
    n_ra = Reactant.to_rarray(fill(n));

    @test @jit(nnorm(x_ra, n_ra)) ≈ nnorm(x, n)
end

function sinkhorn(μ, ν, C)
    λ = eltype(C)(0.8)
    K = @. exp(-C/λ)

    u = fill!(similar(μ), one(eltype(μ)))
    v = similar(ν)

    @trace for _ in 1:10
        v = ν ./ (K' * u)
        u = μ ./ (K * v)
    end

    Diagonal(u) * K * Diagonal(v)
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

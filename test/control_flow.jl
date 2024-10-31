using Reactant, Test

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
        @trace if y_sum > 0
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

    @test @jit(condition2_nested_if(x_ra, y_ra)) ≈ condition2_nested_if(x, y) broken = true
    @test @jit(condition2_if_else_if(x_ra, y_ra)) ≈ condition2_if_else_if(x, y)

    y = -rand(2, 10)
    y_ra = Reactant.to_rarray(y)

    @test @jit(condition2_nested_if(x_ra, y_ra)) ≈ condition2_nested_if(x, y) broken = true
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

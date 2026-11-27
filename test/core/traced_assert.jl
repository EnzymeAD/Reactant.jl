using Reactant
using Enzyme
using Test

function issue_2678_elem(sinphi, m)
    cond = (abs(sinphi) == 1) & (m == 1)

    res = sinphi
    @trace if cond
        res = sign(sinphi) * Inf
        nothing
    else
        sinphi2 = sinphi^2
        drf, ierr = 1 - sinphi2, Base.ifelse(m < 1.5, 0, 1)
        @assert ierr == 0
        res = sinphi * drf
        nothing
    end
    return res
end

broadcast_issue_2678(x, y) = issue_2678_elem.(x, y)

function traced_if_return_with_assert(x, y)
    @trace res = if x > y
        diff = x - y
        @assert diff > 0 "expected positive diff, got $diff"
        diff * x
    else
        diff = y - x
        @assert diff >= 0 && !(diff < 0 && x > y)
        diff + y
    end
    return res
end

function traced_for_with_assert(x)
    acc = zero(eltype(x))
    @trace for i in eachindex(x)
        xi = @allowscalar x[i]
        @assert isfinite(xi) && (xi >= -1000 || xi < 1000) "invalid element $xi"
        acc = acc + xi * xi
    end
    return acc
end

function traced_while_with_assert(x)
    cur = x
    steps = 0
    @trace while cur > 1
        @assert cur > 0 "cur must stay positive: $cur"
        cur = cur * 0.5
        steps += 1
    end
    return cur + steps
end

@trace function traced_fn_def_with_assert(x)
    @assert x > 0 "x must be positive, got $x"
    return log(x) + x^2
end

function standalone_traced_assert(x)
    @trace @assert x >= 0 "negative input $x"
    return sqrt(x)
end

macro custom_nonneg_assert(val)
    return esc(:(@assert $val >= 0))
end

function macro_expanded_assert_in_trace(x)
    out = x
    @trace if x >= 0
        @custom_nonneg_assert x
        out = sqrt(x)
    else
        out = -sqrt(-x)
    end
    return out
end

function shortcircuit_host_bool_in_trace(x, opt)
    res = x
    @trace if x > 0
        @assert opt === nothing || opt > 0
        @assert !(opt !== nothing && opt <= 0)
        res = x + 1
    else
        res = x - 1
    end
    return res
end

function static_false_assert_in_trace(x)
    res = x
    @trace if sum(x) > 0
        @assert ndims(x) == 3 "expected 3D array"
        res = x .+ 1
    end
    return res
end

function non_bool_traced_assert(x)
    @trace @assert x
    return x
end

loss_issue_2678(x, y) = sum(issue_2678_elem.(x, y))
grad_rev_issue_2678(x, y) = Enzyme.gradient(Reverse, loss_issue_2678, x, Const(y))[1]
grad_fwd_for_assert(x) = Enzyme.gradient(Forward, traced_for_with_assert, x)[1]

@testset "Traced @assert (#2678)" begin
    @testset "Issue #2678 broadcast MWE" begin
        x_h = Float64[0.2, -0.5, 1.0, 0.8]
        y_h = Float64[0.3, 0.7, 1.0, 1.2]
        expected = broadcast_issue_2678(x_h, y_h)

        x_r = Reactant.to_rarray(x_h)
        y_r = Reactant.to_rarray(y_h)

        compiled = @compile broadcast_issue_2678(x_r, y_r)
        res_compiled = compiled(x_r, y_r)
        @test Array(res_compiled) ≈ expected

        res_jit = @jit broadcast_issue_2678(x_r, y_r)
        @test Array(res_jit) ≈ expected

        # Plain Julia execution still enforces the assertion
        @test issue_2678_elem(0.5, 0.5) ≈ 0.5 * (1 - 0.25)
        @test_throws AssertionError issue_2678_elem(0.5, 2.0)
    end

    @testset "@trace if return, for, while, function, and standalone @assert" begin
        a_r = ConcreteRNumber(3.5)
        b_r = ConcreteRNumber(1.25)
        @test @jit(traced_if_return_with_assert(a_r, b_r)) ≈
            traced_if_return_with_assert(3.5, 1.25)
        @test @jit(traced_if_return_with_assert(b_r, a_r)) ≈
            traced_if_return_with_assert(1.25, 3.5)

        v_h = Float32[1.5, -2.0, 3.0, 0.5]
        v_r = Reactant.to_rarray(v_h)
        @test @jit(traced_for_with_assert(v_r)) ≈ traced_for_with_assert(v_h)

        w_r = ConcreteRNumber(6.0)
        @test @jit(traced_while_with_assert(w_r)) ≈ traced_while_with_assert(6.0)

        @test @jit(traced_fn_def_with_assert(ConcreteRNumber(2.5))) ≈
            traced_fn_def_with_assert(2.5)
        @test traced_fn_def_with_assert(2.5) ≈ log(2.5) + 2.5^2
        @test_throws AssertionError traced_fn_def_with_assert(-1.0)

        @test @jit(standalone_traced_assert(ConcreteRNumber(4.0))) ≈ 2.0
        @test standalone_traced_assert(4.0) ≈ 2.0
        @test_throws AssertionError standalone_traced_assert(-4.0)

        @test @jit(macro_expanded_assert_in_trace(ConcreteRNumber(9.0))) ≈ 3.0
        @test @jit(macro_expanded_assert_in_trace(ConcreteRNumber(-9.0))) ≈ -3.0
    end

    @testset "Short-circuiting, static false assertions, and TypeError" begin
        @test @jit(shortcircuit_host_bool_in_trace(ConcreteRNumber(2.0), nothing)) ≈ 3.0
        @test @jit(shortcircuit_host_bool_in_trace(ConcreteRNumber(2.0), 5)) ≈ 3.0
        @test_throws AssertionError @compile shortcircuit_host_bool_in_trace(
            ConcreteRNumber(2.0), -1
        )

        v_r = Reactant.to_rarray(Float32[1.0, 2.0])
        @test_throws AssertionError @compile static_false_assert_in_trace(v_r)

        @test_throws TypeError @compile non_bool_traced_assert(ConcreteRNumber(1))
        @test_throws TypeError non_bool_traced_assert(1)
    end

    @testset "Enzyme autodiff compatibility" begin
        x_h = Float64[0.2, -0.5, 0.4, 0.8]
        y_h = Float64[0.3, 0.7, 0.9, 1.2]
        x_r = Reactant.to_rarray(x_h)
        y_r = Reactant.to_rarray(y_h)

        dx = @jit grad_rev_issue_2678(x_r, y_r)
        expected_dx = 1 .- 3 .* (x_h .^ 2)
        @test Array(dx) ≈ expected_dx

        v_h = Float64[1.5, -2.0, 3.0, 0.5]
        v_r = Reactant.to_rarray(v_h)
        dv = @jit grad_fwd_for_assert(v_r)
        @test Array(dv) ≈ 2 .* v_h
    end
end

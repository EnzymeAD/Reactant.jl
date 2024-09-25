using LuxLib, Reactant, Enzyme, NNlib

@testset "Fused Dense" begin end

@testset "Bias Activation" begin end

@testset "Fast Activation" begin
    # Here we are testing that fast_activation doesn't switch to the faster versions
    sumabs2(f, x) = sum(abs2, fast_activation(f, x))
    sumabs2!!(f, x) = sum(abs2, fast_activation!!(f, copy(x)))

    function ∇sumabs2(f, x)
        dx = Enzyme.make_zero(x)
        Enzyme.autodiff(Reverse, sumabs2, Active, Const(f), Duplicated(x, dx))
        return dx
    end

    function ∇sumabs2!!(f, x)
        dx = Enzyme.make_zero(x)
        Enzyme.autodiff(Reverse, sumabs2, Active, Const(f), Duplicated(x, dx))
        return dx
    end

    x_act = randn(Float32, 10, 10)
    x_act_ca = Reactant.ConcreteRArray(x_act)

    @testset "Activation: $act" for act in (
        identity, relu, sigmoid, tanh, tanh_fast, sigmoid_fast, gelu, abs2
    )
        f_compile = Reactant.compile(sumabs2, (act, x_act))
        f_compile!! = Reactant.compile(sumabs2!!, (act, x_act))

        y_simple = sumabs2(act, x_act)
        y_simple!! = sumabs2!!(act, x_act)
        y_compile = f_compile(act, x_act_ca)
        y_compile!! = f_compile!!(act, x_act_ca)

        @test y_simple ≈ y_compile
        @test y_simple!! ≈ y_compile!!

        ∂x_enz = Enzyme.make_zero(x_act)
        Enzyme.autodiff(Reverse, sumabs2, Active, Const(act), Duplicated(x_act, ∂x_enz))

        ∂x_enz!! = Enzyme.make_zero(x_act)
        Enzyme.autodiff(Reverse, sumabs2!!, Active, Const(act), Duplicated(x_act, ∂x_enz!!))

        ∇sumabs2_compiled = Reactant.compile(∇sumabs2, (act, x_act_ca))
        ∂x_compile = ∇sumabs2_compiled(act, x_act_ca)

        ∇sumabs2!!_compiled = Reactant.compile(∇sumabs2!!, (act, x_act_ca))
        ∂x_compile!! = ∇sumabs2!!_compiled(act, x_act_ca)

        @test ∂x_enz ≈ ∂x_compile broken = (act === gelu)
        @test ∂x_enz!! ≈ ∂x_compile!! broken = (act === gelu)
    end
end

@testset "Fused Conv" begin end

using SpecialFunctions, Reactant, Test

const RunningOnMetal = contains(string(Reactant.devices()[1]), "MTL")

call_polygamma_int(x) = SpecialFunctions.polygamma(2, x)
call_logabsbeta_left(b) = SpecialFunctions.logabsbeta(-1.5, b)
call_logabsbeta_right(a) = SpecialFunctions.logabsbeta(a, 0.6)
call_logabsbinomial_left(k) = SpecialFunctions.logabsbinomial(10, k)
call_logabsbinomial_right(n) = SpecialFunctions.logabsbinomial(n, 3)

RunningOnMetal || @testset "polygamma, trigamma, and logabs* (#2262)" begin
    @testset "trigamma and polygamma precision & integer order" begin
        for x in (0.5f0, 1.5f0, 3.25)
            rx = ConcreteRNumber(x)
            rtri = @jit SpecialFunctions.trigamma(rx)
            @test Reactant.to_number(rtri) isa typeof(SpecialFunctions.trigamma(x))
            @test isapprox(
                Reactant.to_number(rtri), SpecialFunctions.trigamma(x); rtol=1e-5
            )

            rpoly_const = @jit call_polygamma_int(rx)
            @test Reactant.to_number(rpoly_const) isa
                typeof(SpecialFunctions.polygamma(2, x))
            @test isapprox(
                Reactant.to_number(rpoly_const),
                SpecialFunctions.polygamma(2, x);
                rtol=1e-5,
            )

            rpoly_traced = @jit SpecialFunctions.polygamma(ConcreteRNumber(2), rx)
            @test Reactant.to_number(rpoly_traced) isa
                typeof(SpecialFunctions.polygamma(2, x))
            @test isapprox(
                Reactant.to_number(rpoly_traced),
                SpecialFunctions.polygamma(2, x);
                rtol=1e-5,
            )
        end

        xa = Reactant.to_rarray(Float32[0.5, 1.5, 2.5])
        @test isapprox(
            Array(@jit(SpecialFunctions.trigamma.(xa))),
            SpecialFunctions.trigamma.(Float32[0.5, 1.5, 2.5]);
            rtol=1e-5,
        )
        @test isapprox(
            Array(@jit(call_polygamma_int.(xa))),
            SpecialFunctions.polygamma.(2, Float32[0.5, 1.5, 2.5]);
            rtol=1e-5,
        )
    end

    @testset "logabsgamma" begin
        for x in (2.5f0, 0.5, -0.5, -1.5f0, -2.5, -3.5, 0.0, -1.0, -2.0, Inf, -Inf, NaN)
            lg_ref, s_ref = SpecialFunctions.logabsgamma(x)
            lg_got_r, s_got_r = @jit SpecialFunctions.logabsgamma(ConcreteRNumber(x))
            lg_got = Reactant.to_number(lg_got_r)
            s_got = Reactant.to_number(s_got_r)
            @test s_got === s_ref
            if isnan(lg_ref)
                @test isnan(lg_got)
            elseif isinf(lg_ref)
                @test lg_got == lg_ref
            else
                @test isapprox(lg_got, lg_ref; rtol=1e-5, atol=1e-6)
            end
        end
    end

    @testset "logabsbeta and overflow-safe logbeta" begin
        for (a, b) in (
            (0.5f0, 0.6f0),
            (-1.5, 0.6),
            (-2.5, -1.5),
            (-2.0, 3.0),
            (-2.0, -1.0),
            (-5.0, 2.0),
            (250.0, 300.0),
        )
            lb_ref, s_ref = SpecialFunctions.logabsbeta(a, b)
            lb_got_r, s_got_r = @jit SpecialFunctions.logabsbeta(
                ConcreteRNumber(a), ConcreteRNumber(b)
            )
            lb_got = Reactant.to_number(lb_got_r)
            s_got = Reactant.to_number(s_got_r)
            @test s_got === s_ref
            if isinf(lb_ref)
                @test lb_got == lb_ref
            else
                @test isapprox(lb_got, lb_ref; rtol=1e-5, atol=1e-6)
            end
        end

        @test isapprox(
            Reactant.to_number(
                @jit(
                    SpecialFunctions.logbeta(
                        ConcreteRNumber(250.0), ConcreteRNumber(300.0)
                    )
                )
            ),
            SpecialFunctions.logbeta(250.0, 300.0);
            rtol=1e-6,
        )

        lb_l, s_l = @jit call_logabsbeta_left(ConcreteRNumber(0.6))
        @test Reactant.to_number(s_l) === -1
        @test isapprox(
            Reactant.to_number(lb_l),
            SpecialFunctions.logabsbeta(-1.5, 0.6)[1];
            rtol=1e-6,
        )

        lb_r, s_r = @jit call_logabsbeta_right(ConcreteRNumber(-1.5))
        @test Reactant.to_number(s_r) === -1
        @test isapprox(
            Reactant.to_number(lb_r),
            SpecialFunctions.logabsbeta(-1.5, 0.6)[1];
            rtol=1e-6,
        )
    end

    @testset "logabsbinomial" begin
        for (n, k) in (
            (5, 2),
            (5, 0),
            (5, 5),
            (5, 6),
            (5, -1),
            (-5, 2),
            (-5, 3),
            (10, 1),
            (Int32(20), Int32(10)),
            (1000, 500),
        )
            v_ref, s_ref = SpecialFunctions.logabsbinomial(n, k)
            v_got_r, s_got_r = @jit SpecialFunctions.logabsbinomial(
                ConcreteRNumber(n), ConcreteRNumber(k)
            )
            v_got = Reactant.to_number(v_got_r)
            s_got = Reactant.to_number(s_got_r)
            @test s_got == s_ref
            if isinf(v_ref)
                @test v_got == v_ref
            else
                @test isapprox(v_got, v_ref; rtol=1e-6, atol=1e-8)
            end
        end

        vl, sl = @jit call_logabsbinomial_left(ConcreteRNumber(3))
        @test Reactant.to_number(sl) == 1.0
        @test isapprox(
            Reactant.to_number(vl), SpecialFunctions.logabsbinomial(10, 3)[1]; rtol=1e-6
        )

        vr, sr = @jit call_logabsbinomial_right(ConcreteRNumber(-5))
        @test Reactant.to_number(sr) == -1.0
        @test isapprox(
            Reactant.to_number(vr), SpecialFunctions.logabsbinomial(-5, 3)[1]; rtol=1e-6
        )
    end
end

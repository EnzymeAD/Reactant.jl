using EnumX, Reactant, Test
using Reactant: @trace, TracedEnum, ConcreteRNumber

@enum Fruit apple = 1 banana = 2 cherry = 3
@enum Small::UInt8 low = 7 high = 200
@enumx Code Default Success MaxIters

fresh() = Reactant.to_rarray(Float32[1, 1])   # sum == 2

@testset "Enum tracing" begin
    @testset "constant results" begin
        f_const(u) = (Fruit(1), Code.Success)
        @test @jit(f_const(fresh())) == (apple, Code.Success)
    end

    @testset "ifelse" begin
        f_ifelse(u) = ifelse(sum(u) > 1, Code.Success, Code.MaxIters)
        res = @jit f_ifelse(fresh())
        @test res isa TracedEnum{Code.T}
        @test res == Code.Success
        @test Code.Success == res
        @test convert(Code.T, res) === Code.Success
        @test Code.T(res) === Code.Success
        @test Integer(res) === Int32(1)
        @test Int(res) === 1
        @test Int32(@jit(f_ifelse(Reactant.to_rarray(Float32[0, 0])))) ===
            Int32(Code.MaxIters)

        res2 = @jit f_ifelse(fresh())
        @test res2 !== res
        @test isequal(res, res2)
        @test hash(res) == hash(res2) == hash(Code.Success)
        @test length(Set([res, res2])) == 1
        @test Dict(res => 1)[res2] == 1
    end

    @testset "comparisons and conversions inside the kernel" begin
        function f_cmp(u)
            code = ifelse(sum(u) > 1, Code.Success, Code.MaxIters)
            return (
                code == Code.Success,
                Code.Success == code,
                code != Code.Success,
                code < Code.MaxIters,
                Code.MaxIters > code,
                Int(code),
                Integer(code) + Int32(1),
                Code.T(Int32(code) + Int32(1)),
            )
        end
        res = @jit f_cmp(fresh())
        @test res[1] == true
        @test res[2] == true
        @test res[3] == false
        @test res[4] == true
        @test res[5] == true
        @test res[6] == 1
        @test res[7] == Int32(2)
        @test res[8] == Code.MaxIters
    end

    @testset "@trace if" begin
        function f_two_armed(u, threshold)
            code = Code.Default
            @trace if sum(u) > threshold
                code = Code.Success
            else
                code = Code.MaxIters
            end
            return code
        end
        @test @jit(f_two_armed(fresh(), 1.0f0)) == Code.Success
        @test @jit(f_two_armed(fresh(), 3.0f0)) == Code.MaxIters

        function f_one_armed(u, threshold)
            code = Reactant.ReactantCore.promote_to_traced(Code.Default)
            @trace if sum(u) > threshold
                code = Code.Success
            end
            return code
        end
        @test @jit(f_one_armed(fresh(), 1.0f0)) == Code.Success
        @test @jit(f_one_armed(fresh(), 3.0f0)) == Code.Default
    end

    @testset "mutable struct field" begin
        mutable struct EnumCache{U,C,B}
            u::U
            code::C
            done::B
        end
        function f_field(u, threshold)
            c = EnumCache(
                u,
                Reactant.ReactantCore.promote_to_traced(Code.Default),
                Reactant.ReactantCore.promote_to_traced(false),
            )
            @trace if sum(c.u) > threshold
                c.code = Code.Success
                c.done = true
            end
            return c.code, c.done
        end
        @test @jit(f_field(fresh(), 1.0f0)) == (Code.Success, true)
        @test @jit(f_field(fresh(), 3.0f0)) == (Code.Default, false)
    end

    @testset "@trace while carrying an enum" begin
        # Loop-carried scalars must already be traced to be written back after the loop,
        # the same as for plain numbers.
        function f_while(u, threshold)
            code = Reactant.ReactantCore.promote_to_traced(Code.Default)
            i = Reactant.ReactantCore.promote_to_traced(0)
            @trace while (i < 5) & (code == Code.Default)
                u = u ./ 2
                i += 1
                code = ifelse(sum(u) < threshold, Code.Success, code)
            end
            return u, code, i
        end
        u, code, i = @jit f_while(fresh(), 0.6f0)
        @test u ≈ Float32[0.25, 0.25]
        @test code == Code.Success
        @test i == 2
        u, code, i = @jit f_while(fresh(), 0.0f0)
        @test code == Code.Default
        @test i == 5
    end

    @testset "non-default base type" begin
        f_small(u) = ifelse(sum(u) > 1, high, low)
        res = @jit f_small(fresh())
        @test res isa TracedEnum{Small}
        @test res == high
        @test Integer(res) === UInt8(200)
        f_small_int(u) = Integer(ifelse(sum(u) > 1, high, low))
        @test @jit(f_small_int(fresh())) isa ConcreteRNumber{UInt8}
        @test Small(Reactant.to_rarray(low; track_numbers=Number)) === low
    end

    @testset "enum arguments" begin
        f_arg(u, fruit) = (fruit == banana, Int(fruit))
        fruit = Reactant.to_rarray(banana; track_numbers=Number)
        @test fruit isa TracedEnum{Fruit}
        @test fruit == banana
        @test Reactant.to_rarray(banana) === banana
        res = @jit f_arg(fresh(), fruit)
        @test res[1] == true
        @test res[2] == 2
    end
end

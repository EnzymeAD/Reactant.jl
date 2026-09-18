using EnumX, Reactant, Test
using Reactant: @trace, TracedEnum, ConcreteEnum, ConcreteRNumber

@enum Fruit apple = 1 banana = 2 cherry = 3
@enum Small::UInt8 low = 7 high = 200
@enumx Code Default Success MaxIters

fresh() = Reactant.to_rarray(Float32[1, 1])

@testset "Enum tracing" begin
    @testset "constant results" begin
        f_const(u) = (Fruit(1), Code.Success)
        @test @jit(f_const(fresh())) == (apple, Code.Success)
    end

    @testset "ifelse" begin
        f_ifelse(u) = ifelse(sum(u) > 1, Code.Success, Code.MaxIters)
        res = @jit f_ifelse(fresh())
        @test res isa ConcreteEnum{Code.T}
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

        function f_one_armed(u, threshold, promote)
            code = if promote
                Reactant.ReactantCore.promote_to_traced(Code.Default)
            else
                Code.Default
            end
            @trace if sum(u) > threshold
                code = Code.Success
            end
            return code
        end
        for promote in (false, true)
            @test @jit(f_one_armed(fresh(), 1.0f0, promote)) == Code.Success
            @test @jit(f_one_armed(fresh(), 3.0f0, promote)) == Code.Default
        end

        # The other arm has a `MissingTracedValue` at this position.
        function f_branch_local(u)
            @trace if sum(u) > 1
                status = Code.Success
                u = u .* 2
            end
            return u
        end
        @test @jit(f_branch_local(fresh())) ≈ Float32[2, 2]
        @test @jit(f_branch_local(Reactant.to_rarray(Float32[0, 0]))) ≈ Float32[0, 0]
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

        function f_update(c, threshold)
            @trace if sum(c.u) > threshold
                c.code = Code.Success
                c.done = true
            end
            return c.code, c.done
        end
        for (threshold, expected) in ((1.0f0, Code.Success), (3.0f0, Code.Default))
            c = Reactant.to_rarray(
                EnumCache(Float32[1, 1], Code.Default, false);
                track_numbers=Union{Number,Base.Enum},
            )
            @test @jit(f_update(c, threshold)) == (expected, expected == Code.Success)
            @test c.code == expected
        end

        reset_cache = Reactant.to_rarray(
            EnumCache(Float32[1, 1], Code.Default, false);
            track_numbers=Union{Number,Base.Enum},
        )
        update = @compile f_update(reset_cache, 3.0f0)
        @test update(reset_cache, 3.0f0) == (Code.Default, false)
        reset_cache.code = Code.Success
        @test reset_cache.code.value isa ConcreteRNumber{Int32}
        @test update(reset_cache, 3.0f0) == (Code.Success, false)
        reset_cache.code = Code.MaxIters
        @test update(reset_cache, 3.0f0) == (Code.MaxIters, false)

        function f_field_untouched(u, threshold)
            c = EnumCache(u, Code.Default, Reactant.ReactantCore.promote_to_traced(false))
            @trace if sum(c.u) > threshold
                c.done = true
            end
            return c.code, c.done
        end
        @test @jit(f_field_untouched(fresh(), 1.0f0)) == (Code.Default, true)

        # A field declared as `ConcreteEnum{E}` cannot hold the traced representation.
        mutable struct DeclaredEnumField{U}
            u::U
            code::ConcreteEnum{Code.T}
        end
        f_declared(c) = c.code == Code.Default
        declared = DeclaredEnumField(
            fresh(), Reactant.to_rarray(Code.Default; track_numbers=Base.Enum)
        )
        @test_throws Reactant.NoFieldMatchError @jit(f_declared(declared))
    end

    @testset "@trace while carrying an enum" begin
        # Trace scalar state before the loop so updates can be written back.
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
        @test res isa ConcreteEnum{Small}
        @test res == high
        @test Integer(res) === UInt8(200)
        f_small_int(u) = Integer(ifelse(sum(u) > 1, high, low))
        @test @jit(f_small_int(fresh())) isa ConcreteRNumber{UInt8}
        @test Small(Reactant.to_rarray(low; track_numbers=Base.Enum)) === low
    end

    @testset "enum arguments" begin
        f_arg(u, fruit) = (fruit == banana, Int(fruit))
        fruit = Reactant.to_rarray(banana; track_numbers=Base.Enum)
        @test fruit isa ConcreteEnum{Fruit}
        @test fruit == banana
        @test Reactant.to_rarray(banana) === banana
        res = @jit f_arg(fresh(), fruit)
        @test res[1] == true
        @test res[2] == 2
    end

    @testset "concrete and traced representations" begin
        fruit = Reactant.to_rarray(banana; track_numbers=Base.Enum)
        @test !(fruit isa Number)
        @test convert(typeof(fruit), apple) == apple
        @test convert(ConcreteEnum{Fruit}, cherry) == cherry
        @test convert(typeof(fruit), fruit) === fruit

        function f_representation(fruit)
            @assert fruit isa TracedEnum{Fruit,Int32}
            @assert fruit.value isa Reactant.TracedRNumber{Int32}
            return fruit, fruit
        end
        first, second = @jit f_representation(fruit)
        @test first isa ConcreteEnum{Fruit}
        @test first.value isa ConcreteRNumber{Int32}
        @test first === second
        @test first == banana
        @test @jit(f_representation(first))[1] == banana

        small = Reactant.to_rarray(low; track_numbers=Base.Enum)
        converted = convert(typeof(small), high)
        @test converted.value isa ConcreteRNumber{UInt8}
        @test Small(converted) === high
    end

    @testset "aliased closure captures" begin
        first_capture = nothing
        second_capture = nothing
        function capture_status(u)
            status = ifelse(sum(u) > 0, Code.Success, Code.Default)
            first_capture = status
            second_capture = status
            return nothing
        end
        for (u, expected) in ((Float32[1], Code.Success), (Float32[-1], Code.Default))
            @jit capture_status(Reactant.to_rarray(u))
            @test first_capture isa ConcreteEnum{Code.T}
            @test first_capture == expected
            @test first_capture === second_capture
        end
    end

    @testset "track_numbers opt-in" begin
        # Enums are not `Number`s.
        @test Reactant.to_rarray(banana; track_numbers=true) === banana
        @test Reactant.to_rarray(banana; track_numbers=Number) === banana
        @test Reactant.to_rarray(banana; track_numbers=Base.Enum) isa ConcreteEnum{Fruit}
        numbers_only = Reactant.to_rarray(
            EnumCache(Float32[1], Code.Default, false); track_numbers=true
        )
        @test numbers_only.code === Code.Default
        @test numbers_only.done isa ConcreteRNumber{Bool}
        both = Reactant.to_rarray(
            EnumCache(Float32[1], Code.Default, false);
            track_numbers=Union{Number,Base.Enum},
        )
        @test both.code isa ConcreteEnum{Code.T}
        @test both.done isa ConcreteRNumber{Bool}
    end

    @testset "non-member payload" begin
        # The traced constructor does not check membership.
        f_invalid(u) =
            Code.T(Int32(ifelse(sum(u) > 1, Code.Success, Code.Default)) + Int32(5))
        r = @jit f_invalid(fresh())
        @test Integer(r) === Int32(6)
        @test r != Code.Success
        @test_throws ArgumentError Code.T(r)
        r2 = @jit f_invalid(fresh())
        @test isequal(r, r2)
        @test hash(r) == hash(r2)
        @test length(Set([r, r2, Code.Success])) == 2
        @test Dict(r => 1)[r2] == 1
    end

    @testset "scalar broadcasting" begin
        fruit = Reactant.to_rarray(banana; track_numbers=Base.Enum)
        f_broadcast(fruit) = [apple, banana, cherry] .== fruit
        @test f_broadcast(fruit) == [false, true, false]
        @test @jit(f_broadcast(fruit)) == [false, true, false]
        @test @jit(f_broadcast(Reactant.to_rarray(apple; track_numbers=Base.Enum))) ==
            [true, false, false]
    end
end

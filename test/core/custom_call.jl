using Reactant, Test
using Reactant: CustomCall

@testset "CustomCall" begin
    @testset "simple element-wise add" begin
        function my_add!(out, x, y)
            out .= x .+ y
            return nothing
        end

        function traced_add(x, y)
            return CustomCall.custom_call(my_add!, ((Float32, (4,)),), x, y)
        end

        x = Reactant.to_rarray(Float32[1.0, 2.0, 3.0, 4.0])
        y = Reactant.to_rarray(Float32[10.0, 20.0, 30.0, 40.0])
        result = @jit traced_add(x, y)
        @test Array(result) ≈ Float32[11.0, 22.0, 33.0, 44.0]
    end

    @testset "scaling" begin
        function my_scale!(out, x)
            out .= 2.0 .* x
            return nothing
        end

        function traced_scale(x)
            return CustomCall.custom_call(my_scale!, ((Float64, (3,)),), x)
        end

        x = Reactant.to_rarray([1.0, 2.0, 3.0])
        result = @jit traced_scale(x)
        @test Array(result) ≈ [2.0, 4.0, 6.0]
    end

    @testset "multiple outputs" begin
        function my_split!(out1, out2, x)
            out1 .= x[1:3]
            out2 .= x[4:6]
            return nothing
        end

        function traced_split(x)
            return CustomCall.custom_call(my_split!, ((Float32, (3,)), (Float32, (3,))), x)
        end

        x = Reactant.to_rarray(Float32[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        a, b = @jit traced_split(x)
        @test Array(a) ≈ Float32[1.0, 2.0, 3.0]
        @test Array(b) ≈ Float32[4.0, 5.0, 6.0]
    end

    @testset "2D matrix transpose" begin
        function my_transpose!(out, x)
            for i in 1:2, j in 1:3
                out[j, i] = x[i, j]
            end
            return nothing
        end

        function traced_transpose(x)
            return CustomCall.custom_call(my_transpose!, ((Float32, (3, 2)),), x)
        end

        x = Reactant.to_rarray(Float32[1 2 3; 4 5 6])
        result = @jit traced_transpose(x)
        @test Array(result) ≈ Float32[1 4; 2 5; 3 6]
    end

    @testset "no input (generator)" begin
        function my_ones!(out)
            out .= 1.0f0
            return nothing
        end

        function traced_ones()
            return CustomCall.custom_call(my_ones!, ((Float32, (4,)),))
        end

        result = @jit traced_ones()
        @test Array(result) ≈ Float32[1.0, 1.0, 1.0, 1.0]
    end

    @testset "integer types" begin
        function my_inc!(out, x)
            out .= x .+ 1
            return nothing
        end

        function traced_inc(x)
            return CustomCall.custom_call(my_inc!, ((Int64, (3,)),), x)
        end

        x = Reactant.to_rarray(Int64[10, 20, 30])
        result = @jit traced_inc(x)
        @test Array(result) == Int64[11, 21, 31]
    end

    @testset "Pair convenience API" begin
        function my_double!(out, x)
            out .= 2.0f0 .* x
            return nothing
        end

        function traced_double(x)
            return CustomCall.custom_call(my_double!, Float32 => (4,), x)
        end

        x = Reactant.to_rarray(Float32[1.0, 2.0, 3.0, 4.0])
        result = @jit traced_double(x)
        @test Array(result) ≈ Float32[2.0, 4.0, 6.0, 8.0]
    end

    @testset "has_side_effect=false" begin
        function my_pure_fn!(out, x)
            out .= x .* x
            return nothing
        end

        function traced_pure(x)
            return CustomCall.custom_call(
                my_pure_fn!, ((Float32, (2,)),), x; has_side_effect=false
            )
        end

        x = Reactant.to_rarray(Float32[3.0, 4.0])
        result = @jit traced_pure(x)
        @test Array(result) ≈ Float32[9.0, 16.0]
    end

    @testset "type mapping" begin
        for (jl_type, xla_code) in [
            (Bool, Int32(1)),
            (Int8, Int32(2)),
            (Int16, Int32(3)),
            (Int32, Int32(4)),
            (Int64, Int32(5)),
            (UInt8, Int32(6)),
            (UInt16, Int32(7)),
            (UInt32, Int32(8)),
            (UInt64, Int32(9)),
            (Float16, Int32(10)),
            (Float32, Int32(11)),
            (Float64, Int32(12)),
            (ComplexF32, Int32(15)),
            (ComplexF64, Int32(18)),
        ]
            @test CustomCall.xla_element_type(jl_type) == xla_code
            @test CustomCall.julia_element_type(xla_code) == jl_type
        end
    end
end

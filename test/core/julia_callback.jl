using Reactant, CUDA, Test

const RunningOnCPU = contains(string(Reactant.devices()[1]), "CPU")
const RunningOnCUDA = contains(string(Reactant.devices()[1]), "CUDA")

# platforms that support cfunction with closures
# (requires LLVM back-end support for trampoline intrinsics)
const cfunction_closure = Sys.ARCH === :x86_64 || Sys.ARCH === :i686

if (RunningOnCPU || RunningOnCUDA) && cfunction_closure
    @testset "simple element-wise add" begin
        function my_add!(out, x, y)
            out .= x .+ y
            return nothing
        end

        function traced_add(x, y)
            return Reactant.Ops.julia_callback(my_add!, ((eltype(x), size(x)),), x, y)
        end

        x = Reactant.TestUtils.construct_test_array(Float32, 4, 32)
        x_ra = Reactant.to_rarray(x)
        y = Reactant.TestUtils.construct_test_array(Float32, 4, 32) .+ 10.0f0
        y_ra = Reactant.to_rarray(y)
        result = @jit traced_add(x_ra, y_ra)
        @test Array(result) ≈ x .+ y
    end

    @testset "scaling" begin
        function my_scale!(out, x, alpha::Number)
            out .= alpha .* x
            return nothing
        end

        function traced_scale(x, alpha)
            return Reactant.Ops.julia_callback(my_scale!, ((eltype(x), size(x)),), x, alpha)
        end

        x = Reactant.TestUtils.construct_test_array(Float64, 3, 7)
        x_ra = Reactant.to_rarray(x)
        alpha = 3.0f0
        alpha_ra = Reactant.to_rarray(alpha; track_numbers=true)
        result = @jit traced_scale(x_ra, alpha_ra)
        @test Array(result) ≈ x .* alpha
    end

    @testset "multiple outputs" begin
        function my_split!(out1, out2, x)
            out1 .= x[1:3]
            out2 .= x[4:6]
            return nothing
        end

        function traced_split(x)
            return Reactant.Ops.julia_callback(
                my_split!, ((Float32, (3,)), (Float32, (3,))), x
            )
        end

        x = Reactant.to_rarray(Float32[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        a, b = @jit traced_split(x)
        @test Array(a) ≈ Float32[1.0, 2.0, 3.0]
        @test Array(b) ≈ Float32[4.0, 5.0, 6.0]
    end

    @testset "2D matrix transpose" begin
        function my_transpose!(out, x)
            @allowscalar for i in 1:2, j in 1:3
                out[j, i] = x[i, j]
            end
            return nothing
        end

        function traced_transpose(x)
            return Reactant.Ops.julia_callback(my_transpose!, ((Float32, (3, 2)),), x)
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
            return Reactant.Ops.julia_callback(my_ones!, ((Float32, (4,)),))
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
            return Reactant.Ops.julia_callback(my_inc!, ((Int64, (3,)),), x)
        end

        x = Reactant.to_rarray(Int64[10, 20, 30])
        result = @jit traced_inc(x)
        @test Array(result) == Int64[11, 21, 31]
    end

    @testset "has_side_effect=false" begin
        function my_pure_fn!(out, x)
            out .= x .* x
            return nothing
        end

        function traced_pure(x)
            return Reactant.Ops.julia_callback(
                my_pure_fn!, ((Float32, (2,)),), x; has_side_effect=false
            )
        end

        x = Reactant.to_rarray(Float32[3.0, 4.0])
        result = @jit traced_pure(x)
        @test Array(result) ≈ Float32[9.0, 16.0]
    end
end

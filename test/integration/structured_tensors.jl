using Reactant, Test, LinearAlgebra, FileCheck

const RunningOnTPU = contains(string(Reactant.devices()[1]), "TPU")

raise_to_syrk(x, y) = 3 .* (x * transpose(x)) .+ 5 .* y
raise_to_syrk2(x, y) = 3 .* (transpose(x) * x) .+ 5 .* y

@testset "syrk optimizations" begin
    @testset for elty in (Float32, Float64, ComplexF32, ComplexF64)
        RunningOnTPU && elty == ComplexF64 && continue

        x = Reactant.TestUtils.construct_test_array(elty, 4, 5)
        y1 = Reactant.TestUtils.construct_test_array(elty, 4, 4)
        y2 = Reactant.TestUtils.construct_test_array(elty, 5, 5)
        x_ra = Reactant.to_rarray(x)

        @testset "fn: $(fn) | y: $(size(y))" for (fn, y) in
                                                 ((raise_to_syrk, y1), (raise_to_syrk2, y2))
            y_ra = Reactant.to_rarray(y)

            hlo = @code_hlo compile_options = CompileOptions(;
                disable_structured_tensors_detection_passes=false,
                optimization_passes=:before_jit,
            ) fn(x_ra, y_ra)
            @test @filecheck begin
                @check "enzymexla.blas.syrk"
                hlo
            end

            fn_compile = @compile compile_options = CompileOptions(;
                disable_structured_tensors_detection_passes=false
            ) fn(x_ra, y_ra)

            @test fn_compile(x_ra, y_ra) ≈ fn(x, y) atol = 1e-3 rtol = 1e-3
        end
    end
end

function raise_to_symm(x, y)
    z = x * transpose(x) # Symmetric Tensors
    return 3 .* z * y
end

function raise_to_symm2(x, y)
    z = x * transpose(x) # Symmetric Tensors
    return 3 .* y * z
end

@testset "symm optimizations" begin
    @testset for elty in (Float32, Float64, ComplexF32, ComplexF64)
        RunningOnTPU && elty == ComplexF64 && continue

        x = Reactant.TestUtils.construct_test_array(elty, 4, 5)
        y1 = Reactant.TestUtils.construct_test_array(elty, 4, 7)
        y2 = Reactant.TestUtils.construct_test_array(elty, 7, 4)
        x_ra = Reactant.to_rarray(x)

        @testset "fn: $(fn) | y: $(size(y))" for (fn, y) in
                                                 ((raise_to_symm, y1), (raise_to_symm2, y2))
            y_ra = Reactant.to_rarray(y)

            hlo = @code_hlo compile_options = CompileOptions(;
                disable_structured_tensors_detection_passes=false,
                optimization_passes=:before_jit,
            ) fn(x_ra, y_ra)
            @test @filecheck begin
                @check "enzymexla.blas.symm"
                hlo
            end

            fn_compile = @compile compile_options = CompileOptions(;
                disable_structured_tensors_detection_passes=false
            ) fn(x_ra, y_ra)

            @test fn_compile(x_ra, y_ra) ≈ fn(x, y) atol = 1e-2 rtol = 1e-2
        end
    end
end

function raise_to_trmm(x, y)
    z = UpperTriangular(x)
    return z * y
end

function raise_to_trmm2(x, y)
    z = LowerTriangular(x)
    return y * z
end

@testset "trmm optimizations" begin
    @testset for elty in (Float32, Float64, ComplexF32, ComplexF64)
        RunningOnTPU && elty == ComplexF64 && continue

        x = Reactant.TestUtils.construct_test_array(elty, 100, 100)
        y1 = Reactant.TestUtils.construct_test_array(elty, 100, 50)
        y2 = Reactant.TestUtils.construct_test_array(elty, 50, 100)
        x_ra = Reactant.to_rarray(x)

        @testset "fn: $(fn) | y: $(size(y))" for (fn, y) in
                                                 ((raise_to_trmm, y1), (raise_to_trmm2, y2))
            y_ra = Reactant.to_rarray(y)

            hlo = @code_hlo compile_options = CompileOptions(;
                disable_structured_tensors_detection_passes=false,
                optimization_passes=:before_jit,
            ) fn(x_ra, y_ra)
            @test @filecheck begin
                @check "enzymexla.blas.trmm"
                hlo
            end

            fn_compile = @compile compile_options = CompileOptions(;
                disable_structured_tensors_detection_passes=false
            ) fn(x_ra, y_ra)

            @test fn_compile(x_ra, y_ra) ≈ fn(x, y) atol = 1e-2 rtol = 1e-2
        end
    end
end

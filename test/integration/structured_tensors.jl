using Reactant, Test, LinearAlgebra

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
            @test occursin("enzymexla.blas.syrk", repr(hlo))

            fn_compile = @compile compile_options = CompileOptions(;
                disable_structured_tensors_detection_passes=false
            ) fn(x_ra, y_ra)

            @test fn_compile(x_ra, y_ra) ≈ fn(x, y) atol = 1e-3 rtol = 1e-3
        end
    end
end

function raise_to_symm3(x, y)
    z = x * transpose(x) # Symmetric Tensors
    return 3 .* z * y
end

function raise_to_symm4(x, y)
    z = x * transpose(x) # Symmetric Tensors
    return 3 .* y * z
end

elty = Float32

x = Reactant.TestUtils.construct_test_array(elty, 4, 5)
y = Reactant.TestUtils.construct_test_array(elty, 4, 7)

x_ra = Reactant.to_rarray(x)
y_ra = Reactant.to_rarray(y)

@code_hlo raise_to_symm3(x_ra, y_ra)
@code_hlo compile_options = CompileOptions(;
    disable_structured_tensors_detection_passes=false, optimization_passes=:before_jit
) raise_to_symm3(x_ra, y_ra)
@code_hlo compile_options = CompileOptions(;
    disable_structured_tensors_detection_passes=false
) raise_to_symm3(x_ra, y_ra)

@jit raise_to_symm3(x_ra, y_ra)
@jit compile_options = CompileOptions(; disable_structured_tensors_detection_passes=false) raise_to_symm3(
    x_ra, y_ra
)

# @testset "syrk optimizations" begin
#     @testset for elty in (Float32, Float64, ComplexF32, ComplexF64)
#         RunningOnTPU && elty == ComplexF64 && continue

#         x = Reactant.TestUtils.construct_test_array(elty, 4, 5)
#         y1 = Reactant.TestUtils.construct_test_array(elty, 4, 4)
#         y2 = Reactant.TestUtils.construct_test_array(elty, 5, 5)
#         x_ra = Reactant.to_rarray(x)

#         @testset "fn: $(fn) | y: $(size(y))" for (fn, y) in
#                                                  ((raise_to_syrk, y1), (raise_to_syrk2, y2))
#             y_ra = Reactant.to_rarray(y)

#             hlo = @code_hlo compile_options = CompileOptions(;
#                 disable_structured_tensors_detection_passes=false,
#                 optimization_passes=:before_jit,
#             ) fn(x_ra, y_ra)
#             @test occursin("enzymexla.blas.syrk", repr(hlo))

#             fn_compile = @compile compile_options = CompileOptions(;
#                 disable_structured_tensors_detection_passes=false
#             ) fn(x_ra, y_ra)

#             @test fn_compile(x_ra, y_ra) ≈ fn(x, y) atol = 1e-3 rtol = 1e-3
#         end
#     end
# end

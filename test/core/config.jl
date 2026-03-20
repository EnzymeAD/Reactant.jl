using Reactant, Test, FileCheck

@testset "Dot General Config" begin
    x_ra = Reactant.to_rarray(ones(Float32, 4, 4))
    y_ra = Reactant.to_rarray(ones(Float32, 4, 4))

    @testset "Precision" begin
        hlo = @code_hlo *(x_ra, y_ra)
        @test @filecheck begin
            @check_not "algorithm"
            @check "precision = [DEFAULT, DEFAULT]"
            repr(hlo)
        end

        hlo = with_config(; dot_general_precision=PrecisionConfig.HIGH) do
            @code_hlo *(x_ra, y_ra)
        end
        @test @filecheck begin
            @check_not "algorithm"
            @check "precision = [HIGH, HIGH]"
            repr(hlo)
        end

        hlo = with_config(; dot_general_precision=PrecisionConfig.HIGHEST) do
            @code_hlo *(x_ra, y_ra)
        end
        @test @filecheck begin
            @check_not "algorithm"
            @check "precision = [HIGHEST, HIGHEST]"
            repr(hlo)
        end

        hlo = with_config(;
            dot_general_precision=(PrecisionConfig.HIGH, PrecisionConfig.DEFAULT)
        ) do
            @code_hlo *(x_ra, y_ra)
        end
        @test @filecheck begin
            @check_not "algorithm"
            @check "precision = [HIGH, DEFAULT]"
            repr(hlo)
        end
    end

    @testset "Algorithm" begin
        @test_throws ErrorException with_config(;
            dot_general_algorithm=DotGeneralAlgorithmPreset.ANY_F8_ANY_F8_F32
        ) do
            @code_hlo *(x_ra, y_ra)
        end

        hlo = with_config(; dot_general_algorithm=DotGeneralAlgorithmPreset.F32_F32_F32) do
            @code_hlo *(x_ra, y_ra)
        end
        @test @filecheck begin
            @check "algorithm ="
            repr(hlo)
        end

        @test_throws AssertionError with_config(;
            dot_general_algorithm=DotGeneralAlgorithmPreset.F64_F64_F64
        ) do
            @code_hlo *(x_ra, y_ra)
        end

        x_ra = Reactant.to_rarray(ones(Float16, 4, 4))
        y_ra = Reactant.to_rarray(ones(Float16, 4, 4))

        hlo = with_config(;
            dot_general_algorithm=DotGeneralAlgorithm(
                Float16, Float16, Float32, 1, 1, 1, false
            ),
        ) do
            @code_hlo *(x_ra, y_ra)
        end

        @test @filecheck begin
            @check "algorithm = <lhs_precision_type = f16, rhs_precision_type = f16, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>"
            repr(hlo)
        end
    end
end

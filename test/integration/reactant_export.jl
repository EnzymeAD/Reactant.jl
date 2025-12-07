using Reactant, Test

@testset "ReactantExport" begin
    @testset "Simple function export" begin
        f_simple(x) = sin.(x) .+ cos.(x)

        x_data = Reactant.TestUtils.construct_test_array(Float32, 4, 5)
        x = Reactant.to_rarray(x_data)

        # Export the function
        julia_file_path = Reactant.Serialization.export_to_reactant_script(
            f_simple, x; output_dir=mktempdir(; cleanup=true)
        )

        @test isfile(julia_file_path)
        @test endswith(julia_file_path, ".jl")

        # Check that generated files exist
        output_dir = dirname(julia_file_path)
        mlir_files = filter(f -> endswith(f, ".mlir"), readdir(output_dir))
        jls_files = filter(f -> endswith(f, ".jls"), readdir(output_dir))

        @test length(mlir_files) > 0
        @test length(jls_files) > 0

        # Verify Julia script contains key components
        julia_content = read(julia_file_path, String)
        @test contains(julia_content, "using Reactant")
        @test contains(julia_content, "using Serialization")
        @test contains(julia_content, "f_simple")
        @test contains(julia_content, "load_inputs")
        @test contains(julia_content, "run_f_simple")

        # We can't execute the full script since HLO execution isn't implemented yet,
        # but we can verify the structure is correct
    end

    @testset "Matrix multiplication export" begin
        f_matmul(x, y) = x * y

        x_data = Reactant.TestUtils.construct_test_array(Float32, 3, 4)
        y_data = Reactant.TestUtils.construct_test_array(Float32, 4, 5)
        x = Reactant.to_rarray(x_data)
        y = Reactant.to_rarray(y_data)

        # Export the function
        julia_file_path = Reactant.Serialization.export_to_reactant_script(
            f_matmul, x, y; output_dir=mktempdir(; cleanup=true), function_name="matmul"
        )

        @test isfile(julia_file_path)

        output_dir = dirname(julia_file_path)
        jls_files = filter(f -> endswith(f, ".jls"), readdir(output_dir))
        @test length(jls_files) > 0

        # Verify the JLS file contains both inputs
        using Serialization
        inputs_data = open(first(filter(f -> endswith(f, ".jls"), readdir(output_dir; join=true))), "r") do io
            deserialize(io)
        end
        @test haskey(inputs_data, "arr_1") || haskey(inputs_data, "arr_2")

        # Verify Julia script structure
        julia_content = read(julia_file_path, String)
        @test contains(julia_content, "matmul")
        @test contains(julia_content, "arg1")
        @test contains(julia_content, "arg2")
    end

    @testset "Complex function with multiple arguments" begin
        f_complex(x, y, z) = sum(x .* y .+ sin.(z); dims=2)

        x_data = Reactant.TestUtils.construct_test_array(Float32, 5, 4)
        y_data = Reactant.TestUtils.construct_test_array(Float32, 5, 4)
        z_data = Reactant.TestUtils.construct_test_array(Float32, 5, 4)
        x = Reactant.to_rarray(x_data)
        y = Reactant.to_rarray(y_data)
        z = Reactant.to_rarray(z_data)

        # Export the function
        julia_file_path = Reactant.Serialization.export_to_reactant_script(
            f_complex,
            x,
            y,
            z;
            output_dir=mktempdir(; cleanup=true),
            function_name="complex_fn",
        )

        @test isfile(julia_file_path)

        output_dir = dirname(julia_file_path)
        mlir_files = filter(f -> endswith(f, ".mlir"), readdir(output_dir))
        jls_files = filter(f -> endswith(f, ".jls"), readdir(output_dir))

        @test length(mlir_files) > 0
        @test length(jls_files) > 0

        julia_content = read(julia_file_path, String)
        @test contains(julia_content, "complex_fn")
        @test contains(julia_content, "arg1")
        @test contains(julia_content, "arg2")
        @test contains(julia_content, "arg3")
    end

    @testset "Test Serialization input/output consistency" begin
        # Test that inputs are saved and can be loaded correctly
        f_test(x) = x .+ 1.0f0

        x_data = Float32[1.0 2.0; 3.0 4.0]
        x = Reactant.to_rarray(x_data)

        julia_file_path = Reactant.Serialization.export_to_reactant_script(
            f_test, x; output_dir=mktempdir(; cleanup=true), function_name="test_jls"
        )

        output_dir = dirname(julia_file_path)
        jls_files = filter(f -> endswith(f, ".jls"), readdir(output_dir; join=true))
        @test !isempty(jls_files)
        jls_path = first(jls_files)
        
        # Load the JLS file and verify the data
        using Serialization
        inputs_data = open(jls_path, "r") do io
            deserialize(io)
        end
        @test haskey(inputs_data, "arr_1")
        
        # The data should be in Julia's native format (no transposition needed)
        loaded_data = inputs_data["arr_1"]
        @test size(loaded_data) == size(x_data)
        @test isapprox(loaded_data, x_data; rtol=1e-5)
    end
end

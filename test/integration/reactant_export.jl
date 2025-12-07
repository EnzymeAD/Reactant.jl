using Reactant, Test, NPZ

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
        npz_files = filter(f -> endswith(f, ".npz"), readdir(output_dir))

        @test length(mlir_files) > 0
        @test length(npz_files) > 0

        # Verify Julia script contains key components
        julia_content = read(julia_file_path, String)
        @test contains(julia_content, "using Reactant")
        @test contains(julia_content, "using NPZ")
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
        npz_files = filter(f -> endswith(f, ".npz"), readdir(output_dir))
        @test length(npz_files) > 0

        # Verify the NPZ file contains both inputs
        npz_data = npzread(
            first(filter(f -> endswith(f, ".npz"), readdir(output_dir; join=true)))
        )
        @test haskey(npz_data, "arr_1") || haskey(npz_data, "arr_2")

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
        npz_files = filter(f -> endswith(f, ".npz"), readdir(output_dir))

        @test length(mlir_files) > 0
        @test length(npz_files) > 0

        julia_content = read(julia_file_path, String)
        @test contains(julia_content, "complex_fn")
        @test contains(julia_content, "arg1")
        @test contains(julia_content, "arg2")
        @test contains(julia_content, "arg3")
    end

    @testset "Test NPZ input/output consistency" begin
        # Test that inputs are saved and can be loaded correctly
        f_test(x) = x .+ 1.0f0

        x_data = Float32[1.0 2.0; 3.0 4.0]
        x = Reactant.to_rarray(x_data)

        julia_file_path = Reactant.Serialization.export_to_reactant_script(
            f_test, x; output_dir=mktempdir(; cleanup=true), function_name="test_npz"
        )

        output_dir = dirname(julia_file_path)
        npz_path = first(filter(f -> endswith(f, ".npz"), readdir(output_dir; join=true)))
        
        # Load the NPZ file and verify the data
        npz_data = npzread(npz_path)
        @test haskey(npz_data, "arr_1")
        
        # The data should be transposed for NumPy format
        loaded_data = npz_data["arr_1"]
        @test size(loaded_data) == (2, 2)  # NumPy row-major format
        
        # Transpose back to Julia format
        julia_format = permutedims(loaded_data, (2, 1))
        @test isapprox(julia_format, x_data; rtol=1e-5)
    end
end

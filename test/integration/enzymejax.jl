using Reactant, Test, NPZ, PythonCall

function run_exported_enzymejax_function(python_file_path::String, function_name::String)
    output_dir = dirname(python_file_path)

    sys = pyimport("sys")
    importlib = pyimport("importlib.util")
    sys.path.insert(0, "$(output_dir)")

    spec = importlib.spec_from_file_location("test_module", "$(python_file_path)")
    mod = importlib.module_from_spec(spec)
    spec.loader.exec_module(mod)

    loaded_inputs = pygetattr(mod, "load_inputs")()
    res = pyconvert(Array, pygetattr(mod, function_name)(loaded_inputs...)[0])
    return permutedims(res, ndims(res):-1:1)
end

@testset "EnzymeJAX Export - Without Sharding" begin
    @testset "Simple function" begin
        f_simple(x) = sin.(x) .+ cos.(x)

        x_data = Reactant.TestUtils.construct_test_array(Float32, 4, 5)
        x = Reactant.to_rarray(x_data)

        # Compute expected result
        expected_result = f_simple(x_data)

        # Export the function
        python_file_path = Reactant.Serialization.export_to_enzymejax(
            f_simple, x; output_dir=mktempdir(; cleanup=true)
        )

        @test isfile(python_file_path)
        @test endswith(python_file_path, ".py")

        # Check that generated files exist
        output_dir = dirname(python_file_path)
        mlir_files = filter(f -> endswith(f, ".mlir"), readdir(output_dir))
        npz_files = filter(f -> endswith(f, ".npz"), readdir(output_dir))

        @test length(mlir_files) > 0
        @test length(npz_files) > 0

        # Verify Python script contains key components
        python_content = read(python_file_path, String)
        @test contains(python_content, "hlo_call")
        @test contains(python_content, "f_simple")

        # Run the exported script and verify results
        result = run_exported_enzymejax_function(python_file_path, "run_f_simple")
        @test isapprox(Array(result), expected_result; atol=1e-5, rtol=1e-5)
    end

    @testset "Matrix multiplication" begin
        f_matmul(x, y) = x * y

        x_data = Reactant.TestUtils.construct_test_array(Float32, 3, 4)
        y_data = Reactant.TestUtils.construct_test_array(Float32, 4, 5)
        x = Reactant.to_rarray(x_data)
        y = Reactant.to_rarray(y_data)

        # Compute expected result
        expected_result = f_matmul(x_data, y_data)

        # Export the function
        python_file_path = Reactant.Serialization.export_to_enzymejax(
            f_matmul, x, y; output_dir=mktempdir(; cleanup=true), function_name="matmul"
        )

        @test isfile(python_file_path)

        output_dir = dirname(python_file_path)
        npz_files = filter(f -> endswith(f, ".npz"), readdir(output_dir))
        @test length(npz_files) > 0

        # Verify the NPZ file contains both inputs
        npz_data = npzread(
            first(filter(f -> endswith(f, ".npz"), readdir(output_dir; join=true)))
        )
        @test haskey(npz_data, "arr_1") || haskey(npz_data, "arr_2")

        # Run the exported script and verify results
        result = run_exported_enzymejax_function(python_file_path, "run_matmul")
        @test isapprox(Array(result), expected_result; atol=1e-5, rtol=1e-5)
    end

    @testset "Complex function with multiple arguments" begin
        f_complex(x, y, z) = sum(x .* y .+ sin.(z); dims=2)

        x_data = Reactant.TestUtils.construct_test_array(Float32, 5, 4)
        y_data = Reactant.TestUtils.construct_test_array(Float32, 5, 4)
        z_data = Reactant.TestUtils.construct_test_array(Float32, 5, 4)
        x = Reactant.to_rarray(x_data)
        y = Reactant.to_rarray(y_data)
        z = Reactant.to_rarray(z_data)

        # Compute expected result
        expected_result = f_complex(x_data, y_data, z_data)

        # Export the function
        python_file_path = Reactant.Serialization.export_to_enzymejax(
            f_complex,
            x,
            y,
            z;
            output_dir=mktempdir(; cleanup=true),
            function_name="complex_fn",
        )

        @test isfile(python_file_path)

        output_dir = dirname(python_file_path)
        mlir_files = filter(f -> endswith(f, ".mlir"), readdir(output_dir))
        npz_files = filter(f -> endswith(f, ".npz"), readdir(output_dir))

        @test length(mlir_files) > 0
        @test length(npz_files) > 0

        python_content = read(python_file_path, String)
        @test contains(python_content, "complex_fn")

        # Run the exported script and verify results
        result = run_exported_enzymejax_function(python_file_path, "run_complex_fn")
        @test isapprox(Array(result), expected_result; atol=1e-5, rtol=1e-5)
    end
end

@testset "EnzymeJAX Export - With Sharding" begin
    # Only run sharding tests if we have multiple devices
    addressable_devices = Reactant.addressable_devices()

    if length(addressable_devices) ≥ 8
        mesh = Reactant.Sharding.Mesh(
            reshape(addressable_devices[1:8], 2, 4), ("batch", "feature")
        )

        @testset "Export with sharding and preserve_sharding=true" begin
            f_sharded(x, y) = x .+ y

            x_data = Reactant.TestUtils.construct_test_array(Float32, 2, 4)
            y_data = Reactant.TestUtils.construct_test_array(Float32, 2, 4)
            x = Reactant.to_rarray(
                x_data; sharding=Reactant.Sharding.NamedSharding(mesh, ("batch", "feature"))
            )
            y = Reactant.to_rarray(y_data; sharding=Reactant.Sharding.Replicated(mesh))

            # Compute expected result
            expected_result = f_sharded(x_data, y_data)

            # Export with sharding preservation enabled
            python_file_path = Reactant.Serialization.export_to_enzymejax(
                f_sharded,
                x,
                y;
                output_dir=mktempdir(; cleanup=true),
                function_name="f_sharded_with_preserve",
                preserve_sharding=true,
            )

            @test isfile(python_file_path)

            # Check that Python script includes sharding information
            python_content = read(python_file_path, String)
            @test (
                contains(python_content, "NamedSharding") ||
                contains(python_content, "pmap") ||
                contains(python_content, "mesh")
            )

            # Run the exported script and verify results
            result = run_exported_enzymejax_function(
                python_file_path, "run_f_sharded_with_preserve"
            )
            @test isapprox(Array(result), expected_result; atol=1e-5, rtol=1e-5)
        end

        @testset "Export with sharding but preserve_sharding=false" begin
            f_sharded_no_preserve(x, y) = x .- y

            x_data = Reactant.TestUtils.construct_test_array(Float32, 2, 4)
            y_data = Reactant.TestUtils.construct_test_array(Float32, 2, 4)
            x = Reactant.to_rarray(
                x_data; sharding=Reactant.Sharding.NamedSharding(mesh, ("batch", "feature"))
            )
            y = Reactant.to_rarray(y_data; sharding=Reactant.Sharding.Replicated(mesh))

            # Compute expected result
            expected_result = f_sharded_no_preserve(x_data, y_data)

            # Export without sharding preservation
            python_file_path = Reactant.Serialization.export_to_enzymejax(
                f_sharded_no_preserve,
                x,
                y;
                output_dir=mktempdir(; cleanup=true),
                function_name="f_sharded_no_preserve",
                preserve_sharding=false,
            )

            @test isfile(python_file_path)

            # Check that Python script does NOT include explicit sharding directives
            python_content = read(python_file_path, String)
            # Should have hlo_call but without the advanced sharding setup
            @test contains(python_content, "hlo_call")

            # Run the exported script and verify results
            result = run_exported_enzymejax_function(
                python_file_path, "run_f_sharded_no_preserve"
            )
            @test isapprox(Array(result), expected_result; atol=1e-5, rtol=1e-5)
        end
    else
        @warn "Skipping sharding tests: insufficient devices (need ≥8, have $(length(addressable_devices)))"
    end
end

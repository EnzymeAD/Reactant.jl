using Reactant, Test, NPZ, PythonCall, FileCheck

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
        @test @filecheck begin
            @check_dag "hlo_call"
            @check_dag "f_simple"
            python_content
        end

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

        # Export function with no data
        python_file_path = Reactant.Serialization.export_to_enzymejax(
            f_matmul,
            x,
            y;
            output_dir=mktempdir(; cleanup=true),
            function_name="matmul",
            export_data_as_npz=false,
        )

        @test isfile(python_file_path)

        python_content = read(python_file_path, String)
        @test @filecheck begin
            @check "prng_key = jax.random.PRNGKey(0)"
            python_content
        end
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
        @test @filecheck begin
            @check "complex_fn"
            python_content
        end

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
            @test !contains(python_content, "SHARDING_METADATA")
            # Should have hlo_call but without the advanced sharding setup
            @test @filecheck begin
                @check "hlo_call"
                python_content
            end

            # Run the exported script and verify results
            result = run_exported_enzymejax_function(
                python_file_path, "run_f_sharded_no_preserve"
            )
            @test isapprox(Array(result), expected_result; atol=1e-5, rtol=1e-5)
        end

        @testset "Export sharding attributes (is_closed, priority, subaxes)" begin
            x_data = Reactant.TestUtils.construct_test_array(Float32, 4, 8)
            y_data = Reactant.TestUtils.construct_test_array(Float32, 4, 8)
            z_data = Reactant.TestUtils.construct_test_array(Float32, 8, 4)

            sharding_x = Reactant.Sharding.NamedSharding(
                mesh, ("batch", "feature"); is_closed=(false, true), priority=(2, 5)
            )
            sharding_y = Reactant.Sharding.NamedSharding(
                mesh,
                (nothing, "feature");
                subaxes=[[nothing], [(2, 2)]],
                is_closed=(true, false),
                priority=(-1, 3),
            )
            sharding_z = Reactant.Sharding.NamedSharding(
                mesh, (("batch", "feature"), nothing); is_closed=(false, false)
            )

            x = Reactant.to_rarray(x_data; sharding=sharding_x)
            y = Reactant.to_rarray(y_data; sharding=sharding_y)
            z = Reactant.to_rarray(z_data; sharding=sharding_z)

            info_x = Reactant.Serialization.EnzymeJAX._extract_sharding_info(x)
            @test info_x.type == "NamedSharding"
            @test info_x.is_closed == (false, true)
            @test info_x.priority == (2, 5)
            @test info_x.subaxes == [[nothing], [nothing]]

            info_y = Reactant.Serialization.EnzymeJAX._extract_sharding_info(y)
            @test info_y.type == "NamedSharding"
            @test info_y.is_closed == (true, false)
            @test info_y.priority == (-1, 3)
            @test info_y.subaxes == [[nothing], [(2, 2)]]

            info_z = Reactant.Serialization.EnzymeJAX._extract_sharding_info(z)
            @test info_z.is_closed == (false, false)
            @test info_z.priority == (-1, -1)

            expected_xy = x_data + y_data
            python_file_path = Reactant.Serialization.export_to_enzymejax(
                +,
                x,
                y;
                output_dir=mktempdir(; cleanup=true),
                function_name="f_sharded_attrs",
                preserve_sharding=true,
            )
            @test isfile(python_file_path)

            python_content = read(python_file_path, String)
            @test contains(python_content, "SHARDING_METADATA")
            @test contains(python_content, "\"is_closed\": (True, False)")
            @test contains(python_content, "\"priority\": (5, 2)")
            @test contains(python_content, "\"is_closed\": (False, True)")
            @test contains(python_content, "\"priority\": (3, -1)")
            @test contains(python_content, "\"subaxes\": (((2, 2),), (None,))")
            @test contains(
                python_content,
                "#sdy.sharding<@mesh, [{\\\"feature\\\"}p5, {\\\"batch\\\", ?}p2]>",
            )
            @test contains(
                python_content, "#sdy.sharding<@mesh, [{\\\"feature\\\":(2)2, ?}p3, {}]>"
            )
            @test contains(
                python_content,
                "jax.make_mesh((2, 2, 2), ('batch', 'feature_sub_0', 'feature_sub_1'), (jax.sharding.AxisType.Auto,) * 3)",
            )
            @test contains(
                python_content,
                "jax.sharding.NamedSharding(mesh0, P(('feature_sub_0', 'feature_sub_1'), 'batch'))",
            )
            @test contains(
                python_content,
                "jax.sharding.NamedSharding(mesh0, P('feature_sub_1', None))",
            )

            result_xy = run_exported_enzymejax_function(
                python_file_path, "run_f_sharded_attrs"
            )
            @test isapprox(Array(result_xy), expected_xy; atol=1e-5, rtol=1e-5)

            expected_z = -z_data
            python_file_path_z = Reactant.Serialization.export_to_enzymejax(
                -,
                z;
                output_dir=mktempdir(; cleanup=true),
                function_name="f_sharded_multiaxis",
                preserve_sharding=true,
            )
            python_content_z = read(python_file_path_z, String)
            @test contains(
                python_content_z,
                "jax.sharding.NamedSharding(mesh0, P(None, ('batch', 'feature')))",
            )
            @test contains(python_content_z, "\"is_closed\": (False, False)")
            result_z = run_exported_enzymejax_function(
                python_file_path_z, "run_f_sharded_multiaxis"
            )
            @test isapprox(Array(result_z), expected_z; atol=1e-5, rtol=1e-5)
        end
    else
        @warn "Skipping sharding tests: insufficient devices (need ≥8, have $(length(addressable_devices)))"
    end
end

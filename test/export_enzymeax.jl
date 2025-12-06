using Reactant
using Test

@testset "Export to EnzymeJAX" begin
    # Create a temporary directory for the export
    tmpdir = mktempdir()

    try
        # Define a simple function
        function simple_add(x, y)
            return x .+ y
        end

        # Create some example inputs
        x = Reactant.to_rarray(Float32[1, 2, 3])
        y = Reactant.to_rarray(Float32[4, 5, 6])

        # Export to EnzymeJAX
        mlir_path, python_path, input_paths = Reactant.Serialization.export_to_enzymejax(
            simple_add, x, y; output_dir=tmpdir, function_name="simple_add"
        )

        # Verify that all files were created
        @test isfile(mlir_path)
        @test isfile(python_path)
        @test length(input_paths) == 2
        @test all(isfile, input_paths)

        # Verify MLIR file has content
        mlir_content = read(mlir_path, String)
        @test !isempty(mlir_content)
        @test occursin("module", mlir_content)

        # Verify Python file has content and correct structure
        python_content = read(python_path, String)
        @test !isempty(python_content)
        @test occursin("from enzyme_ad.jax import hlo_call", python_content)
        @test occursin("def run_simple_add", python_content)
        @test occursin("def load_inputs", python_content)
        @test occursin("if __name__ == \"__main__\":", python_content)

        # Verify input files exist and have reasonable sizes
        for input_path in input_paths
            @test filesize(input_path) > 0
        end

        println("✓ All export_to_enzymejax tests passed!")
        println("  - MLIR file created: $(mlir_path)")
        println("  - Python file created: $(python_path)")
        println("  - Input files created: $(length(input_paths))")
    finally
        # Clean up
        rm(tmpdir; recursive=true, force=true)
    end
end

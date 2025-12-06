using Reactant
using Test

@testset "Export to EnzymeJAX - Multi-dimensional Arrays" begin
    tmpdir = mktempdir()
    
    try
        # Define a function with 2D arrays
        function matrix_multiply(x, y)
            return x * y
        end
        
        # Create 2D arrays - Julia uses column-major order
        x = Reactant.to_rarray(Float32[1 2 3; 4 5 6])  # 2x3 matrix
        y = Reactant.to_rarray(Float32[7 8; 9 10; 11 12])  # 3x2 matrix
        
        # Export to EnzymeJAX
        mlir_path, python_path, input_paths = Reactant.Serialization.export_to_enzymeax(
            matrix_multiply, x, y;
            output_dir=tmpdir,
            function_name="matrix_multiply"
        )
        
        @test isfile(mlir_path)
        @test isfile(python_path)
        @test length(input_paths) == 2
        
        # Read Python file and check for correct shape information
        python_content = read(python_path, String)
        
        # The shapes should be transposed for Python (row-major)
        # Julia x: (2, 3) -> Python: (3, 2)
        # Julia y: (3, 2) -> Python: (2, 3)
        @test occursin("(3, 2)", python_content)  # Transposed shape of x
        @test occursin("(2, 3)", python_content)  # Transposed shape of y
        
        println("✓ Multi-dimensional array export test passed!")
        
    finally
        rm(tmpdir; recursive=true, force=true)
    end
end

@testset "Export to EnzymeJAX - 3D Arrays" begin
    tmpdir = mktempdir()
    
    try
        # Define a function with 3D arrays (like image data)
        function add_3d(x, y)
            return x .+ y
        end
        
        # Create 3D arrays - e.g., (height, width, channels, batch)
        # Julia: (28, 28, 1, 4) -> Python: (4, 1, 28, 28)
        x = Reactant.to_rarray(rand(Float32, 28, 28, 1, 4))
        y = Reactant.to_rarray(rand(Float32, 28, 28, 1, 4))
        
        # Export to EnzymeJAX
        mlir_path, python_path, input_paths = Reactant.Serialization.export_to_enzymeax(
            add_3d, x, y;
            output_dir=tmpdir,
            function_name="add_3d"
        )
        
        @test isfile(mlir_path)
        @test isfile(python_path)
        
        # Check that Python file mentions the transposed shape
        python_content = read(python_path, String)
        @test occursin("(4, 1, 28, 28)", python_content)
        
        println("✓ 3D array export test passed!")
        
    finally
        rm(tmpdir; recursive=true, force=true)
    end
end

@testset "Export to EnzymeJAX - File Content Verification" begin
    tmpdir = mktempdir()
    
    try
        function simple_fn(x)
            return x .* 2.0f0
        end
        
        x = Reactant.to_rarray(Float32[1.0, 2.0, 3.0, 4.0])
        
        mlir_path, python_path, input_paths = Reactant.Serialization.export_to_enzymeax(
            simple_fn, x;
            output_dir=tmpdir,
            function_name="test_fn"
        )
        
        # Verify MLIR contains necessary elements
        mlir_content = read(mlir_path, String)
        @test occursin("module", mlir_content)
        
        # Verify Python file structure
        python_content = read(python_path, String)
        @test occursin("import jax", python_content)
        @test occursin("import numpy as np", python_content)
        @test occursin("from enzyme_ad.jax import hlo_call", python_content)
        @test occursin("def run_test_fn(arg1)", python_content)
        @test occursin("source=_hlo_code", python_content)
        @test occursin("jax.jit(run_test_fn)", python_content)
        
        println("✓ File content verification test passed!")
        
    finally
        rm(tmpdir; recursive=true, force=true)
    end
end

println("\n✅ All comprehensive tests passed!")

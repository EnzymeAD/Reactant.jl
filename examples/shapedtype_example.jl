# Example: Using ShapeDtypeStruct for compilation without concrete arrays
#
# This example demonstrates how to use Reactant.ShapeDtypeStruct to compile
# functions without having to construct full ConcreteRArray instances with
# actual data. This is useful for:
# 1. Faster compilation when you only need shape/dtype information
# 2. Memory efficiency when working with large arrays
# 3. Similar workflow to JAX's ShapeDtypeStruct

using Reactant

# Example 1: Basic usage with a simple function
println("Example 1: Basic compilation with ShapeDtypeStruct")
println("=" ^ 60)

# Define a simple function that sums an array
f_sum(x) = sum(x)

# Instead of creating a full ConcreteRArray:
# x = Reactant.ConcreteRArray(rand(Float32, 10, 20))  # This allocates memory!

# Use ShapeDtypeStruct to specify only shape and dtype:
spec = Reactant.ShapeDtypeStruct((10, 20), Float32)
println("Created ShapeDtypeStruct: ", spec)
println("  Shape: ", size(spec))
println("  Element type: ", eltype(spec))
println("  Dimensions: ", ndims(spec))

# Compile the function using the spec
compiled_f_sum = Reactant.compile(f_sum, (spec,))
println("✓ Function compiled successfully")

# Now execute with actual data
x_actual = Reactant.ConcreteRArray(rand(Float32, 10, 20))
result = compiled_f_sum(x_actual)
println("Result: ", result, " (type: ", typeof(result), ")")
println()

# Example 2: Multiple arguments
println("Example 2: Multiple arguments with ShapeDtypeStruct")
println("=" ^ 60)

f_add(x, y) = x .+ y

spec1 = Reactant.ShapeDtypeStruct((5, 5), Float64)
spec2 = Reactant.ShapeDtypeStruct((5, 5), Float64)

compiled_f_add = Reactant.compile(f_add, (spec1, spec2))
println("✓ Function with 2 arguments compiled")

x_data = Reactant.ConcreteRArray(rand(Float64, 5, 5))
y_data = Reactant.ConcreteRArray(rand(Float64, 5, 5))
result_add = compiled_f_add(x_data, y_data)
println("Result shape: ", size(result_add))
println()

# Example 3: Different dtypes
println("Example 3: Compilation with different dtypes")
println("=" ^ 60)

f_sin(x) = sin.(x)

for dtype in [Float32, Float64]
    spec = Reactant.ShapeDtypeStruct((100,), dtype)
    compiled = Reactant.compile(f_sin, (spec,))
    
    x = Reactant.ConcreteRArray(rand(dtype, 100))
    result = compiled(x)
    println("✓ Compiled and ran for dtype: ", dtype)
end
println()

# Example 4: Benefits demonstration
println("Example 4: Memory efficiency")
println("=" ^ 60)

# For very large arrays, you can compile without allocating the full array:
large_spec = Reactant.ShapeDtypeStruct((10000, 10000), Float32)
println("Created spec for large array: ", size(large_spec))
println("  This doesn't allocate ", prod(size(large_spec)) * sizeof(Float32) / 1e9, " GB of memory!")

# Compile a function for this large array
f_large(x) = sum(x .* x)
compiled_large = Reactant.compile(f_large, (large_spec,))
println("✓ Compiled function for large array without allocating memory")
println()

println("All examples completed successfully!")
println()
println("Key Takeaways:")
println("1. ShapeDtypeStruct allows compilation without data allocation")
println("2. Same compiled function can be used with actual ConcreteRArray data")
println("3. Useful for large arrays and rapid prototyping")
println("4. Similar API to JAX's ShapeDtypeStruct")

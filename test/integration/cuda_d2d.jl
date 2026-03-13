using Reactant, Test, CUDA
using Reactant: ConcretePJRTArray, Sharding, XLA, MLIR

if !CUDA.functional()
    @info "CUDA not functional — skipping D2D CuArray interop tests"
else # CUDA.functional()

# ============================================================
# 1. Basic correctness: CuArray -> ConcreteRArray
# ============================================================
@testset "CuArray -> ConcreteRArray" begin
    @testset "1D arrays" begin
        for n in [1, 10, 256, 10_000, 1_000_000]
            cu = CuArray(collect(Float32, 1:n))
            ra = ConcretePJRTArray(cu)
            @test Array(ra) == Array(cu)
        end
    end

    @testset "Multi-dimensional" begin
        cu = CuArray(reshape(collect(Float32, 1:600), 20, 30))
        ra = ConcretePJRTArray(cu)
        @test size(ra) == (20, 30)
        @test Array(ra) == Array(cu)

        cu3 = CuArray(reshape(collect(Float32, 1:2400), 10, 20, 12))
        ra3 = ConcretePJRTArray(cu3)
        @test size(ra3) == (10, 20, 12)
        @test Array(ra3) == Array(cu3)
    end

    @testset "Element types" begin
        for T in [Float32, Float64, Int32, Int64]
            cu = CuArray(T[1, 2, 3, 4, 5])
            ra = ConcretePJRTArray(cu)
            @test Array(ra) == Array(cu)
            @test eltype(ra) == T
        end
    end

    @testset "Random data (non-trivial bit patterns)" begin
        cu = CUDA.rand(Float32, 403, 7625)
        CUDA.synchronize()
        ra = ConcretePJRTArray(cu)
        @test Array(ra) == Array(cu)
    end
end

# ============================================================
# 2. Basic correctness: ConcreteRArray -> CuArray
# ============================================================
@testset "ConcreteRArray -> CuArray" begin
    @testset "1D arrays" begin
        for n in [1, 10, 256, 10_000, 1_000_000]
            data = collect(Float32, 1:n)
            ra = Reactant.to_rarray(data)
            cu = CuArray(ra)
            @test Array(cu) == data
        end
    end

    @testset "Multi-dimensional" begin
        data = reshape(collect(Float32, 1:600), 20, 30)
        ra = Reactant.to_rarray(data)
        cu = CuArray(ra)
        @test size(cu) == (20, 30)
        @test Array(cu) == data
    end

    @testset "Element types" begin
        for T in [Float32, Float64, Int32, Int64]
            data = T[1, 2, 3, 4, 5]
            ra = Reactant.to_rarray(data)
            cu = CuArray(ra)
            @test Array(cu) == data
            @test eltype(cu) == T
        end
    end
end

# ============================================================
# 3. Round-trip correctness
# ============================================================
@testset "Round-trip CuArray -> RArray -> CuArray" begin
    cu_orig = CUDA.rand(Float32, 403, 7625)
    CUDA.synchronize()
    orig_data = Array(cu_orig)
    ra = ConcretePJRTArray(cu_orig)
    cu_back = CuArray(ra)
    @test Array(cu_back) == orig_data
end

@testset "Round-trip RArray -> CuArray -> RArray" begin
    data = Float32.(randn(100, 200))
    ra_orig = Reactant.to_rarray(data)
    cu = CuArray(ra_orig)
    ra_back = ConcretePJRTArray(cu)
    @test Array(ra_back) == data
end

# ============================================================
# 4. Stress tests: forward direction (CuArray -> ConcreteRArray)
#
#    Strategy: launch async CUDA.jl operations that write to a
#    CuArray, then immediately convert to ConcreteRArray without
#    an explicit CUDA.synchronize() at the call site.
#    ConcretePJRTArray internally synchronizes the CUDA stream
#    before the D2D copy, so these tests verify that the internal
#    synchronization correctly captures all pending writes.
# ============================================================
@testset "Forward: convert immediately after async CUDA write" begin
    n = 100_000
    for trial in 1:50
        cu = CUDA.zeros(Float32, n)
        # Async kernel that fills the array (runs on CUDA.jl's stream)
        cu .= Float32(trial)
        # No explicit sync here -- ConcretePJRTArray handles it internally
        ra = ConcretePJRTArray(cu)
        result = Array(ra)
        @test all(result .== Float32(trial))
    end
end

@testset "Forward: convert after async CUDA broadcast chain" begin
    n = 50_000
    for trial in 1:50
        a = CUDA.rand(Float32, n)
        b = CUDA.rand(Float32, n)
        # Multi-step async computation on CUDA.jl's stream
        c = a .* b .+ Float32(trial)
        expected = Array(a) .* Array(b) .+ Float32(trial)
        # No explicit sync here -- ConcretePJRTArray handles it internally
        ra = ConcretePJRTArray(c)
        @test Array(ra) ≈ expected
    end
end

# ============================================================
# 5. Race condition stress tests: reverse direction
#    (ConcreteRArray -> CuArray)
#
#    Strategy: produce ConcreteRArrays via @jit compiled functions
#    (which run on XLA's internal streams), then immediately convert
#    to CuArray. AwaitBufferReady must ensure the XLA computation
#    completes before the D2D copy.
# ============================================================
add_arrays(x, y) = x .+ y

@testset "Reverse: convert output of @jit immediately" begin
    for trial in 1:50
        a_data = Float32.(randn(1000))
        b_data = Float32.(randn(1000))
        a = Reactant.to_rarray(a_data)
        b = Reactant.to_rarray(b_data)
        result_ra = @jit add_arrays(a, b)
        # Immediate conversion -- AwaitBufferReady must sync XLA's stream
        cu = CuArray(result_ra)
        @test Array(cu) ≈ a_data .+ b_data
    end
end

matmul(x, y) = x * y

@testset "Reverse: convert output of @jit matmul immediately" begin
    for trial in 1:20
        a_data = Float32.(randn(128, 64))
        b_data = Float32.(randn(64, 32))
        a = Reactant.to_rarray(a_data)
        b = Reactant.to_rarray(b_data)
        result_ra = @jit matmul(a, b)
        cu = CuArray(result_ra)
        @test Array(cu) ≈ a_data * b_data
    end
end

# ============================================================
# 6. Interleaved operations: alternate between CUDA.jl and
#    Reactant operations on the same data, forcing repeated
#    cross-context transfers.
# ============================================================
@testset "Interleaved CUDA.jl and Reactant operations" begin
    data = Float32.(randn(1000))
    cu = CuArray(data)

    for i in 1:20
        # CUDA.jl: scale by 2
        cu .= cu .* 2f0

        # Transfer to Reactant
        ra = ConcretePJRTArray(cu)

        # Reactant: add 1 via @jit
        add_one(x) = x .+ 1f0
        ra = @jit add_one(ra)

        # Transfer back to CUDA.jl
        cu = CuArray(ra)

        # Track expected value
        data .= data .* 2f0 .+ 1f0
    end

    @test Array(cu) ≈ data
end

# ============================================================
# 7. Rapid repeated conversions (same buffer, many times)
#    Tests that repeated use of the same memory doesn't cause
#    stale reads or double-frees.
# ============================================================
@testset "Rapid repeated forward conversions" begin
    cu = CUDA.rand(Float32, 10_000)
    CUDA.synchronize()
    expected = Array(cu)

    results = [Array(ConcretePJRTArray(cu)) for _ in 1:100]
    @test all(r == expected for r in results)
end

@testset "Rapid repeated reverse conversions" begin
    data = Float32.(randn(10_000))
    ra = Reactant.to_rarray(data)

    results = [Array(CuArray(ra)) for _ in 1:100]
    @test all(r == data for r in results)
end

# ============================================================
# 8. Overwrite source after conversion
#    Verify the converted array is an independent copy, not a
#    view. Mutating the source must not affect the destination.
# ============================================================
@testset "Forward: source mutation doesn't affect destination" begin
    cu = CuArray(Float32[1, 2, 3, 4, 5])
    ra = ConcretePJRTArray(cu)
    # Overwrite the CuArray
    cu .= 0f0
    CUDA.synchronize()
    # The ConcreteRArray should still have the original data
    @test Array(ra) == Float32[1, 2, 3, 4, 5]
end

@testset "Reverse: source mutation doesn't affect destination" begin
    data = Float32[10, 20, 30, 40, 50]
    ra = Reactant.to_rarray(data)
    cu = CuArray(ra)
    # The original ConcreteRArray is immutable from Julia's perspective,
    # but verify the CuArray is independent by checking it persists
    # after ra goes out of scope
    ra = nothing
    GC.gc()
    @test Array(cu) == Float32[10, 20, 30, 40, 50]
end

# ============================================================
# 9. Edge cases
# ============================================================
@testset "Edge cases" begin
    @testset "Single element" begin
        cu = CuArray(Float32[42.0])
        ra = ConcretePJRTArray(cu)
        @test Array(ra) == Float32[42.0]

        cu_back = CuArray(ra)
        @test Array(cu_back) == Float32[42.0]
    end

    @testset "Empty array" begin
        cu = CuArray(Float32[])
        ra = ConcretePJRTArray(cu)
        @test size(ra) == (0,)
        @test Array(ra) == Float32[]

        ra_empty = Reactant.to_rarray(Float32[])
        cu_back = CuArray(ra_empty)
        @test size(cu_back) == (0,)
        @test Array(cu_back) == Float32[]
    end

    @testset "Large array (100MB)" begin
        cu = CUDA.rand(Float32, 25_000_000)  # 100MB
        CUDA.synchronize()
        expected = Array(cu)
        ra = ConcretePJRTArray(cu)
        @test Array(ra) == expected
    end

    @testset "Float16" begin
        data = Float16[1.0, 2.0, 3.5, 4.25, 5.0]
        cu = CuArray(data)
        ra = ConcretePJRTArray(cu)
        @test Array(ra) == data
        @test eltype(ra) == Float16

        cu_back = CuArray(ra)
        @test Array(cu_back) == data
    end

    @testset "BFloat16" begin
        data = Core.BFloat16[1.0, 2.0, 3.5, 4.25, 5.0]
        cu = CuArray(data)
        ra = ConcretePJRTArray(cu)
        @test Array(ra) == data
        @test eltype(ra) == Core.BFloat16

        cu_back = CuArray(ra)
        @test Array(cu_back) == data
    end

    @testset "Bool arrays" begin
        data = Bool[true, false, true, true, false]
        cu = CuArray(data)
        ra = ConcretePJRTArray(cu)
        @test Array(ra) == data
        @test eltype(ra) == Bool

        cu_back = CuArray(ra)
        @test Array(cu_back) == data
    end

    @testset "Non-contiguous view (materializes contiguous copy)" begin
        # Non-contiguous views dispatch to a convenience method that
        # materializes a contiguous CuArray before the D2D copy.
        cu_full = CuArray(reshape(collect(Float32, 1:100), 10, 10))
        cu_view = @view cu_full[1:2:end, :]  # stride-2 view, not contiguous
        ra = ConcretePJRTArray(cu_view)
        @test Array(ra) == Array(cu_view)
    end

    @testset "High-dimensional array (5D)" begin
        data = Float32.(randn(2, 3, 4, 5, 6))
        cu = CuArray(data)
        ra = ConcretePJRTArray(cu)
        @test size(ra) == (2, 3, 4, 5, 6)
        @test Array(ra) ≈ data

        cu_back = CuArray(ra)
        @test Array(cu_back) ≈ data
    end
end

end # if CUDA.functional()

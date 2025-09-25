using PythonCall, Reactant, Test

pyimport("sys").path.append(@__DIR__)

softmax_kernel = pyimport("softmax").softmax_kernel

const RunningOnCUDA = contains(string(Reactant.devices()[1]), "CUDA")

function softmax_naive(x::AbstractMatrix{T}) where {T}
    x_max = maximum(x; dims=1)
    z = x .- x_max
    num = exp.(z)
    denom = sum(num; dims=1)
    return num ./ denom
end

function softmax_triton(x::AbstractMatrix{T}) where {T}
    x_transposed = permutedims(x, (2, 1)) # match python array layout
    out = similar(x_transposed)
    n_rows, n_cols = size(x_transposed)

    BLOCK_SIZE = nextpow(2, n_cols)

    function grid_fn(metadata)
        occupancy = (
            metadata.device_properties.regs_per_block ÷
            (metadata.num_regs * metadata.device_properties.warp_size * metadata.num_warps)
        )

        num_programs = min(
            metadata.device_properties.multi_processor_count * min(
                occupancy,
                metadata.device_properties.shared_mem_per_block ÷ metadata.metadata.shared,
            ),
            n_rows,
        )
        return num_programs
    end

    softmax_kernel(
        out,
        x_transposed,
        Reactant.rowmajor_stride(x_transposed, 1),
        Reactant.rowmajor_stride(out, 1),
        n_rows,
        n_cols,
        BLOCK_SIZE,
        num_stages=3;
        grid=grid_fn,
    )

    return permutedims(out, (2, 1))
end

@testset "softmax" begin
    if RunningOnCUDA
        x_ra = Reactant.to_rarray(rand(Float32, 132, 2056))

        @test @jit(softmax_triton(x_ra)) ≈ @jit(softmax_naive(x_ra))
    end
end

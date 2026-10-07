using NonuniformFFTs, Reactant, Test, CUDA, LinearAlgebra, KernelAbstractions

const RunningOnCPU = contains(string(Reactant.devices()[1]), "CPU")
const RunningOnCUDA = contains(string(Reactant.devices()[1]), "CUDA")

# platforms that support cfunction with closures
# (requires LLVM back-end support for trampoline intrinsics)
const cfunction_closure = Sys.ARCH === :x86_64 || Sys.ARCH === :i686

function nfft!(
    out::AbstractVector{<:Complex}, x::AbstractVector{<:Real}, v::AbstractVector{<:Real}
)
    plan_nufft = PlanNUFFT(
        eltype(x), 256; m=HalfSupport(4), backend=KernelAbstractions.get_backend(x)
    )
    set_points!(plan_nufft, x)
    exec_type1!(out, plan_nufft, v)
    return nothing
end

function nfft(x, v)
    ûs = similar(v, Complex{eltype(x)}, 129)
    nfft!(ûs, x, v)
    return ûs
end

function traced_nfft(x, v)
    out = Reactant.Ops.julia_callback(
        nfft!, ((Complex{Reactant.unwrapped_eltype(x)}, 129),), x, v
    )
    return out
end

if (RunningOnCPU || RunningOnCUDA) && cfunction_closure
    @testset "NonuniformFFTs callback" begin
        Np = 100  # number of non-uniform points

        # Generate some non-uniform random data
        T = Float64                # non-uniform data is real (can also be complex)
        xp = rand(T, Np) .* T(2π)  # non-uniform points in [0, 2π]
        vp = randn(T, Np)          # random values at points

        res = nfft(xp, vp)

        x_ra = Reactant.to_rarray(xp)
        v_ra = Reactant.to_rarray(vp)

        res_ra = @jit traced_nfft(x_ra, v_ra)

        @test res ≈ res_ra atol = 1e-5 rtol = 1e-5
    end
end

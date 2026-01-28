using AbstractFFTs
const AFTR = Base.get_extension(Reactant, :ReactantAbstractFFTsExt)

struct ReactantNFFTPlan{T,D,K<:AbstractArray,arrTc,vecI,vecII,FP,BP,INV,SM} <:
       AbstractNFFTPlan{T,D,1}
    N::NTuple{D,Int}
    NOut::NTuple{1,Int}
    J::Int
    k::K
    Ñ::NTuple{D,Int}
    dims::UnitRange{Int}
    forwardFFT::FP
    backwardFFT::BP
    tmpVec::arrTc
    tmpVecHat::arrTc
    deconvolveIdx::vecI
    windowLinInterp::vecII
    windowHatInvLUT::INV
    B::SM
end


function AbstractNFFTs.plan_nfft(
    arr::Type{<:Reactant.AnyTracedRArray},
    k::AbstractMatrix,
    N::NTuple{D,Int},
    rest...;
    kargs...,
) where {D}
    p = ReactantNFFTPlan(arr, k, N; kargs...)
    return p
end

function Reactant.make_tracer(
    seen,
    @nospecialize(prev::LinearAlgebra.Adjoint{T,<:NFFT.AbstractNFFTPlan}),
    @nospecialize(path),
    mode;
    @nospecialize(track_numbers::Type = Union{}),
    @nospecialize(sharding = Reactant.Sharding.NoSharding()),
    @nospecialize(runtime),
    kwargs...,
) where {T}
    return prev
end

function Reactant.traced_type_inner(
    @nospecialize(T::Type{<:LinearAlgebra.Adjoint{F,<:NFFT.AbstractNFFTPlan}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(ndevices),
    @nospecialize(runtime)
) where {F}
    return T
end


function ReactantNFFTPlan(
    k::AbstractArray{T}, N::NTuple{D,Int}; fftflags=nothing, kwargs...
) where {T,D}


    dims = 1:D
    CT = complex(T)
    params, N, NOut, J, Ñ, dims_ = NFFT.initParams(k, N, dims; kwargs...)
    FP = plan_fft!(zeros(CT, N))
    BP = plan_bfft!(zeros(CT, N))

    FP = AFTR.reactant_fftplan(AFTR.reactant_fftplan_type(typeof(FP)), FP)
    BP = AFTR.reactant_fftplan(AFTR.reactant_fftplan_type(typeof(BP)), BP)

    params.storeDeconvolutionIdx = true # GPU_NFFT only works this way
    params.precompute = NFFT.FULL # GPU_NFFT only works this way

    windowLinInterp, windowPolyInterp, windowHatInvLUT, deconvolveIdx, B = NFFT.precomputation(
        k, N[dims_], Ñ[dims_], params
    )

    U = params.storeDeconvolutionIdx ? N : ntuple(d -> 0, Val(D))

    tmpVec = ConcreteRArray(zeros(CT, Ñ))
    tmpVecHat = ConcreteRArray(zeros(CT, U))
    deconvIdx = ConcreteRArray(Int.(deconvolveIdx))
    winHatInvLUT = ConcreteRArray(complex(windowHatInvLUT[1]))
    B_ = (ConcreteRArray(complex.(Array(B))))

    return ReactantNFFTPlan{
        T,
        D,
        typeof(k),
        typeof(tmpVec),
        typeof(deconvIdx),
        typeof(windowLinInterp),
        typeof(FP),
        typeof(BP),
        typeof(winHatInvLUT),
        typeof(B_),
    }(
        N,
        NOut,
        J,
        k,
        Ñ,
        dims_,
        FP,
        BP,
        tmpVec,
        tmpVecHat,
        deconvIdx,
        windowLinInterp,
        winHatInvLUT,
        B_,
    )
end

AbstractNFFTs.size_in(p::ReactantNFFTPlan) = p.N
AbstractNFFTs.size_out(p::ReactantNFFTPlan) = p.NOut

function AbstractNFFTs.convolve!(
    p::ReactantNFFTPlan{T,D}, g::Reactant.AnyTracedRArray, fHat::Reactant.AnyTracedRArray
) where {D,T}
    mul!(fHat, transpose(p.B), vec(g))
    return nothing
end

function AbstractNFFTs.convolve_transpose!(
    p::ReactantNFFTPlan{T,D}, fHat::Reactant.AnyTracedRArray, g::Reactant.AnyTracedRArray
) where {D,T}
    mul!(vec(g), p.B, fHat)
    return nothing
end

# function AbstractNFFTs.deconvolve_transpose!(p::ReactantNFFTPlan{T,D}, g::Reactant.AnyTracedRArray, f::Reactant.AnyTracedRArray) where {D,T}
#     p.tmpVecHat[:] .= broadcast(p.deconvolveIdx) do idx
#       g[idx]
#     end
#     f[:] .= vec(p.tmpVecHat) .* p.windowHatInvLUT
#     return
# end

function Base.:*(p::ReactantNFFTPlan{T}, f::Reactant.AnyTracedRArray; kargs...) where {T}
    fHat = similar(f, Complex{T}, size_out(p))
    mul!(fHat, p, f; kargs...)
    return fHat
end

function AbstractNFFTs.deconvolve!(
    p::ReactantNFFTPlan{T,D}, f::AbstractArray, g::AbstractArray
) where {D,T}
    tmp = f .* reshape(p.windowHatInvLUT, size(f))
    @allowscalar @inbounds gv = @view(g[p.deconvolveIdx])
    @allowscalar g[p.deconvolveIdx] = reshape(tmp, :)
    return nothing
end

"""  in-place NFFT on the GPU"""
function LinearAlgebra.mul!(
    fHat::Reactant.AnyTracedRArray,
    p::ReactantNFFTPlan{T,D},
    f::Reactant.AnyTracedRArray;
    verbose=false,
    timing::Union{Nothing,TimingStats}=nothing,
) where {T,D}
    NFFT.consistencyCheck(p, f, fHat)

    fill!(p.tmpVec, zero(Complex{T}))
    t1 = @elapsed @inbounds deconvolve!(p, f, p.tmpVec)
    fHat .= p.tmpVec[1:length(fHat)]
    p.forwardFFT * p.tmpVec
    return t3 = @elapsed @inbounds NFFT.convolve!(p, p.tmpVec, fHat)
end

function NFFT.nfft(k::AbstractMatrix, f::Reactant.AnyTracedRArray, args...; kwargs...)
    p = ReactantNFFTPlan(typeof(f), k, size(f))
    return p * f
end

#   """  in-place adjoint NFFT on the GPU"""
# function LinearAlgebra.mul!(f::Reactant.AnyTracedRArray, pl::Adjoint{Complex{T},<:ReactantNFFTPlan{T,D}}, fHat::Reactant.AnyTracedRArray;
#     verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T,D}
# #   NFFT.consistencyCheck(p, f, fHat)
#     p = pl.parent

#     t1 = @elapsed @inbounds NFFT.convolve_transpose!(p, fHat, p.tmpVec)
#     tmp = ifft(p.tmpVec)
#     p.tmpVec .= tmp.*length(tmp)
#     t3 = @elapsed @inbounds NFFT.deconvolve_transpose!(p, p.tmpVec, f)
#     if verbose
#         @info "Timing: conv=$t1 fft=$t2 deconv=$t3"
#     end
#     if timing != nothing
#       timing.conv_adjoint = t1
#       timing.fft_adjoint = t2
#       timing.deconv_adjoint = t3
#     end

#     return f
#   end

function NFFT.initParams(
    k::AbstractMatrix{T},
    N::NTuple{D,Int},
    dims::Union{Integer,UnitRange{Int64}}=1:D;
    kargs...,
) where {D,T}
    # convert dims to a unit range
    dims_ = (typeof(dims) <: Integer) ? (dims:dims) : dims

    params = NFFTParams{T,D}(; kargs...)
    m, σ, reltol = accuracyParams(; kargs...)
    params.m = m
    params.σ = σ
    params.reltol = reltol

    # Taken from NFFT3
    m2K = [1, 3, 7, 9, 14, 17, 20, 23, 24]
    K = m2K[min(m + 1, length(m2K))]
    params.LUTSize = 2^(K) * (m) # ensure that LUTSize is dividable by (m)

    if length(dims_) != size(k, 1)
        throw(ArgumentError("Nodes x have dimension $(size(k,1)) != $(length(dims_))"))
    end

    doTrafo = ntuple(d -> d ∈ dims_, Val(D))

    Ñ = ntuple(d ->
        doTrafo[d] ? (ceil(Int, params.σ * N[d]) ÷ 2) * 2 : # ensure that n is an even integer
        N[d], Val(D))

    params.σ = Ñ[dims_[1]] / N[dims_[1]]

    #params.blockSize = ntuple(d-> Ñ[d] , D) # just one block
    if haskey(kargs, :blockSize)
        params.blockSize = kargs[:blockSize]
    else
        params.blockSize = ntuple(d -> NFFT._blockSize(Ñ, d), Val(D))
    end

    J = size(k, 2)

    # calculate output size
    NOut = Int[]
    Mtaken = false
    ntuple(Val(D)) do d
        if !doTrafo[d]
            return N[d]
        elseif !Mtaken
            return J
            Mtaken = true
        end
    end
    for d in 1:D
        if !doTrafo[d]
            push!(NOut, N[d])
        elseif !Mtaken
            push!(NOut, J)
            Mtaken = true
        end
    end
    # Sort nodes in lexicographic way
    if params.sortNodes
        k .= sortslices(k; dims=2)
    end
    return params, N, Tuple(NOut), J, Ñ, dims_
end

function NFFT.precomputation(k::AbstractVecOrMat, N::NTuple{D,Int}, Ñ, params) where {D}
    m = params.m
    σ = params.σ
    window = params.window
    LUTSize = params.LUTSize
    precompute = params.precompute

    win, win_hat = getWindow(window) # highly type instable. But what should be do
    J = size(k, 2)

    windowHatInvLUT_ = Vector{Vector{T}}(undef, D)
    precomputeWindowHatInvLUT(windowHatInvLUT_, win_hat, N, Ñ, m, σ, T)

    if params.storeDeconvolutionIdx
        windowHatInvLUT = Vector{Vector{T}}(undef, 1)
        windowHatInvLUT[1], deconvolveIdx = precompWindowHatInvLUT(
            params, N, Ñ, windowHatInvLUT_
        )
    else
        windowHatInvLUT = windowHatInvLUT_
        deconvolveIdx = Array{Int64,1}(undef, 0)
    end

    if precompute == LINEAR
        windowLinInterp = precomputeLinInterp(win, m, σ, LUTSize, T)
        windowPolyInterp = Matrix{T}(undef, 0, 0)
        B = sparse([], [], T[])
    elseif precompute == POLYNOMIAL
        windowLinInterp = Vector{T}(undef, 0)
        windowPolyInterp = precomputePolyInterp(win, m, σ, T)
        B = sparse([], [], T[])
    elseif precompute == FULL
        windowLinInterp = Vector{T}(undef, 0)
        windowPolyInterp = Matrix{T}(undef, 0, 0)
        B = precomputeB(win, k, N, Ñ, m, J, σ, LUTSize, T)
        #windowLinInterp = precomputeLinInterp(win, windowLinInterp, Ñ, m, σ, LUTSize, T) # These versions are for debugging
        #B = precomputeB(windowLinInterp, k, N, Ñ, m, J, σ, LUTSize, T)
    elseif precompute == TENSOR
        windowLinInterp = Vector{T}(undef, 0)
        windowPolyInterp = Matrix{T}(undef, 0, 0)
        B = sparse([], [], T[])
    else
        windowLinInterp = Vector{T}(undef, 0)
        windowPolyInterp = Matrix{T}(undef, 0, 0)
        B = sparse([], [], T[])
        error("precompute = $precompute not supported by NFFT.jl!")
    end

    return (windowLinInterp, windowPolyInterp, windowHatInvLUT, deconvolveIdx, B)
end

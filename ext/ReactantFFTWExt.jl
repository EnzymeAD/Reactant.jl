module ReactantFFTWExt

using FFTW
using AbstractFFTs
using LinearAlgebra
using Reactant

const FORWARD = -1
const BACKWARD = 1

#################################
# c2c plans
function Base.:*(plan::FFTW.cFFTWPlan{T,FORWARD,true}, x::Reactant.TracedRArray) where {T}
    return fft!(x, fftdims(plan))
end

function Base.:*(
    plan::FFTW.cFFTWPlan{T,FORWARD,false}, x::Reactant.TracedRArray{T}
) where {T}
    return fft(x, fftdims(plan))
end

function LinearAlgebra.mul!(
    y::Reactant.TracedRArray{T},
    plan::FFTW.cFFTWPlan{T,FORWARD},
    x::Reactant.TracedRArray{T},
) where {T}
    return copyto!(y, fft(x, fftdims(plan)))
end

# This is the actual ifft! operation. Backwards in FFTW actually means bfft
const IcFFTWPlan{T,inplace} = AbstractFFTs.ScaledPlan{
    T,<: FFTW.cFFTWPlan{T,BACKWARD,inplace}
}
const IrFFTWPlan{T,inplace} = AbstractFFTs.ScaledPlan{
    T,<: FFTW.rFFTWPlan{T,BACKWARD,inplace}
}
# TODO support bfft
function Base.:*(plan::IcFFTWPlan{T,true}, x::Reactant.TracedRArray{T}) where {T}
    return ifft!(x, fftdims(plan))
end

function Base.:*(plan::IcFFTWPlan{T,false}, x::Reactant.TracedRArray{T}) where {T}
    return ifft(x, fftdims(plan))
end

function LinearAlgebra.mul!(
    y::Reactant.TracedRArray{T}, plan::IcFFTWPlan{T}, x::Reactant.TracedRArray{T}
) where {T}
    return copyto!(y, ifft(x, fftdims(plan)))
end

#################################
## r2c plans
function Base.:*(
    plan::FFTW.rFFTWPlan{T,FORWARD,false}, x::Reactant.TracedRArray{T}
) where {T}
    return rfft(x, fftdims(plan))
end

reallength(p::IrFFTWPlan{T}) where {T} = p.p.osz[first(fftdims(p))] # original real length

# TODO add support for irfft plans
function Base.:*(plan::IrFFTWPlan{T,false}, x::Reactant.TracedRArray{T}) where {T}
    d = reallength(plan) # original real length
    return irfft(x, d, fftdims(plan))
end

# inplace versions do not exist because types always differ!

function LinearAlgebra.mul!(
    y::Reactant.TracedRArray{<:Complex},
    plan::FFTW.rFFTWPlan{<:Real,FORWARD},
    x::Reactant.TracedRArray{<:Real},
)
    return copyto!(y, rfft(x, fftdims(plan)))
end


function LinearAlgebra.mul!(
    y::Reactant.TracedRArray{<:Real}, plan::IrFFTWPlan{<:Complex}, x::Reactant.TracedRArray{<:Complex}
)
    d = reallength(plan) # original real length
    return copyto!(y, irfft(x, d, fftdims(plan)))
end

end

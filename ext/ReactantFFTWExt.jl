module ReactantFFTWExt

using FFTW
using AbstractFFTs
using LinearAlgebra
using Reactant

const FORWARD = -1
const BACKWARD = 1

# Is this actually okay to do? I am assuming that AbstractFFTsExt will already be loaded since it 
# is a dependency of this extension
const AFTEx = Base.get_extension(Reactant, :ReactantAbstractFFTsExt)

#################################
# c2c plans
# This is the actual ifft! operation. Backwards in FFTW actually means bfft

dimstype(::Type{<:FFTW.cFFTWPlan{T,K,inplace,N,G}}) where {T,K,inplace,N,G} = G
dimstype(::Type{<:FFTW.rFFTWPlan{T,K,inplace,N,G}}) where {T,K,inplace,N,G} = G

function AFTEx.reactant_fftplan_type(P::Type{<:FFTW.cFFTWPlan{T,FORWARD,true}}) where {T}
    return AFTEx.ReactantFFTInPlacePlan{T,dimstype(P)}
end
function AFTEx.make_reactant_fftplan(plan::FFTW.cFFTWPlan{T,FORWARD,true}) where {T}
    return AFTEx.ReactantFFTInPlacePlan{T}(fftdims(plan))
end

function AFTEx.reactant_fftplan_type(P::Type{<:FFTW.cFFTWPlan{T,FORWARD,false}}) where {T}
    return AFTEx.ReactantFFTPlan{T,dimstype(P)}
end
function AFTEx.make_reactant_fftplan(plan::FFTW.cFFTWPlan{T,FORWARD,false}) where {T}
    return AFTEx.ReactantFFTPlan{T}(fftdims(plan))
end

# XLA does plan based on IFFT not BFFT so we need to renormalize
function AFTEx.reactant_fftplan_type(P::Type{<:FFTW.cFFTWPlan{T,BACKWARD,true}}) where {T}
    return AbstractFFTs.ScaledPlan{T,AFTEx.ReactantIFFTInPlacePlan{T,dimstype(P)},real(T)}
end
function AFTEx.make_reactant_fftplan(plan::FFTW.cFFTWPlan{T,BACKWARD,true}) where {T}
    return AFTEx.ReactantIFFTInPlacePlan{T}(fftdims(plan)) *
           AFTEx.normbfft(T, size(plan), fftdims(plan))
end

function AFTEx.reactant_fftplan_type(P::Type{<:FFTW.cFFTWPlan{T,BACKWARD,false}}) where {T}
    return AbstractFFTs.ScaledPlan{T,AFTEx.ReactantIFFTPlan{T,dimstype(P)},real(T)}
end
function AFTEx.make_reactant_fftplan(plan::FFTW.cFFTWPlan{T,BACKWARD,false}) where {T}
    nrm = AFTEx.normbfft(T, size(plan), fftdims(plan))
    return AFTEx.ReactantIFFTPlan{T}(fftdims(plan)) * nrm
end

AFTEx.reallength(p::FFTW.rFFTWPlan{T,BACKWARD}) where {T} = p.osz[first(fftdims(p))] # original real length
# We don't define the inplace versions becuase the types always differ
function AFTEx.reactant_fftplan_type(P::Type{<:FFTW.rFFTWPlan{T,FORWARD}}) where {T}
    return AFTEx.ReactantRFFTPlan{T,dimstype(P)}
end
function AFTEx.make_reactant_fftplan(plan::FFTW.rFFTWPlan{T,FORWARD,false}) where {T}
    return AFTEx.ReactantRFFTPlan{T}(fftdims(plan))
end

function AFTEx.reactant_fftplan_type(P::Type{<:FFTW.rFFTWPlan{T,BACKWARD}}) where {T}
    return AbstractFFTs.ScaledPlan{T,AFTEx.ReactantIRFFTPlan{T,dimstype(P)},real(T)}
end

function AFTEx.make_reactant_fftplan(plan::FFTW.rFFTWPlan{T,BACKWARD}) where {T}
    osz = AbstractFFTs.brfft_output_size(size(plan), AFTEx.reallength(plan), fftdims(plan)) # just to make sure it is defined
    nrm = AFTEx.normbfft(real(T), osz, fftdims(plan))
    return AFTEx.ReactantIRFFTPlan{T}(fftdims(plan), AFTEx.reallength(plan)) * nrm
end

end

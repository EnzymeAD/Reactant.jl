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

function AFTEx.reactant_fftplan(plan::FFTW.cFFTWPlan{T,FORWARD,true}) where {T}
    return AFTEx.ReactantFFTInPlacePlan{T}(fftdims(plan))
end
function AFTEx.reactant_fftplan(plan::FFTW.cFFTWPlan{T,FORWARD,false}) where {T}
    return AFTEx.ReactantFFTPlan{T}(fftdims(plan))
end

# XLA does plan based on IFFT not BFFT so we need to renormalize
function AFTEx.reactant_fftplan(plan::FFTW.cFFTWPlan{T,BACKWARD,true}) where {T}
    return AFTEx.ReactantIFFTInPlacePlan{T}(fftdims(plan)) * AFTEx.normbfft(T, size(plan), fftdims(plan))
end

function AFTEx.reactant_fftplan(plan::FFTW.cFFTWPlan{T,BACKWARD,false}) where {T}
    nrm = AbstractFFTs.normalization(real(T), size(plan), fftdims(plan))
    return AFTEx.ReactantFFTPlan{T}(fftdims(plan)) * AFTEx.normbfft(T, size(plan), fftdims(plan))
end

reallength(p::IrFFTWPlan{T}) where {T} = p.p.osz[first(fftdims(p))] # original real length
# We don't define the inplace versions becuase the types always differ
function AFTEx.reactant_fftplan(plan::FFTW.rFFTWPlan{T,FORWARD,false}) where {T}
    return AFTEx.ReactantRFFTPlan{T}(fftdims(plan))
end

function AFTEx.reactant_fftplan(plan::FFTW.rFFTWPlan{T,BACKWARD,false}) where {T}
    nrm = AFTEx.normbrfft(T, size(plan), fftdims(plan))
    return AFTEx.ReactantIRFFTPlan{T}(fftdims(plan), reallength(plan)) * nrm
end

end

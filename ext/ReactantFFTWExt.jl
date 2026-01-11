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
const IcFFTWPlan3{T,inplace} = AbstractFFTs.ScaledPlan{
    T,<:FFTW.cFFTWPlan{T,BACKWARD,inplace}
}

function AFTEx.reactant_fftplan(plan::FFTW.cFFTWPlan{T,FORWARD,true}) where {T}
    return AFTEx.ReactantFFTInPlacePlan{T}(fftdims(plan))
end
function AFTEx.reactant_fftplan(plan::FFTW.cFFTWPlan{T,FORWARD,false}) where {T}
    return AFTEx.ReactantFFTPlan{T}(fftdims(plan))
end
function AFTEx.reactant_fftplan(plan::IcFFTWPlan3{T,false}) where {T}
    return AFTEx.ReactantIFFTPlan{T}(fftdims(plan))
end
function AFTEx.reactant_fftplan(plan::IcFFTWPlan3{T,true}) where {T}
    return AFTEx.ReactantIFFTInPlacePlan{T}(fftdims(plan))
end

const IrFFTWPlan{T,inplace} = AbstractFFTs.ScaledPlan{
    T,<:FFTW.rFFTWPlan{T,BACKWARD,inplace}
}

reallength(p::IrFFTWPlan{T}) where {T} = p.p.osz[first(fftdims(p))] # original real length

# We don't define the inplace versions becuase the types always differ
function AFTEx.reactant_fftplan(plan::FFTW.rFFTWPlan{T,FORWARD,false}) where {T}
    return AFTEx.ReactantRFFTPlan{T}(fftdims(plan))
end
function AFTEx.reactant_fftplan(plan::IrFFTWPlan{T,false}) where {T}
    return AFTEx.ReactantIRFFTPlan{T}(fftdims(plan), reallength(plan))
end

end

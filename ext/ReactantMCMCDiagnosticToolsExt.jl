module ReactantMCMCDiagnosticToolsExt

using Reactant.ProbProg: ProbProg
using MCMCDiagnosticTools: ess, rhat

function ProbProg._compute_ess(samples::AbstractVector)
    x = collect(Float64, samples)
    n = length(x)
    if n < 4
        return Float64(n)
    end
    x_matrix = reshape(x, n, 1)
    return ess(x_matrix)
end

function ProbProg._compute_rhat(samples::AbstractVector)
    x = collect(Float64, samples)
    n = length(x)
    if n < 4
        return NaN
    end
    x_matrix = reshape(x, n, 1)
    return rhat(x_matrix)
end

end

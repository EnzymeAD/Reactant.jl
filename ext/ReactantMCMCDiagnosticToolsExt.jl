module ReactantMCMCDiagnosticToolsExt

using Reactant.ProbProg: ProbProg, ParameterSummary
using Statistics: mean, std, median, quantile
using MCMCDiagnosticTools: ess, rhat

function _compute_ess(samples::AbstractVector)
    x = collect(Float64, samples)
    n = length(x)
    if n < 4
        return Float64(n)
    end
    x_matrix = reshape(x, n, 1)
    return ess(x_matrix)
end

function _compute_rhat(samples::AbstractVector)
    x = collect(Float64, samples)
    n = length(x)
    if n < 4
        return NaN
    end
    x_matrix = reshape(x, n, 1)
    return rhat(x_matrix)
end

function ProbProg._compute_parameter_summary(name::String, samples::AbstractVector)
    return ParameterSummary(
        name,
        mean(samples),
        std(samples; corrected=true),
        median(samples),
        quantile(samples, 0.05),
        quantile(samples, 0.95),
        _compute_ess(samples),
        _compute_rhat(samples),
    )
end

end

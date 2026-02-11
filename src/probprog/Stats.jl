using PrettyTables: pretty_table, fmt__printf

struct ParameterSummary
    name::String
    mean::Float64
    std::Float64
    median::Float64
    q5::Float64
    q95::Float64
    n_eff::Float64
    r_hat::Float64
end

struct MCMCSummary <: AbstractVector{ParameterSummary}
    parameters::Vector{ParameterSummary}
    _name_to_idx::Dict{String,Int}

    function MCMCSummary(parameters::Vector{ParameterSummary})
        name_to_idx = Dict{String,Int}()
        for (i, p) in enumerate(parameters)
            name_to_idx[p.name] = i
        end
        return new(parameters, name_to_idx)
    end
end

Base.size(s::MCMCSummary) = (length(s.parameters),)
Base.getindex(s::MCMCSummary, i::Int) = s.parameters[i]
Base.IndexStyle(::Type{MCMCSummary}) = IndexLinear()

function Base.getindex(s::MCMCSummary, name::Symbol)
    return s[string(name)]
end

function Base.getindex(s::MCMCSummary, name::String)
    idx = get(s._name_to_idx, name, nothing)
    if idx === nothing
        throw(KeyError(name))
    end
    return s.parameters[idx]
end

Base.haskey(s::MCMCSummary, name::Symbol) = haskey(s._name_to_idx, string(name))
Base.haskey(s::MCMCSummary, name::String) = haskey(s._name_to_idx, name)
Base.keys(s::MCMCSummary) = (p.name for p in s.parameters)

function _column_names()
    return (:name, :mean, :std, :median, :q5, :q95, :n_eff, :r_hat)
end

Base.iterate(s::MCMCSummary) = iterate(s.parameters)
Base.iterate(s::MCMCSummary, state) = iterate(s.parameters, state)
Base.length(s::MCMCSummary) = length(s.parameters)
Base.eltype(::Type{MCMCSummary}) = ParameterSummary

function _to_namedtuple(p::ParameterSummary)
    return (;
        name=p.name,
        mean=p.mean,
        std=p.std,
        median=p.median,
        q5=p.q5,
        q95=p.q95,
        n_eff=p.n_eff,
        r_hat=p.r_hat,
    )
end

function Base.NamedTuple(s::MCMCSummary)
    return (;
        name=[p.name for p in s.parameters],
        mean=[p.mean for p in s.parameters],
        std=[p.std for p in s.parameters],
        median=[p.median for p in s.parameters],
        q5=[p.q5 for p in s.parameters],
        q95=[p.q95 for p in s.parameters],
        n_eff=[p.n_eff for p in s.parameters],
        r_hat=[p.r_hat for p in s.parameters],
    )
end

function Base.show(io::IO, ::MIME"text/plain", s::MCMCSummary)
    if isempty(s.parameters)
        println(io, "MCMCSummary (empty)")
        return nothing
    end

    data = Matrix{Any}(undef, length(s.parameters), 8)
    for (i, p) in enumerate(s.parameters)
        data[i, 1] = p.name
        data[i, 2] = p.mean
        data[i, 3] = p.std
        data[i, 4] = p.median
        data[i, 5] = p.q5
        data[i, 6] = p.q95
        data[i, 7] = p.n_eff
        data[i, 8] = p.r_hat
    end

    return pretty_table(
        io,
        data;
        column_labels=["", "mean", "std", "median", "5.0%", "95.0%", "n_eff", "r_hat"],
        alignment=[:r, :r, :r, :r, :r, :r, :r, :r],
        formatters=[fmt__printf("%.2f", collect(2:8))],
        fit_table_in_display_vertically=false,
    )
end

function Base.show(io::IO, s::MCMCSummary)
    return print(io, "MCMCSummary($(length(s.parameters)) parameters)")
end

function Base.show(io::IO, p::ParameterSummary)
    return print(
        io,
        "ParameterSummary($(p.name): mean=$(round(p.mean; digits=2)), std=$(round(p.std; digits=2)))",
    )
end

function _compute_parameter_summary end

function mcmc_summary(
    samples::AbstractMatrix{<:Real}; names::Union{Nothing,AbstractVector{String}}=nothing
)
    _, n_dims = size(samples)
    parameters = ParameterSummary[]

    for d in 1:n_dims
        name = if names !== nothing
            names[d]
        else
            "x[$(d - 1)]"
        end
        push!(parameters, _compute_parameter_summary(name, samples[:, d]))
    end

    return MCMCSummary(parameters)
end

function mcmc_summary(trace::Trace)
    sorted_choices = sort(collect(trace.choices); by=x -> string(x[1]))
    parameters = ParameterSummary[]

    for (key, value) in sorted_choices
        v = value
        while v isa Tuple && length(v) == 1
            v = v[1]
        end

        if !(v isa AbstractArray)
            continue
        end

        if ndims(v) == 1
            push!(parameters, _compute_parameter_summary(string(key), v))
        elseif ndims(v) == 2
            n_samples, n_dims = size(v)

            if n_dims == 1
                push!(parameters, _compute_parameter_summary(string(key), v[:, 1]))
            else
                for d in 1:n_dims
                    name = "$(key)[$(d-1)]"
                    push!(parameters, _compute_parameter_summary(name, v[:, d]))
                end
            end
        end
    end

    return MCMCSummary(parameters)
end

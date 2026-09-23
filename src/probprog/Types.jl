using OrderedCollections: OrderedSet
using ..Reactant: Reactant

mutable struct Trace
    choices::Dict{Symbol,Any}
    retval::Any
    weight::Any
    subtraces::Dict{Symbol,Any}

    function Trace()
        return new(Dict{Symbol,Any}(), nothing, nothing, Dict{Symbol,Any}())
    end
end

struct Address
    path::Vector{Symbol}

    Address(path::Vector{Symbol}) = new(path)
end

Address(sym::Symbol) = Address([sym])
Address(syms::Symbol...) = Address([syms...])

Base.:(==)(a::Address, b::Address) = a.path == b.path
Base.hash(a::Address, h::UInt) = hash(a.path, h)

mutable struct Constraint <: AbstractDict{Address,Any}
    dict::Dict{Address,Any}

    function Constraint(pairs::Pair...)
        dict = Dict{Address,Any}()
        for pair in pairs
            symbols = Symbol[]
            current = pair
            while isa(current, Pair) && isa(current.first, Symbol)
                push!(symbols, current.first)
                current = current.second
            end
            dict[Address(symbols...)] = current
        end
        return new(dict)
    end

    Constraint() = new(Dict{Address,Any}())
    Constraint(d::Dict{Address,Any}) = new(d)
end

Base.getindex(c::Constraint, k::Address) = c.dict[k]
Base.setindex!(c::Constraint, v, k::Address) = (c.dict[k] = v)
Base.delete!(c::Constraint, k::Address) = delete!(c.dict, k)
Base.keys(c::Constraint) = keys(c.dict)
Base.values(c::Constraint) = values(c.dict)
Base.iterate(c::Constraint) = iterate(c.dict)
Base.iterate(c::Constraint, state) = iterate(c.dict, state)
Base.length(c::Constraint) = length(c.dict)
Base.isempty(c::Constraint) = isempty(c.dict)
Base.haskey(c::Constraint, k::Address) = haskey(c.dict, k)
Base.get(c::Constraint, k::Address, default) = get(c.dict, k, default)

extract_addresses(constraint::Constraint) = Set(keys(constraint))

const Selection = OrderedSet{Address}

struct TraceEntry
    symbol::Symbol
    shape::Tuple
    num_elements::Int
    offset::Int
    parent_path::Vector{Symbol}
end

mutable struct TracedTrace
    entries::Vector{TraceEntry}
    position_size::Int
    address_stack::Vector{Symbol}
end
TracedTrace() = TracedTrace(TraceEntry[], 0, Symbol[])

mutable struct DualAveragingState
    log_step_size::Any
    log_step_size_avg::Any
    gradient_avg::Any
    step_count::Any
    prox_center::Any
end

mutable struct WelfordState
    mean::Any
    m2::Any
    n::Any
end

mutable struct AdaptationState
    dual_averaging::DualAveragingState
    welford::WelfordState
    window_idx::Any
end

function adaptation_operands(da::DualAveragingState)
    return (
        da.log_step_size,
        da.log_step_size_avg,
        da.gradient_avg,
        da.step_count,
        da.prox_center,
    )
end

function adaptation_operands(a::AdaptationState)
    return (
        adaptation_operands(a.dual_averaging)...,
        a.welford.mean,
        a.welford.m2,
        a.welford.n,
        a.window_idx,
    )
end

function AdaptationState(vals)
    @assert length(vals) == 9 "adaptation_state must have 9 entries, got $(length(vals))"
    da = DualAveragingState(vals[1], vals[2], vals[3], vals[4], vals[5])
    welford = WelfordState(vals[6], vals[7], vals[8])
    return AdaptationState(da, welford, vals[9])
end

function Base.copy(da::DualAveragingState)
    return DualAveragingState(
        copy(da.log_step_size),
        copy(da.log_step_size_avg),
        copy(da.gradient_avg),
        copy(da.step_count),
        copy(da.prox_center),
    )
end

Base.copy(w::WelfordState) = WelfordState(copy(w.mean), copy(w.m2), copy(w.n))

function Base.copy(a::AdaptationState)
    return AdaptationState(copy(a.dual_averaging), copy(a.welford), copy(a.window_idx))
end

struct MCMCConfig
    algorithm::Symbol
    max_tree_depth::Int
    max_delta_energy::Float64
    trajectory_length::Float64
    thinning::Int
    strong_zero::Bool
end

function MCMCConfig(;
    algorithm::Symbol=:NUTS,
    max_tree_depth::Int=10,
    max_delta_energy::Float64=1000.0,
    trajectory_length::Float64=2π,
    thinning::Int=1,
    strong_zero::Bool=false,
)
    return MCMCConfig(
        algorithm,
        max_tree_depth,
        max_delta_energy,
        trajectory_length,
        thinning,
        strong_zero,
    )
end

mutable struct MCMCState
    position::Any
    gradient::Any
    potential_energy::Any
    step_size::Any
    inverse_mass_matrix::Any
    rng::Any
    adaptation::Union{Nothing,AdaptationState}
    config::MCMCConfig
end

function MCMCState(
    position,
    gradient,
    potential_energy,
    step_size,
    inverse_mass_matrix,
    rng,
    adaptation::Union{Nothing,AdaptationState}=nothing;
    config::MCMCConfig=MCMCConfig(),
)
    return MCMCState(
        position,
        gradient,
        potential_energy,
        step_size,
        inverse_mass_matrix,
        rng,
        adaptation,
        config,
    )
end

function Base.copy(s::MCMCState)
    return MCMCState(
        copy(s.position),
        copy(s.gradient),
        copy(s.potential_energy),
        copy(s.step_size),
        copy(s.inverse_mass_matrix),
        copy(s.rng),
        s.adaptation === nothing ? nothing : copy(s.adaptation);
        config=s.config,
    )
end

function _config_to_dict(c::MCMCConfig)
    return Dict{String,Any}(
        "algorithm" => String(c.algorithm),
        "max_tree_depth" => c.max_tree_depth,
        "max_delta_energy" => c.max_delta_energy,
        "trajectory_length" => c.trajectory_length,
        "thinning" => c.thinning,
        "strong_zero" => c.strong_zero,
    )
end

function _config_from_dict(d)
    return MCMCConfig(;
        algorithm=Symbol(d["algorithm"]),
        max_tree_depth=d["max_tree_depth"],
        max_delta_energy=d["max_delta_energy"],
        trajectory_length=d["trajectory_length"],
        thinning=d["thinning"],
        strong_zero=d["strong_zero"],
    )
end

function _sampler_to_dict(s::MCMCState)
    return Dict{String,Any}(
        "position" => Array(s.position),
        "gradient" => Array(s.gradient),
        "potential_energy" => Array(s.potential_energy)[],
        "step_size" => Array(s.step_size)[],
        "inverse_mass_matrix" => Array(s.inverse_mass_matrix),
        "rng" => Array(s.rng),
        "config" => _config_to_dict(s.config),
    )
end

function _sampler_from_dict(d)
    config = haskey(d, "config") ? _config_from_dict(d["config"]) : MCMCConfig()
    return MCMCState(
        Reactant.to_rarray(d["position"]),
        Reactant.to_rarray(d["gradient"]),
        Reactant.to_rarray(fill(d["potential_energy"])),
        Reactant.to_rarray(fill(d["step_size"])),
        Reactant.to_rarray(d["inverse_mass_matrix"]),
        Reactant.to_rarray(d["rng"]),
        nothing;
        config=config,
    )
end

function _adaptation_to_dict(a::AdaptationState)
    da = a.dual_averaging
    return Dict{String,Any}(
        "da_log_step_size" => Array(da.log_step_size)[],
        "da_log_step_size_avg" => Array(da.log_step_size_avg)[],
        "da_gradient_avg" => Array(da.gradient_avg)[],
        "da_step_count" => Array(da.step_count)[],
        "da_prox_center" => Array(da.prox_center)[],
        "welford_mean" => Array(a.welford.mean),
        "welford_m2" => Array(a.welford.m2),
        "welford_n" => Array(a.welford.n)[],
        "window_idx" => Array(a.window_idx)[],
    )
end

function _adaptation_from_dict(d::Dict)
    da = DualAveragingState(
        Reactant.to_rarray(fill(d["da_log_step_size"])),
        Reactant.to_rarray(fill(d["da_log_step_size_avg"])),
        Reactant.to_rarray(fill(d["da_gradient_avg"])),
        Reactant.to_rarray(fill(d["da_step_count"])),
        Reactant.to_rarray(fill(d["da_prox_center"])),
    )
    welford = WelfordState(
        Reactant.to_rarray(d["welford_mean"]),
        Reactant.to_rarray(d["welford_m2"]),
        Reactant.to_rarray(fill(d["welford_n"])),
    )
    return AdaptationState(da, welford, Reactant.to_rarray(fill(d["window_idx"])))
end

get_choices(trace::Trace) = trace.choices

function select(addrs::Address...)
    return OrderedSet{Address}(collect(addrs))
end

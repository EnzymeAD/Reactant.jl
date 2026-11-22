using Enzyme: Enzyme
using EnzymeCore: Const, ReverseWithPrimal
using GPUArraysCore: @allowscalar
using ..Reactant:
    Reactant,
    AbstractConcreteArray,
    AbstractRNG,
    Compiler,
    MLIR,
    ReactantRNG,
    to_number,
    to_rarray

struct _DiscoverSimulateThunk{F,R,A}
    f::F
    rng::R
    args::A
end

function (d::_DiscoverSimulateThunk)()
    MLIR.IR.@dispose ctx = Reactant.ReactantContext() begin
        Compiler.compile_mlir(ctx, simulate, (d.rng, d.f, d.args...); optimize=:none)
    end
    return nothing
end

function _discover_model_trace(f::Function, rng::AbstractRNG, args::Tuple)
    tt = TracedTrace()
    with_trace(_DiscoverSimulateThunk(f, rng, args), tt)
    return tt
end

function _entry_address(entry::TraceEntry)
    path = Vector{Symbol}(undef, length(entry.parent_path) + 1)
    copyto!(path, 1, entry.parent_path, 1, length(entry.parent_path))
    path[end] = entry.symbol
    return Address(path)
end

function _find_trace_entry(entries::Vector{TraceEntry}, addr::Address)
    for entry in entries
        if _entry_address(entry) == addr
            return entry
        end
    end
    return nothing
end

function _resolve_observations(
    observations::Constraint, constraints::Union{Nothing,Constraint}
)
    if constraints !== nothing
        if !isempty(observations)
            throw(
                ArgumentError(
                    "Cannot specify both `observations` and `constraints`; pass only one."
                ),
            )
        end
        return constraints
    end
    return observations
end

function _default_probprog_rng(rng::Union{Nothing,AbstractRNG})
    rng !== nothing && return rng
    return ReactantRNG(to_rarray(UInt64[0, 1]))
end

function _collect_addresses(addrs::Union{AbstractSet{Address},AbstractVector{Address}})
    out = Address[]
    for a in addrs
        push!(out, a)
    end
    return out
end

function _prepare_logdensity_spec(
    f::Function,
    rng::AbstractRNG,
    args::Tuple,
    selection::Union{Nothing,Selection},
    obs::Constraint,
)
    tt = _discover_model_trace(f, rng, args)
    if isempty(tt.entries)
        throw(ArgumentError("ProbProg model has no traceable `sample` sites."))
    end

    obs_addrs = Address[]
    obs_flat = Float64[]
    for addr in keys(obs)
        entry = _find_trace_entry(tt.entries, addr)
        if entry === nothing
            throw(
                ArgumentError(
                    "Observation address $(addr) was not found in model sample sites."
                ),
            )
        end
        val = obs[addr]
        n_val = length(val)
        if n_val != entry.num_elements
            throw(
                DimensionMismatch(
                    "Observation at $(addr) has $(n_val) elements, expected $(entry.num_elements).",
                ),
            )
        end
        push!(obs_addrs, addr)
        append!(obs_flat, Float64.(vec(collect(val))))
    end
    obs_set = Set{Address}(obs_addrs)

    resolved_selection = Selection()
    if selection === nothing
        for entry in tt.entries
            addr = _entry_address(entry)
            if !(addr in obs_set)
                push!(resolved_selection, addr)
            end
        end
    else
        for addr in selection
            if addr in obs_set
                throw(
                    ArgumentError(
                        "Address $(addr) cannot appear in both `selection` and `observations`.",
                    ),
                )
            end
            entry = _find_trace_entry(tt.entries, addr)
            if entry === nothing
                throw(
                    ArgumentError(
                        "Selected address $(addr) was not found in model sample sites."
                    ),
                )
            end
            push!(resolved_selection, addr)
        end
    end

    if isempty(resolved_selection)
        throw(ArgumentError("No parameter addresses selected for log-density evaluation."))
    end

    entries = filter_entries_by_selection(tt.entries, resolved_selection)
    dimension = 0
    for entry in entries
        dimension += entry.num_elements
    end

    sel_addrs = _collect_addresses(resolved_selection)
    all_constrained = vcat(sel_addrs, obs_addrs)
    obs_tensor = isempty(obs_addrs) ? nothing : to_rarray(reshape(obs_flat, 1, :))
    return resolved_selection, entries, dimension, all_constrained, obs_tensor
end

struct _TracedLogpObjective{F}
    f::F
    constrained_addresses::Vector{Address}
end

function (obj::_TracedLogpObjective)(θ, rng::AbstractRNG, obs_tensor, args::Tuple)
    θ_row = reshape(θ, 1, :)
    constraint_2d = obs_tensor === nothing ? θ_row : hcat(θ_row, reshape(obs_tensor, 1, :))
    _, weight, _ = generate(
        rng, constraint_2d, obj.f, args...; constrained_addresses=obj.constrained_addresses
    )
    return @allowscalar weight[]
end

struct _ModelLogpAndGradKernel{F}
    objective::_TracedLogpObjective{F}
end

function (k::_ModelLogpAndGradKernel)(θ, rng::AbstractRNG, obs_tensor, args::Tuple)
    grads, val = Enzyme.gradient(
        ReverseWithPrimal, Const(k.objective), θ, Const(rng), Const(obs_tensor), Const(args)
    )
    return val, grads[1]
end

"""
    logp(rng::AbstractRNG, θ, f::Function, args...; selection, observations_tensor=nothing, observation_addresses=Address[])
    logp(f::Function, θ, args...; selection=nothing, observations=Constraint(), constraints=nothing, rng=nothing)

Evaluate the joint log-density defined by a `ProbProg` model `f` at parameter vector `θ`.

Inside a `@compile optimize = :probprog` context, pass `rng` as the first argument to emit
the `impulse.generate` weight directly into the traced module. Outside `@compile`, calling
`logp(f, θ, args...; ...)` compiles and evaluates the model log-density at `θ`.
"""
function logp(
    rng::AbstractRNG,
    θ,
    f::Function,
    args::Vararg{Any,Nargs};
    selection::Union{AbstractSet{Address},AbstractVector{Address}},
    observations_tensor=nothing,
    observation_addresses::Union{AbstractSet{Address},AbstractVector{Address}}=Address[],
) where {Nargs}
    all_constrained = vcat(
        _collect_addresses(selection), _collect_addresses(observation_addresses)
    )
    obj = _TracedLogpObjective(f, all_constrained)
    return obj(θ, rng, observations_tensor, args)
end

"""
    logp_and_grad(rng::AbstractRNG, θ, f::Function, args...; selection, observations_tensor=nothing, observation_addresses=Address[])
    logp_and_grad(f::Function, θ, args...; selection=nothing, observations=Constraint(), constraints=nothing, rng=nothing)

Evaluate both the model log-density and its gradient with respect to `θ`, returning
`(logp, gradlogp)`.
"""
function logp_and_grad(
    rng::AbstractRNG,
    θ,
    f::Function,
    args::Vararg{Any,Nargs};
    selection::Union{AbstractSet{Address},AbstractVector{Address}},
    observations_tensor=nothing,
    observation_addresses::Union{AbstractSet{Address},AbstractVector{Address}}=Address[],
) where {Nargs}
    all_constrained = vcat(
        _collect_addresses(selection), _collect_addresses(observation_addresses)
    )
    kernel = _ModelLogpAndGradKernel(_TracedLogpObjective(f, all_constrained))
    return kernel(θ, rng, observations_tensor, args)
end

struct CompiledLogp{TThunk,TRng,TObs,TArgs}
    thunk::TThunk
    rng::TRng
    obs_tensor::TObs
    args::TArgs
    selection::Selection
    entries::Vector{TraceEntry}
    dimension::Int
end

function (c::CompiledLogp)(θ::AbstractConcreteArray)
    if length(θ) != c.dimension
        throw(
            DimensionMismatch(
                "Expected parameter array with $(c.dimension) elements, got $(length(θ))."
            ),
        )
    end
    θ_vec =
        (ndims(θ) == 1 && eltype(θ) === Float64) ? θ : to_rarray(Float64.(vec(Array(θ))))
    return c.thunk(θ_vec, c.rng, c.obs_tensor, c.args)
end

function (c::CompiledLogp)(θ::AbstractArray)
    if length(θ) != c.dimension
        throw(
            DimensionMismatch(
                "Expected parameter array with $(c.dimension) elements, got $(length(θ))."
            ),
        )
    end
    θ_dev = to_rarray(Float64.(vec(collect(θ))))
    val_dev = c.thunk(θ_dev, c.rng, c.obs_tensor, c.args)
    return Float64(to_number(val_dev))
end

struct CompiledLogpAndGrad{TThunk,TRng,TObs,TArgs}
    thunk::TThunk
    rng::TRng
    obs_tensor::TObs
    args::TArgs
    selection::Selection
    entries::Vector{TraceEntry}
    dimension::Int
end

function (c::CompiledLogpAndGrad)(θ::AbstractConcreteArray)
    if length(θ) != c.dimension
        throw(
            DimensionMismatch(
                "Expected parameter array with $(c.dimension) elements, got $(length(θ))."
            ),
        )
    end
    θ_vec =
        (ndims(θ) == 1 && eltype(θ) === Float64) ? θ : to_rarray(Float64.(vec(Array(θ))))
    val_dev, grad_dev = c.thunk(θ_vec, c.rng, c.obs_tensor, c.args)
    grad_shaped = ndims(θ) == 1 ? grad_dev : to_rarray(reshape(Array(grad_dev), size(θ)))
    return val_dev, grad_shaped
end

function (c::CompiledLogpAndGrad)(θ::AbstractArray)
    if length(θ) != c.dimension
        throw(
            DimensionMismatch(
                "Expected parameter array with $(c.dimension) elements, got $(length(θ))."
            ),
        )
    end
    θ_dev = to_rarray(Float64.(vec(collect(θ))))
    val_dev, grad_dev = c.thunk(θ_dev, c.rng, c.obs_tensor, c.args)
    grad_host = reshape(Array(grad_dev), size(θ))
    return Float64(to_number(val_dev)), grad_host
end

function (c::CompiledLogpAndGrad)(F, G, θ::AbstractArray)
    val, grad = c(θ)
    if G !== nothing
        copyto!(G, grad)
    end
    return F !== nothing ? val : nothing
end

"""
    compile_logp(f::Function, args...; selection=nothing, observations=Constraint(), constraints=nothing, rng=nothing)
    compile_logp(f::Function, observations::Constraint, args...; selection=nothing, rng=nothing)

Compile a reusable callable `g(θ) -> logp` evaluating the log-density of `ProbProg` model `f`
at an arbitrary parameter vector `θ` (either a Julia `AbstractArray` or a Reactant
`AbstractConcreteArray`). If `selection` is omitted, all sample addresses not present in
`observations` are treated as parameters in model execution order.
"""
function compile_logp(
    f::Function,
    args::Vararg{Any,Nargs};
    selection::Union{Nothing,Selection}=nothing,
    observations::Constraint=Constraint(),
    constraints::Union{Nothing,Constraint}=nothing,
    rng::Union{Nothing,AbstractRNG}=nothing,
) where {Nargs}
    obs = _resolve_observations(observations, constraints)
    rng_val = _default_probprog_rng(rng)
    resolved_selection, entries, dimension, all_constrained, obs_tensor = _prepare_logdensity_spec(
        f, rng_val, args, selection, obs
    )
    θ_proto = to_rarray(zeros(Float64, dimension))
    objective = _TracedLogpObjective(f, all_constrained)
    thunk = Compiler.compile(
        objective, (θ_proto, rng_val, obs_tensor, args); optimize=:probprog
    )
    return CompiledLogp(
        thunk, rng_val, obs_tensor, args, resolved_selection, entries, dimension
    )
end

function compile_logp(
    f::Function,
    observations::Constraint,
    args::Vararg{Any,Nargs};
    selection::Union{Nothing,Selection}=nothing,
    rng::Union{Nothing,AbstractRNG}=nothing,
) where {Nargs}
    return compile_logp(f, args...; selection=selection, observations=observations, rng=rng)
end

"""
    compile_logp_and_grad(f::Function, args...; selection=nothing, observations=Constraint(), constraints=nothing, rng=nothing)
    compile_logp_and_grad(f::Function, observations::Constraint, args...; selection=nothing, rng=nothing)

Compile a reusable callable `fg(θ) -> (logp, gradlogp)` (also supporting the 3-argument
`fg(F, G, θ)` `only_fg!` convention) for a `ProbProg` model `f`.
"""
function compile_logp_and_grad(
    f::Function,
    args::Vararg{Any,Nargs};
    selection::Union{Nothing,Selection}=nothing,
    observations::Constraint=Constraint(),
    constraints::Union{Nothing,Constraint}=nothing,
    rng::Union{Nothing,AbstractRNG}=nothing,
) where {Nargs}
    obs = _resolve_observations(observations, constraints)
    rng_val = _default_probprog_rng(rng)
    resolved_selection, entries, dimension, all_constrained, obs_tensor = _prepare_logdensity_spec(
        f, rng_val, args, selection, obs
    )
    θ_proto = to_rarray(zeros(Float64, dimension))
    kernel = _ModelLogpAndGradKernel(_TracedLogpObjective(f, all_constrained))
    thunk = Compiler.compile(
        kernel, (θ_proto, rng_val, obs_tensor, args); optimize=:probprog
    )
    return CompiledLogpAndGrad(
        thunk, rng_val, obs_tensor, args, resolved_selection, entries, dimension
    )
end

function compile_logp_and_grad(
    f::Function,
    observations::Constraint,
    args::Vararg{Any,Nargs};
    selection::Union{Nothing,Selection}=nothing,
    rng::Union{Nothing,AbstractRNG}=nothing,
) where {Nargs}
    return compile_logp_and_grad(
        f, args...; selection=selection, observations=observations, rng=rng
    )
end

function logp(
    f::Function,
    θ::AbstractArray,
    args::Vararg{Any,Nargs};
    selection::Union{Nothing,Selection}=nothing,
    observations::Constraint=Constraint(),
    constraints::Union{Nothing,Constraint}=nothing,
    rng::Union{Nothing,AbstractRNG}=nothing,
) where {Nargs}
    compiled = compile_logp(
        f,
        args...;
        selection=selection,
        observations=observations,
        constraints=constraints,
        rng=rng,
    )
    return compiled(θ)
end

function logp_and_grad(
    f::Function,
    θ::AbstractArray,
    args::Vararg{Any,Nargs};
    selection::Union{Nothing,Selection}=nothing,
    observations::Constraint=Constraint(),
    constraints::Union{Nothing,Constraint}=nothing,
    rng::Union{Nothing,AbstractRNG}=nothing,
) where {Nargs}
    compiled = compile_logp_and_grad(
        f,
        args...;
        selection=selection,
        observations=observations,
        constraints=constraints,
        rng=rng,
    )
    return compiled(θ)
end

using Base: ReentrantLock

mutable struct ProbProgTrace
    choices::Dict{Symbol,Any}
    retval::Any
    weight::Any
    subtraces::Dict{Symbol,Any}
    rng::Union{Nothing,AbstractRNG}
    fn::Union{Nothing,Function}
    args::Union{Nothing,Tuple}

    function ProbProgTrace()
        return new(
            Dict{Symbol,Any}(),
            nothing,
            nothing,
            Dict{Symbol,Any}(),
            nothing,
            nothing,
            nothing,
        )
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

const Selection = Set{Address}

const _probprog_ref_lock = ReentrantLock()
const _probprog_refs = IdDict()

function _keepalive!(tr::Any)
    lock(_probprog_ref_lock)
    try
        _probprog_refs[tr] = tr
    finally
        unlock(_probprog_ref_lock)
    end
    return tr
end

get_choices(trace::ProbProgTrace) = trace.choices
select(addrs::Address...) = Set{Address}([addrs...])
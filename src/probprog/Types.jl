using OrderedCollections: OrderedSet

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

get_choices(trace::Trace) = trace.choices

function select(addrs::Address...)
    sorted_addrs = sort(collect(addrs); by=a -> Tuple(string.(a.path)))
    return OrderedSet{Address}(sorted_addrs)
end

using OrderedCollections: OrderedCollections

struct OrderedIdDict{K,V} <: AbstractDict{K,V}
    keys::OrderedCollections.OrderedDict{UInt,K}
    values::OrderedCollections.OrderedDict{UInt,V}

    function OrderedIdDict{K,V}(pairs) where {K,V}
        return new(
            OrderedCollections.OrderedDict{UInt,K}(objectid(k) => k for (k, _) in pairs),
            OrderedCollections.OrderedDict{UInt,V}(objectid(k) => v for (k, v) in pairs),
        )
    end
end

OrderedIdDict() = OrderedIdDict{Any,Any}()
OrderedIdDict{K,V}() where {K,V} = OrderedIdDict{K,V}(Pair{K,V}[])

Base.show(io::IO, d::OrderedIdDict) = show(io, d.keys)

OrderedCollections.isordered(::OrderedIdDict) = true

Base.length(d::OrderedIdDict) = length(d.keys)
Base.isempty(d::OrderedIdDict) = isempty(d.keys)

function Base.getindex(d::OrderedIdDict, k)
    return d.values[objectid(k)]
end

function Base.setindex!(d::OrderedIdDict, v, k)
    d.keys[objectid(k)] = k
    d.values[objectid(k)] = v
    return d
end

function Base.haskey(d::OrderedIdDict, k)
    return haskey(d.keys, objectid(k))
end

function Base.delete!(d::OrderedIdDict, k)
    delete!(d.keys, objectid(k))
    delete!(d.values, objectid(k))
    return d
end

function Base.iterate(d::OrderedIdDict)
    k = iterate(d.keys)
    isnothing(k) && return nothing
    ((_, k), _) = k
    (_, v), state = iterate(d.values)
    return k => v, state
end

function Base.iterate(d::OrderedIdDict, state)
    k = iterate(d.keys, state)
    isnothing(k) && return nothing
    ((_, k), _) = k
    (_, v), state = iterate(d.values, state)
    return k => v, state
end

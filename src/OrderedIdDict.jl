using OrderedCollections: OrderedCollections

struct OrderedIdDict{K,V} <: AbstractDict{K,V}
    kv::OrderedCollections.OrderedDict{UInt,Pair{K,V}}

    function OrderedIdDict{K,V}(pairs) where {K,V}
        return new(
            OrderedCollections.OrderedDict{UInt,Pair{K,V}}(
                objectid(k) => (k => v) for (k, v) in pairs
            ),
        )
    end
end

OrderedIdDict() = OrderedIdDict{Any,Any}()
OrderedIdDict{K,V}() where {K,V} = OrderedIdDict{K,V}(Pair{K,V}[])

Base.show(io::IO, d::OrderedIdDict) = show(io, d.kv)

OrderedCollections.isordered(::OrderedIdDict) = true

Base.length(d::OrderedIdDict) = length(d.kv)
Base.isempty(d::OrderedIdDict) = isempty(d.kv)

function Base.getindex(d::OrderedIdDict, k)
    return last(d.kv[objectid(k)])
end

function Base.setindex!(d::OrderedIdDict, v, k)
    d.kv[objectid(k)] = k => v
    return d
end

function Base.haskey(d::OrderedIdDict, k)
    return haskey(d.kv, objectid(k))
end

function Base.delete!(d::OrderedIdDict, k)
    delete!(d.kv, objectid(k))
    return d
end

function Base.iterate(d::OrderedIdDict)
    r = iterate(d.kv)
    isnothing(r) && return nothing
    (_, kv), state = r
    return kv, state
end

function Base.iterate(d::OrderedIdDict, state)
    r = iterate(d.kv, state)
    isnothing(r) && return nothing
    (_, kv), state = r
    return kv, state
end

struct PersistentStack{T}
    data::T
    prev::Union{Nothing, PersistentStack{T}}
    length::Int
end

PersistentStack(x) = PersistentStack(x, nothing, 1)
PersistentStack{T}(x) where {T} = PersistentStack{T}(x, nothing, 1)

PersistentStack(stack::PersistentStack) = stack
PersistentStack{T}(stack::PersistentStack{T}) where {T} = stack

push(stack::PersistentStack, x) = PersistentStack(x, stack, length(stack) + 1)
pop(stack::PersistentStack) = stack.data

function Base.front(stack::PersistentStack{T}) where {T}
    !isnothing(stack.prev) ? stack.prev : PersistentStack{T}(pop(stack), nothing, 0)
end

Base.IteratorEltype(::Type{<:PersistentStack{T}}) where {T} = Base.HasEltype()
Base.eltype(::Type{<:PersistentStack{T}}) where {T} = T
Base.IteratorSize(::Type{<:PersistentStack}) = Base.HasLength()
Base.length(stack::PersistentStack) = stack.length

function Base.iterate(stack::PersistentStack, state = 1)
    state > length(stack) && return nothing
    return stack[state], state + 1
end

# fast shortcut for Base.Iterators.reverse
function Base.iterate(stack::Base.Iterators.Reverse{<:PersistentStack}, state = stack.itr)
    isempty(stack.itr) && return nothing
    return pop(state), Base.front(state)
end

Base.firstindex(::PersistentStack) = 1
Base.lastindex(stack::PersistentStack) = length(stack)

function Base.getindex(stack::PersistentStack, i::Int)
    if i < 1 || i > length(stack)
        throw(BoundsError(stack, i))
    end
    count = length(stack)
    while count > i
        stack = Base.front(stack)
        count -= 1
    end
    return pop(stack)
end

function Base.getindex(stack::PersistentStack{T}, r::UnitRange{Int}) where {T}
    isempty(r) && return PersistentStack{T}(pop(stack), nothing, 0)
    if first(r) < 1 || last(r) > length(stack)
        throw(BoundsError(stack, r))
    end

    res = PersistentStack{T}(stack[first(r)])
    for i in r[2:end]
        res = push(res, stack[i])
    end
    return res
end

function Base.hash(x::PersistentStack, h::UInt)
    isempty(x) && return hash((), h ⊻ 0x12345678)
    return hash((pop(x), Base.front(x)), h)
end

function Base.:(==)(x::PersistentStack, y::PersistentStack)
    length(x) != length(y) && return false
    isempty(x) && isempty(y) && return true
    pop(x) == pop(y) && Base.front(x) == Base.front(y)
end

function Base.collect(stack::PersistentStack{T}) where {T}
    n = length(stack)
    res = Vector{T}(undef, n)
    state = stack
    for i in 1:n
        res[n-i+1] = pop(state)
        state = state.prev
    end
    return res
end

# calling (stack...,) is bad [median=617.612 ns]. use this instead [median=382.267 ns].
Base.Tuple(stack::PersistentStack) = Tuple(collect(stack))

Base.show(io::IO, stack::PersistentStack) = print(io, Tuple(stack))

struct PersistentStack{T}
    data::T
    prev::Union{Nothing, PersistentStack{T}}
    length::Int
end

PersistentStack(x) = PersistentStack(x, nothing, 1)
PersistentStack{T}(x) where {T} = PersistentStack{T}(x, nothing, 1)

push(stack::PersistentStack, x) = PersistentStack(x, stack, length(stack) + 1)
pop(stack::PersistentStack) = stack.data
Base.front(stack::PersistentStack) = stack.prev

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

function Base.getindex(stack::PersistentStack, i)
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
Base.firstindex(::PersistentStack) = 1
Base.lastindex(stack::PersistentStack) = length(stack)

function Base.collect(stack::PersistentStack{T}) where {T}
    res = T[pop(stack)]
    state = Base.front(stack)
    while !isnothing(state)
        push!(res, pop(state))
        state = Base.front(state)
    end
    reverse!(res)
    return res
end

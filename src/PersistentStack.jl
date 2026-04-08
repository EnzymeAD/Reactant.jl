struct PersistentStack{T}
    data::T
    prev::Union{Nothing, PersistentStack{T}}
end

PersistentStack(x) = PersistentStack(x, nothing)
PersistentStack{T}(x) where {T} = PersistentStack{T}(x, nothing)

push(stack::PersistentStack, x) = PersistentStack(x, stack)
pop(stack::PersistentStack) = stack.data
Base.front(stack::PersistentStack) = stack.prev

function Base.collect(stack::PersistentStack{T}) where {T}
    res = T[pop(stack)]
    state = front(stack)
    while !isnothing(state)
        push!(res, pop(state))
        state = front(state)
    end
    reverse!(res)
    return res
end

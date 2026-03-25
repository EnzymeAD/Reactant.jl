using Reactant, Test

function buffer_equals(x, y)
    if x isa Reactant.ConcreteIFRTArray
        y.data.buffer == x.data.buffer
    elseif x isa Reactant.ConcretePJRTArray
        all(((x, y),) -> x == y, zip(x.data, y.data))
    else
        error("invalid array type $(typeof(x))")
    end
end

function copy_with_broadcast!(a, b)
    a .= b
    return nothing
end

mutable struct X{T}
    x::T
    y::T
end

function br_func!(x, z)
    x.x .= z
    x.y = x.x
    return nothing
end

function f!(x, j)
    x.x .= 2.0 .* x.x
    x.y = x.x
    j .= x.y
    return nothing
end

@testset "Buffer aliasing" begin
    x = Reactant.to_rarray(ones(10))
    y = similar(x)
    @jit copy_with_broadcast!(y, x)
    @test !buffer_equals(x, y)

    x = Reactant.to_rarray(ones(10))
    y = Reactant.to_rarray(ones(10))
    x = X(x, y)
    z = Reactant.to_rarray(ones(10))
    @jit br_func!(x, z)
    @test buffer_equals(x.x, x.y)
    @test !buffer_equals(x.x, z)

    x = Reactant.to_rarray(ones(10))
    y = Reactant.to_rarray(ones(10))
    z = Reactant.to_rarray(ones(10))
    x = X(x, y)
    @jit f!(x, z)
    @test buffer_equals(x.x, x.y)
    @test !buffer_equals(x.y, z)
end

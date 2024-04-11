using Reactant


fastmax(x::AbstractArray{T}) where T = reduce(max, x; dims=1, init = float(T)(-Inf))

if true

    @show fastmax(ones(2, 10)), size(fastmax(ones(2, 10)))

    a = Reactant.ConcreteRArray(ones(2, 10))

    @show fastmax(a), size(fastmax(a))

    f=Reactant.compile(fastmax, (a,))
    
    @show f(a), size(f(a))
end

if true
    function softmax!(x)
        max_ = fastmax(x)
        return x .- max_
    end

    in = ones(2, 10)
    @show softmax!(in)

    in = Reactant.ConcreteRArray(ones(2, 10))

    f=Reactant.compile(softmax!, (in,))
    @show f(in)
end




c = Reactant.ConcreteRArray(ones(3,2))

f=Reactant.compile(cos, (c,))

@show f(c)
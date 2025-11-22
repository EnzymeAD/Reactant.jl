macro stop(n::Int)
    u = :counter #gensym()
    e = esc(u)
    quote
        isdefined(@__MODULE__, $(QuoteNode(u))) || global $e = $n
        global $e
        $e<2 && error("stop")
        $e -= 1
    end
end



#leak each argument to a global variable and store each instance of it
macro lks(args...)
    nargs = [ Symbol(string(arg) * "s")  for arg in args]
    quote
        $([:(
            let val = $(esc(p))
                isdefined(@__MODULE__, $(QuoteNode(n))) || global $(esc(n)) = []
                global $(esc(n))
                push!($(esc(n)), val)
            end
        ) for (p,n) in zip(args, nargs)]...)
    end
end
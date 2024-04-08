module Reactant

include("mlir/MLIR.jl")

abstract type RArray{ElType,Shape,N} <: AbstractArray{ElType, N} end

@inline Base.eltype(::RArray{ElType,Shape}) where {ElType, Shape} = ElType
@inline shape(::RArray{ElType,Shape}) where {ElType, Shape} = Shape
@inline dim(::RArray{ElType,Shape, N}) where {ElType, Shape, N} = N

@inline dim(::Array{ElType, N}) where {ElType, N} = N

struct XLAArray{ElType,Shape,N} <: RArray{ElType, Shape, N}
end

mutable struct ConcreteRArray{ElType,Shape,N} <: RArray{ElType, Shape, N}
	data::XLAArray{ElType, Shape, N}
end

mutable struct TracedRArray{ElType,Shape,N} <: RArray{ElType, Shape, N}
	paths::Tuple{<:Tuple}
	mlir_data::Union{Nothing,Any}
end

using Enzyme

@inline val_value(::Val{T}) where T = T

@inline function traced_type(::Type{T}, seen::ST, ::Val{to_traced}) where {ST,T, to_traced}
	if T <: ConcreteRArray
		if to_traced
			return TracedRArray{eltype(T), shape(T), dim(T)}
		else
			throw("Abstract RArray cannot be made concrete")
		end
	end
	if T <: TracedRArray
		if to_traced
			throw("TracedRArray $T cannot be traced")
		else
			return ConcreteRArray{eltype(T), shape(T), dim(T)}
		end
	end

	if T <: XLAArray
		throw("XLA $T array cannot be traced")
	end
	if T <: RArray
		return T
	end


    if T === Any
        return T
    end

    if T === Symbol
        return T
    end
    
    if T <: Val
    	val = val_value(T)
    	if traced_type(typeof(val), seen, Val(to_traced)) == typeof(val)
    		return T
    	end
		throw("Val type $T cannot be traced")
    end

    if T === Union{}
        return T
    end

    if T == Nothing
        return T
    end

    if T == Char
        return T
    end

    if T <: Complex && !(T isa UnionAll)
        return Complex{traced_type(Enzyme.Compiler.ptreltype(T), seen, Val(to_traced))}
    end

    if T <: AbstractFloat
        return T
    end

    if T <: Ptr
    	return Ptr{traced_type(Enzyme.Compiler.ptreltype(T), seen, Val(to_traced))}
    end

    if T <: Core.LLVMPtr
    	return Core.LLVMPtr{traced_type(Enzyme.Compiler.ptreltype(T), seen, Val(to_traced))}
    end

    if T <: Base.RefValue
    	return Base.RefValue{traced_type(Enzyme.Compiler.ptreltype(T), seen, Val(to_traced))}
    end

    if T <: Array
    	return Array{traced_type(Enzyme.Compiler.ptreltype(T), seen, Val(to_traced)), dim(T)}
    end

    if T <: Integer
        return T
    end

    if Enzyme.Compiler.isghostty(T) || Core.Compiler.isconstType(T)
        return T
    end

    if T <: Function
        return T
    end

    if T <: DataType
        return T
    end
    if T <: Module
        return T
    end
    if T <: AbstractString
        return T
    end

    # unknown number of fields
    if T isa UnionAll
        aT = Base.argument_datatype(T)
        if aT === nothing
        	throw("Unhandled type $T")
        end
        if datatype_fieldcount(aT) === nothing
        	throw("Unhandled type $T")
        end
    end

    if T isa Union
    	return Union(traced_type(seen, T.a, Val(to_traced)), traced_type(seen, T.b, Val(to_traced)))
    end

    # if abstract it must be by reference
    if Base.isabstracttype(T)
    	throw("Unhandled abstract type $T")
    end

    @inline is_concrete_tuple(x::T2) where T2 = (x <: Tuple) && !(x === Tuple) && !(x isa UnionAll)

    @assert !Base.isabstracttype(T)

    if !(Base.isconcretetype(T) || is_concrete_tuple(T) || T isa UnionAll)
        throw(AssertionError("Type $T is not concrete type or concrete tuple"))
    end

    if Enzyme.Compiler.is_concrete_tuple(T) && any(T2 isa Core.TypeofVararg for T2 in T.parameters)
        Tuple{((T2 isa Core.TypeofVararg ? Any : T2) for T2 in T.parameters)...,}
        throw(AssertionError("Type tuple of vararg $T is not supported"))
    end

    if Enzyme.Compiler.is_concrete_tuple(T)
    	return Tuple{traced_type(seen, T2, Val(to_traced)) for T2 in T.parameters}
    end

    if T <: NamedTuple
    	@inline tup_name(::Type{NamedTuple{A, B}}) where {A, B} = A
    	@inline tup_val(::Type{NamedTuple{A, B}}) where {A, B} = B
    	return NamedTuple{tup_name(T), traced_type(seen, tup_val(T), Val(to_traced))}
    end

    if T <: Dict
    	@inline dict_name(::Type{Dict{A, B}}) where {A, B} = A
    	@inline dict_val(::Type{Dict{A, B}}) where {A, B} = B
    	return Dict{dict_name(T), traced_type(seen, dict_val(T), Val(to_traced))}
    end

    if T <: IdDict
    	@inline iddict_name(::Type{IdDict{A, B}}) where {A, B} = A
    	@inline iddict_val(::Type{IdDict{A, B}}) where {A, B} = B
    	return IdDict{iddict_name(T), traced_type(seen, iddict_val(T), Val(to_traced))}
    end

    if Val(T) ∈ seen
        return seen[T]
    end

    seen = (Val(T), seen...)

    changed = false
    subTys = Type[]
    for f in 1:fieldcount(T)
        subT = fieldtype(T, f)
        subTT = traced_type(seen, subT, Val(to_traced))
        changed |= subT != subTT
        push!(subTys, subT)
    end

    if !changed
    	return T
    end

    subParms = []
    for SST in T.parameters
    	if SST <: Type
    		push!(subParms, traced_type(seen, SST, Val(to_traced)))
    	else
    		push!(subParms, SST)
    	end
    end

    TT2 = Core.apply_type(T.name.wrapper, subParms...)
    if fieldcount(T) == fieldcount(TT2)
	    subTys = Type[]
	    legal = true
	    for f in 1:fieldcount(T)
	        subT = fieldtype(T, f)
	        subT2 = fieldtype(TT2, f)
	        subTT = traced_type(seen, subT, Val(to_traced))
	        legal &= subT2 == subTT
	    end
	    if legal
	    	return TT2
	    end
	end

	name = Symbol[]

	return NamedTuple{fieldnames(T), Tuple{subTys...}}
end

function append_path(path, i)
	(path..., i)
end

@enum TraceMode begin
   ConcreteToTraced = 1
   TracedTrack = 2
   TracedToConcrete = 3
end

@inline function make_tracer(seen::IdDict, prev::ConcreteRArray{ElType, Shape, N}, path, mode, data) where {ElType, Shape, N}
	if mode != ConcreteToTraced
		throw("Cannot trace concrete")
	end
    if haskey(seen, prev)
        return seen[prev]::TracedRArray{ElType, Shape, N}
    end
    res = TracedRArray{ElType, Shape, N}((path,), nothing)
    seen[prev] = res
    return res
end

@inline function make_tracer(seen::IdDict, prev::TracedRArray{ElType, Shape, N}, path, mode, data) where {ElType, Shape, N}
	if mode == ConcreteToTraced
		throw("Cannot trace existing trace type")
	end
	if mode == TracedTrack
		prev.paths = (prev.paths..., path)
	    if !haskey(seen, prev)
	        return seen[prev] = prev
	    end
	    return prev
	end

	if mode == TracedToConcrete
	    if haskey(seen, prev)
	        return seen[prev]::ConcreteRArray{ElType, Shape, N}
	    end
	    res = ConcreteRArray{ElType, Shape, N}(nothing)
	    seen[prev] = res
	    return res	    
	end

	throw("Cannot Unknown trace mode")
end

@inline function make_tracer(seen::IdDict, prev::RT, path, mode, data) where {RT<:AbstractFloat}
    return prev
end

@inline function make_tracer(seen::IdDict, prev::Complex{RT}, path, mode, data) where {RT}
    return Complex(make_tracer(seen, prev.re, append_path(path, :re), mode, data), make_tracer(seen, prev.im, append_path(path, :im), mode, data))
end

@inline function make_tracer(seen::IdDict, prev::RT, path, mode, data) where {RT<:Array}
    if haskey(seen, prev)
        return seen[prev]
    end
    TT = traced_type((), eltype(RT), Val(to_traced))
    newa = Array{TT, dim(RT)}(undef, size(prev))
    seen[prev] = newa
    same = true
    for I in eachindex(prev)
        if isassigned(prev, I)
            pv = prev[I]
            nv = make_tracer(seen, pv, append_path(path, I), mode, data)
            if pv !== nv
            	same = false
            end
            @inbounds newa[I] = nv
        end
    end
    if same
    	seen[prev] = prev
    	return prev
    end
    return newa
end

@inline function make_tracer(seen::IdDict, prev::RT, path, mode, data) where {RT<:Tuple}
    return ((make_tracer(seen, v, append_path(path, i), mode, data) for (i, v) in enumerate(prev))...,)
end

@inline function make_tracer(seen::IdDict, prev::NamedTuple{A,RT}, path, mode, data) where {A, RT}
    return NamedTuple{A, traced_type((), RT, Val(to_traced))}(
	    ((make_tracer(seen, getfield(prev, name), append_path(path, name), mode, data) for name in A)...,)
    )
end

@inline function make_tracer(seen::IdDict, prev::Core.Box, path, mode, data)
    if haskey(seen, prev)
        return seen[prev]
    end
    prev2 = prev.contents
    tr = make_tracer(seen, prev2, append_path(path, :contents), mode, data)
    if tr == prev2
	    seen[prev] = prev
    	return prev
    end
    res = Core.Box(tr)
    seen[prev] = res
    return res
end

@inline function make_tracer(seen::IdDict, prev::RT, path, mode, data)::RT where {RT}
    if haskey(seen, prev)
        return seen[prev]
    end
    TT = traced_type((), RT, Val(to_traced))
    @assert !Base.isabstracttype(RT)
    @assert Base.isconcretetype(RT)
    nf = fieldcount(RT)
    
    if ismutable(TT)
        y = ccall(:jl_new_struct_uninit, Any, (Any,), TT)
        seen[prev] = y
        changed = false
        for i in 1:nf
            if isdefined(prev, i)
                xi = getfield(prev, i)
                xi2 = make_tracer(seen, xi, append_path(path, i), mode, data)
                if xi !== xi2
                	changed = true
                end
                ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), y, i-1, xi2)
            end
        end
        if !changed
        	seen[prev] = prev
        	return prev
        end
        return y
    end
    
    if nf == 0
        return prev
    end

    flds = Vector{Any}(undef, nf)
    changed = false
    for i in 1:nf
        if isdefined(prev, i)
            xi = getfield(prev, i)
            xi2 = make_tracer(seen, xi, append_path(path, i), mode, data)
            if xi !== xi2
            	changed = true
            end
            flds[i] = xi2
        else
            nf = i - 1 # rest of tail must be undefined values
            break
        end
    end    
    if !changed
    	seen[prev] = prev
    	return prev
    end
    y = ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), RT, flds, nf)
    seen[prev] = y
    return y
end

function generate_jlfunc(concrete_result, fptr, Nargs, linear_args, linear_results, preserved_args)
	args = ntuple(Val(Nargs)) do i
		Base.@_inline_meta
		Symbol("arg_$i")
	end

	linearized_args = Union{Symbol,Expr}[]

	for arg in linear_args
		res = Symbol("arg_$(arg.path[2])")
		for p in only(arg.path)[3:end]
			res = :(getfield($res, $p))
		end
		push!(linearized_args, res)
	end

	concretize = Expr[]
	for (idx, res) in linear_results
		push!(concretize, :(
			concrete_res_$idx = $(ConcreteRArray{eltype(res),shape(res),len(shape(res))})(linearized_results[$idx])
		))
	end

	delinearized_results = Expr[]

	for (idx, result) in enumerate(linear_results)
		for path in result.path
			if path[1] == "result"
				res = Symbol("result")
				path = path[2:end]
			else
				@assert path[1] == "resarg"
				res = Symbol("arg_$(arg.path[2])")
				path = path[3:end]
			end
			for p in path
				res = :(getfield($res, $p))
			end
			res = :($res.data = concrete_res_$idx )
			push!(delinearized_results, res)
		end
	end

	for (result, arg_idx) in preserved_args
		for path in result.path
			arg = linear_args[arg_idx]
			argpath = only(argres.path)

			if path[1] == "result"
				res = Symbol("result")
				path = path[2:end]
			else
				@assert path[1] == "resarg"
				# We can optimize cases where we set the arg to itself
				if path[2:end] == path[2:end]
					continue
				end
				res = Symbol("arg_$(arg.path[2])")
				path = path[3:end]
			end
			for p in path
				res = :(getfield($res, $p))
			end

			argres = Symbol("arg_$(argpath[2])")
			for p in argpath[3:end]
				argres = :(getfield($argres, $p))
			end

			res = :($res.data = $argres.data )
			push!(delinearized_results, res)
		end
	end

    for i in 1:N
        if base === nothing
            prim = Symbol("primal_$i")
            t = Symbol("PT_$i")
            e = :($prim::$t)
            push!(allargs, e)
            push!(typeargs, t)
        else
            prim = :($base[$base_idx])
            base_idx += 1
        end

	func = quote
		function compiled_f($(args...))
			linearized_results = $fptr($(linearized_args...))
			$concretize
			result = $traced_result
			$delinearized_results
			return result
		end
	end
	@show func
	return eval(func)
end

function compile(f::FTy, args=VarArg{N, Any}; options="") where {FTy, N}
	seen_args = IdDict()
	traced_args = ntuple(Val(N)) do i
		Base.@_inline_meta
		make_tracer(seen_args, args[i], ("args", i,), ConcreteToTraced, #=data=#nothing)
	end

	linear_args = TracedRArray[]
	for (k, v) in seen_args
		if !(v <: TracedRArray)
			continue
		end

		# todo create arg
		v.mlir_arg = linear_args.size()
		push!(linear_args, v)
	end

	result = f(traced_args...)

	seen_results = IdDict()

	traced_result = make_tracer(seen_results, result, ("result",), TracedTrack, #=data=#nothing)

	retraced_args = ntuple(Val(N)) do i
		Base.@_inline_meta
		make_tracer(seen_results, traced_args[i], ("resargs", i,), TracedTrack, #=data=#nothing)
	end


	linear_results = TracedRArray[]

	preserved_args = Tuple{TracedRArray, Int}[]
	for (k, v) in seen_results
		if !(v <: TracedRArray)
			continue
		end

		if v.mlir_arg isa MLIRArgument
			push!(preserved_args, (v, index(v.mlir_arg)))
			continue
		end

		push!(linear_results, v)
	end

	# donated = linear_args - preserved_args

	concrete_seen = IdDict()

	concrete_result = make_tracer(concrete_seen, traced_result, ("result",), TracedToConcrete, #=data=#nothing)

	return generate_jlfunc(concrete_result, fptr, N, linear_args, linear_results, preserved_args)
end



end

end # module
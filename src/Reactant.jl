module Reactant

include("mlir/MLIR.jl")
include("XLA.jl")

abstract type RArray{ElType,Shape,N} <: AbstractArray{ElType, N} end

@inline Base.eltype(::RArray{ElType,Shape}) where {ElType, Shape} = ElType
@inline Base.size(::RArray{ElType,Shape}) where {ElType, Shape} = Shape
@inline dim(::RArray{ElType,Shape, N}) where {ElType, Shape, N} = N

@inline mlir_type(::RArray{ElType,Shape,N}) where {ElType, Shape, N} = MLIR.IR.TensorType(Shape, MLIR.IR.Type(ElType))

@inline dim(::Array{ElType, N}) where {ElType, N} = N
@inline dim(::Type{Array{ElType, N}}) where {ElType, N} = N

struct XLAArray{ElType,Shape,N} <: RArray{ElType, Shape, N}
end

mutable struct ConcreteRArray{ElType,Shape,N} <: RArray{ElType, Shape, N}
	data::XLA.AsyncBuffer
#	data::XLAArray{ElType, Shape, N}
end

function Base.convert(::Type{T}, X::ConcreteRArray{ElType, Shape, N}) where {T<:Array, ElType, Shape, N}
	data = Array{ElType, N}(undef, Shape...)
	XLA.await(X.data)
	buf = X.data.buffer
	GC.@preserve data buf begin
		XLA.BufferToHost(buf, pointer(data))
	end
	return data
end

function Base.print_array(io::IO, X::ConcreteRArray)
	if X.data == XLA.AsyncEmptyBuffer
		println(io, "<Empty buffer>")
		return
	end
	Base.print_array(io, convert(Array, X))
end

function Base.show(io::IO, X::ConcreteRArray)
	if X.data == XLA.AsyncEmptyBuffer
		println(io, "<Empty buffer>")
		return
	end
	Base.show(io, convert(Array, X))
end


@inline function Base.getindex(a::ConcreteRArray{ElType, Shape}, args::Vararg{Int, N}) where {ElType, Shape, N}
	if a.data == XLA.AsyncEmptyBuffer
		throw("Cannot getindex from empty buffer")
	end
	XLA.await(a.data)
	if XLA.BufferOnCPU(a.data.buffer)
		buf = a.data.buffer
		GC.@preserve buf begin
			ptr = Base.unsafe_convert(Ptr{ElType}, XLA.UnsafeBufferPointer(buf))
			start = 0
			for i in 1:N
				start *= Shape[N-i+1]
				start += (args[N-i+1]-1)
			end
			start += 1
			return unsafe_load(ptr, start)
		end
	end
	return convert(Array, X)[args...]
end

@inline function ConcreteRArray(data::Array{ElType, N}; client=XLA.default_backend[], idx=XLA.default_device_idx[]) where {ElType, N}
	device = XLA.ClientGetDevice(client, idx)
	ConcreteRArray{ElType, size(data), N}(XLA.AsyncBuffer(XLA.ArrayFromHostBuffer(client, data, device), nothing))
end

@inline ConcreteRArray(data::T) where {T <: Number} = ConcreteRArray{T, (), 0}(data)

mutable struct TracedRArray{ElType,Shape,N} <: RArray{ElType, Shape, N}
	paths::Tuple
	mlir_data::Union{Nothing,MLIR.IR.Value}
end

include("overloads.jl")

using Enzyme

@inline val_value(::Val{T}) where T = T

@enum TraceMode begin
	ConcreteToTraced = 1
	TracedTrack = 2
	TracedToConcrete = 3
	ArrayToConcrete = 4
 end

@inline is_concrete_tuple(x::T2) where T2 = (x <: Tuple) && !(x === Tuple) && !(x isa UnionAll)
@inline function traced_type(::Type{T}, seen::ST, ::Val{mode}) where {ST,T, mode}
	if T <: ConcreteRArray
		if mode == ConcreteToTraced
			return TracedRArray{eltype(T), size(T), dim(T)}
		elseif mode == TracedToConcrete
			return T
		else
			throw("Abstract RArray cannot be made concrete")
		end
	end
	if T <: TracedRArray
		if mode == ConcreteToTraced
			throw("TracedRArray $T cannot be traced")
		elseif mode == TracedToConcrete
			return ConcreteRArray{eltype(T), size(T), dim(T)}
		else
			throw("Abstract RArray cannot be made concrete")
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
    	if traced_type(typeof(val), seen, Val(mode)) == typeof(val)
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
        return Complex{traced_type(Enzyme.Compiler.ptreltype(T), seen, Val(mode))}
    end

    if T <: AbstractFloat
        return T
    end

    if T <: Ptr
    	return Ptr{traced_type(Enzyme.Compiler.ptreltype(T), seen, Val(mode))}
    end

    if T <: Core.LLVMPtr
    	return Core.LLVMPtr{traced_type(Enzyme.Compiler.ptreltype(T), seen, Val(mode))}
    end

    if T <: Base.RefValue
    	return Base.RefValue{traced_type(Enzyme.Compiler.ptreltype(T), seen, Val(mode))}
    end

    if T <: Array
		if mode == ArrayToConcrete && eltype(T) <: AbstractFloat
			return (ConcreteRArray{eltype(T), Shape, dim(T)} where Shape)
		else
	    	return Array{traced_type(Enzyme.Compiler.ptreltype(T), seen, Val(mode)), dim(T)}
		end
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
    	return Union{traced_type(T.a, seen, Val(mode)), traced_type(T.b, seen, Val(mode))}
    end

    # if abstract it must be by reference
    if Base.isabstracttype(T)
    	throw("Unhandled abstract type $T")
    end

    @assert !Base.isabstracttype(T)

    if !(Base.isconcretetype(T) || is_concrete_tuple(T) || T isa UnionAll)
        throw(AssertionError("Type $T is not concrete type or concrete tuple"))
    end

    if is_concrete_tuple(T) && any(T2 isa Core.TypeofVararg for T2 in T.parameters)
        Tuple{((T2 isa Core.TypeofVararg ? Any : T2) for T2 in T.parameters)...,}
        throw(AssertionError("Type tuple of vararg $T is not supported"))
    end

    if is_concrete_tuple(T)
    	return Tuple{(traced_type(T2, seen, Val(mode)) for T2 in T.parameters)...}
    end

    if T <: NamedTuple
    	@inline tup_name(::Type{NamedTuple{A, B}}) where {A, B} = A
    	@inline tup_val(::Type{NamedTuple{A, B}}) where {A, B} = B
    	return NamedTuple{tup_name(T), traced_type(tup_val(T), seen, Val(mode))}
    end

    if T <: Dict
    	@inline dict_name(::Type{Dict{A, B}}) where {A, B} = A
    	@inline dict_val(::Type{Dict{A, B}}) where {A, B} = B
    	return Dict{dict_name(T), traced_type(dict_val(T), seen, Val(mode))}
    end

    if T <: IdDict
    	@inline iddict_name(::Type{IdDict{A, B}}) where {A, B} = A
    	@inline iddict_val(::Type{IdDict{A, B}}) where {A, B} = B
    	return IdDict{iddict_name(T), traced_type(iddict_val(T), seen, Val(mode))}
    end

    if Val(T) ∈ seen
        return seen[T]
    end

    seen = (Val(T), seen...)

    changed = false
    subTys = Type[]
    for f in 1:fieldcount(T)
        subT = fieldtype(T, f)
        subTT = traced_type(subT, seen, Val(mode))
        changed |= subT != subTT
        push!(subTys, subTT)
    end

    if !changed
    	return T
    end

    subParms = []
    for SST in T.parameters
    	if SST isa Type
			TrT = traced_type(SST, seen, Val(mode))
    		push!(subParms, TrT)
    	else
    		push!(subParms, SST)
    	end
    end

    TT2 = Core.apply_type(T.name.wrapper, subParms...)
    if fieldcount(T) == fieldcount(TT2)
	    legal = true
	    for f in 1:fieldcount(T)
	        subT = fieldtype(T, f)
	        subT2 = fieldtype(TT2, f)
	        subTT = traced_type(subT, seen, Val(mode))
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

@inline function make_tracer(seen::IdDict, prev::RT, path, mode, data) where {RT}
    if haskey(seen, prev)
        return seen[prev]
    end
    TT = traced_type(RT, (), Val(mode))
    @assert !Base.isabstracttype(RT)
    @assert Base.isconcretetype(RT)
    nf = fieldcount(RT)

	if TT <: NamedTuple
        changed = false
		subs = []
        for i in 1:nf
			xi = getfield(prev, i)
			xi2 = make_tracer(seen, xi, append_path(path, i), mode, data)
			if xi !== xi2
				changed = true
			end
			push!(subs, xi2)
        end
		if !changed
			seen[prev] = prev
			return prev
		end
		tup = (subs...,)
		@show TT, subs, tup
		return NamedTuple{TT.parameters[1], typeof(tup)}(tup)
	end
    
    if ismutabletype(TT)
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
    y = ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), TT, flds, nf)
    seen[prev] = y
    return y
end

@inline function make_tracer(seen::IdDict, prev::ConcreteRArray{ElType, Shape, N}, path, mode, data) where {ElType, Shape, N}
	if mode == ArrayToConcrete
		return prev
	end
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
	    res = ConcreteRArray{ElType, Shape, N}(XLA.AsyncEmptyBuffer)
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
	if mode == ArrayToConcrete && eltype(RT) <: AbstractFloat
		return seen[prev] = ConcreteRArray(prev)
	end
    TT = traced_type(eltype(RT), (), Val(mode))
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
    return NamedTuple{A, traced_type(RT, (), Val(mode))}(
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

function generate_jlfunc(concrete_result, client, mod, Nargs, linear_args, linear_results, preserved_args)
	args = ntuple(Val(Nargs)) do i
		Base.@_inline_meta
		Symbol("arg_$i")
	end

	linearized_args = Union{Symbol,Expr}[]

	for arg in linear_args
		path = if length(arg.paths) == 1
			arg.paths[1]
		elseif length(arg.paths) == 2
			@assert arg.paths[1][2:end] == arg.paths[2][2:end]
			@assert (
				(arg.paths[1][1] == "resargs" && arg.paths[2][1] == "args") ||
				(arg.paths[1][1] == "args" && arg.paths[2][1] == "resargs")
			)
			arg.paths[2]
		else
			throw("Invalid path duplication")
		end
		res = Symbol("arg_$(path[2])")
		for p in path[3:end]
			res = :(getfield($res, $p))
		end
		res = :(XLA.synced_buffer($res.data).buffer)
		push!(linearized_args, res)
	end

	concretize = Expr[]
	for (idx, res) in enumerate(linear_results)
		push!(concretize, :(
			$(Symbol("concrete_res_$(idx)")) = linearized_results[$idx]
		))
	end

	delinearized_results = Expr[]

	for (idx, result) in enumerate(linear_results)
		for path in result.paths
			if path[1] == "result"
				res = Symbol("result")
				path = path[2:end]
			else
				@assert path[1] == "resarg"
				res = Symbol("arg_$(path[2])")
				path = path[3:end]
			end
			for p in path
				res = :(getfield($res, $p))
			end
			res = :($res.data = $(Symbol("concrete_res_$(idx)")) )
			push!(delinearized_results, res)
		end
	end

	for (result, arg_idx) in preserved_args
		for path in result.paths
			arg = linear_args[arg_idx]
			argpath = if length(arg.paths) == 1
				arg.paths[1]
			elseif length(arg.paths) == 2
				@assert arg.paths[1][2:end] == arg.paths[2][2:end]
				@assert (
					(arg.paths[1][1] == "resargs" && arg.paths[2][1] == "args") ||
					(arg.paths[1][1] == "args" && arg.paths[2][1] == "resargs")
				)
				arg.paths[2]
			else
				throw("Invalid path duplication")
			end

			if path[1] == "result"
				res = Symbol("result")
				path = path[2:end]
			else
				@assert path[1] == "resargs" || path[1] == "args"
				# We can optimize cases where we set the arg to itself
				if path[2:end] == argpath[2:end]
					continue
				end
				res = Symbol("arg_$(path[2])")
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

	exec = XLA.Compile(client, mod)


    donated_args_set = zeros(UInt8, length(linearized_args))
	for (i, val) in enumerate(linear_args)
		if !in(val, preserved_args)
			donated_args_set[i] = 1
		end
	end

	func = quote
		function compiled_f($(args...))
			linearized_results = XLA.ExecutableCall($exec, [$(linearized_args...)], $donated_args_set,  Val($(length(linear_results))))
			$(concretize...)
			result = $concrete_result
			$(delinearized_results...)
			return result
		end
	end
	return eval(func)
end

const registry = Ref{MLIR.IR.DialectRegistry}()
function __init__()
	registry[] = MLIR.IR.DialectRegistry()
	@ccall MLIR.API.mlir_c.InitializeRegistryAndPasses(registry[]::MLIR.API.MlirDialectRegistry)::Cvoid
end

function compile(f::FTy, args::VAT; pipeline_options="", client=nothing) where {FTy, VAT <: Tuple}
	N = length(args)
	ctx = MLIR.IR.Context()
	Base.append!(registry[], context=ctx)
	@ccall MLIR.API.mlir_c.RegisterDialects(ctx::MLIR.API.MlirContext)::Cvoid
	MLIR.IR.context!(ctx) do
		seen_args = IdDict()
		traced_args = ntuple(Val(N)) do i
			Base.@_inline_meta
			make_tracer(seen_args, args[i], ("args", i,), ConcreteToTraced, #=data=#nothing)
		end

		linear_args = TracedRArray[]
		for (k, v) in seen_args
			if !(v isa TracedRArray)
				continue
			end
			push!(linear_args, v)
		end

		mod = MLIR.IR.Module(MLIR.IR.Location())
		modbody = MLIR.IR.body(mod)

		in_tys = [mlir_type(arg) for arg in linear_args]
		
		func = MLIR.Dialects.func.func_(; sym_name="main_tmp", function_type=MLIR.IR.FunctionType(in_tys, []), body=MLIR.IR.Region())

		fnbody = MLIR.IR.Block(in_tys, [MLIR.IR.Location() for arg in linear_args])
		push!(MLIR.IR.region(func, 1), fnbody)

		for (i, arg) in enumerate(linear_args)
			arg.mlir_data = MLIR.IR.argument(fnbody, i)
		end

		result = MLIR.IR.block!(fnbody) do
			f(traced_args...)
		end

		seen_results = IdDict()

		traced_result = make_tracer(seen_results, result, ("result",), TracedTrack, #=data=#nothing)

		retraced_args = ntuple(Val(N)) do i
			Base.@_inline_meta
			make_tracer(seen_results, traced_args[i], ("resargs", i,), TracedTrack, #=data=#nothing)
		end

		linear_results = TracedRArray[]

		preserved_args = Tuple{TracedRArray, Int}[]
		for (k, v) in seen_results
			if !(v isa TracedRArray)
				continue
			end

			if MLIR.IR.is_block_arg(v.mlir_data)
				push!(preserved_args, (v, 1+MLIR.IR.block_arg_num(v.mlir_data)))
				continue
			end

			push!(linear_results, v)
		end

		out_tys = [mlir_type(arg) for arg in linear_results]

		MLIR.IR.block!(fnbody) do
			MLIR.Dialects.func.return_([res.mlir_data for res in linear_results])
		end

		func2 = MLIR.Dialects.func.func_(; sym_name="main", function_type=MLIR.IR.FunctionType(in_tys, out_tys), body=MLIR.IR.Region())
		MLIR.API.mlirRegionTakeBody(MLIR.IR.region(func2, 1), MLIR.IR.region(func, 1))

		push!(modbody, func2)

		concrete_seen = IdDict()

		concrete_result = make_tracer(concrete_seen, traced_result, ("result",), TracedToConcrete, #=data=#nothing)

		if client === nothing
			if length(linear_args) > 0
				for (k, v) in seen_args
					if !(v isa TracedRArray)
						continue
					end
					client = XLA.client(k.data)
				end
			end
			if client === nothing
				client = XLA.default_backend[]
			end
		end

		return generate_jlfunc(concrete_result, client, mod, N, linear_args, linear_results, preserved_args)
	end
end


end # module
using Reactant
using Reactant: MLIR, Sharding
using Reactant.TracedUtils: get_mlir_data, set_mlir_data!, get_paths, set_paths!
using MPI

for (name, supertype, julia_type) in [(:Communicator, Any, MPI.Comm), (:Request, MPI.AbstractRequest, MPI.Request)]
    traced_type = Symbol("Traced", name)
    concrete_type = Symbol("Concrete", name)

    @eval begin
        struct $concrete_type{NF} <: $supertype
            data::NF
        end

        $concrete_type(x::$julia_type) = $concrete_type(ConcreteRNumber{Int64}(Int64(x.val)))
        $concrete_type{NF}(x::$julia_type) where {NF} = $concrete_type{NF}(NF(Int64(x.val)))

        mutable struct $traced_type <: $supertype
            paths::Tuple
            mlir_data::Union{Nothing, Reactant.MLIR.IR.Value}
        end

        function Base.show(io::IOty, X::$traced_type) where {IOty<:Union{IO,IOContext}}
            return print(io, "$traced_type(paths=", X.paths, ")")
        end

        Reactant.TracedUtils.get_mlir_data(x::$traced_type) = x.mlir_data
        Reactant.TracedUtils.set_mlir_data!(x::$traced_type, data) = (x.mlir_data = data; return x)

        Reactant.TracedUtils.get_paths(x::$traced_type) = x.paths
        Reactant.TracedUtils.set_paths!(x::$traced_type, paths) = (x.paths = paths; return x)

        function Reactant.traced_type_inner(T::Type{$julia_type}, seen, mode::Reactant.TraceMode, @nospecialize(track_numbers::Type), @nospecialize(ndevices), @nospecialize(runtime))
            if mode == Reactant.ArrayToConcrete
                return $concrete_type
            else
                return T
            end
        end

        function Reactant.traced_type_inner(T::Type{$concrete_type}, seen, mode::Reactant.TraceMode, @nospecialize(track_numbers::Type), @nospecialize(ndevices), @nospecialize(runtime))
            if mode == Reactant.ConcreteToTraced
                return $traced_type
            else
                return T
            end
        end

        function Reactant.traced_type_inner(T::Type{$traced_type}, seen, mode::Reactant.TraceMode, @nospecialize(track_numbers::Type), @nospecialize(ndevices), @nospecialize(runtime))
            if mode == Reactant.TracedToConcrete
                return $concrete_type
            else
                return T
            end
        end

        function Reactant.make_tracer(seen, @nospecialize(prev::$julia_type), @nospecialize(path), mode; @nospecialize(sharding = Sharding.NoSharding()), @nospecialize(runtime = nothing), @nospecialize(device = nothing), @nospecialize(client = nothing), kwargs...)
            if Sharding.is_sharded(sharding)
                error("Simultaneous use of sharding and MPI is not supported")
            end

            if mode == Reactant.ArrayToConcrete
                haskey(seen, prev) && return seen[prev]::$concrete_type
                res = if runtime isa Val{:PJRT}
                    $concrete_type(ConcretePJRTNumber{Int64,1}(prev; device, client))
                elseif runtime isa Val{:IFRT}
                    $concrete_type(ConcreteIFRTNumber{Int64,1}(prev; device, client))
                else
                    error("Unsupported runtime $runtime")
                end
                seen[prev] = res
                return res

            else
                return prev
            end
        end

        function Reactant.make_tracer(seen, @nospecialize(prev::$concrete_type), @nospecialize(path), mode; @nospecialize(sharding = Sharding.NoSharding()), @nospecialize(runtime = nothing), @nospecialize(device = nothing), @nospecialize(client = nothing), kwargs...)
            if Sharding.is_sharded(sharding)
                error("Simultaneous use of sharding and MPI is not supported")
            end

            if mode == ArrayToConcrete
                if runtime isa Val{:PJRT}
                    return $concrete_type(ConcretePJRTNumber{Int64,1}(prev; device, client))
                elseif runtime isa Val{:IFRT}
                    return $concrete_type(ConcreteIFRTNumber{Int64,1}(prev; device, client))
                else
                    error("Unsupported runtime $runtime")
                end

            elseif mode == Reactant.ConcreteToTraced
                haskey(seen, prev) && return seen[prev]::$traced_type
                res = $traced_type((path,), nothing)
                seen[prev] = res
                return res

            else
                throw("Trace mode $mode not implemented for $concrete_type")
            end
        end

        # TODO(#2242) for this to work properly in finalize_mlir_fn(), need to add TracedRequest to TracedTypes, currently const
        function Reactant.make_tracer(seen, @nospecialize(prev::$traced_type), @nospecialize(path), mode; @nospecialize(sharding = Sharding.NoSharding()), @nospecialize(runtime = nothing), kwargs...)
            if Sharding.is_sharded(sharding)
                error("Simultaneous use of sharding and MPI is not supported")
            end

            if mode == Reactant.ConcreteToTraced
                throw("Cannot trace existing trace type")

            elseif mode == Reactant.TracedToTypes
                push!(path, MLIR.IR.type(get_mlir_data(prev)))
                return nothing

            elseif mode == Reactant.TracedTrack || mode == Reactant.TracedSetPath
                set_paths!(prev, (get_paths(prev)..., path))
                if !haskey(seen, prev)
                    return seen[prev] = prev
                end
                return prev

            elseif mode == mode == Reactant.NoStopTracedTrack
                set_paths!(prev, (get_paths(prev)..., path))
                if !haskey(seen, prev)
                    seen[prev] = prev # don't return!
                end
                return prev

            elseif mode == Reactant.TracedToConcrete
                if runtime isa Val{:PJRT}
                    haskey(seen, prev) && return seen[prev]::$concrete_type
                    if Sharding.is_sharded(sharding)
                        error("Attempting to use sharding and MPI simultaneously")
                    end

                    res = $concrete_type(ConcretePJRTNumber{Int64,1}((Reactant.PJRT.AsyncEmptyBuffer,), Sharding.NoShardInfo()))
                    seen[prev] = res
                    return res

                elseif runtime isa Val{:IFRT}
                    haskey(seen, prev) && return seen[prev]::$concrete_type
                    if Sharding.is_sharded(sharding)
                        error("Attempting to use sharding and MPI simultaneously")
                    end

                    res = $concrete_type(ConcreteIFRTNumber{Int64,1}((Reactant.IFRT.AsyncEmptyBuffer,), Sharding.NoShardInfo()))
                    seen[prev] = res
                    return res

                else
                    error("Unsupported runtime $runtime")
                end

            else
                throw("Trace mode $mode not implemented for $julia_type")
            end
        end

        function Reactant.Compiler.create_result(
            tocopy::T,
            @nospecialize(path::Tuple),
            result_stores,
            path_to_shard_info,
            to_unreshard_results,
            unresharded_code::Vector{Expr},
            unresharded_arrays_cache,
            used_shardinfo,
            result_cache,
            var_idx,
            resultgen_code,
        ) where {T<:$concrete_type}
            if haskey(result_cache, tocopy)
                return result_cache[tocopy]
            end

            if path_to_shard_info !== nothing && haskey(path_to_shard_info, path)
                error("Attempting to use sharding and MPI simultaneously")
            end

            sym = Symbol("result", var_idx[])
            var_idx[] += 1

            result_cache[tocopy] = sym

            restore = result_stores[path]
            delete!(result_stores, path)

            result = :($T(ConcretePJRTNumber{Int64}($restore)))
            push!(resultgen_code, quote
                $sym = $result
            end)

            return result_cache[tocopy] = sym
        end

        function Reactant.Compiler.traced_setfield!(@nospecialize(obj::$concrete_type), field, val, path)
            Reactant.Compiler.check_aliased_buffer_assignment(obj, field, val, path)
            return Base.setproperty!(obj, field, val, path)
        end

        function Reactant.Compiler.traced_setfield_buffer!(
            ::Val{:PJRT},
            prev,
            cache_dict,
            val::$traced_type,
            concrete_res,
            obj,
            field,
            path,
        )
            if haskey(cache_dict, val)
                cval = cache_dict[val]
            else
                cval = $concrete_type(ConcretePJRTNumber{Int64}(concrete_res))
                cache_dict[val] = cval
            end

            Reactant.Compiler.check_aliased_buffer_assignment(obj, field, cval, path)
            return Base.setproperty!(obj, field, cval, path)
        end
    end
end

Reactant.Ops.mlir_type(x::TracedCommunicator) = MLIR.IR.Type(MLIR.API.enzymexlaCommMpiCommTypeGet(IR.current_context()))
Reactant.Ops.mlir_type(x::TracedRequest) = MLIR.IR.Type(MLIR.API.enzymexlaCommMpiRequestTypeGet(IR.current_context()))

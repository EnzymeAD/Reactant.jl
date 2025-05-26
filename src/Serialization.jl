module Serialization

# TODO: move these deps into an extension

using JLD2
using Reactant: Reactant, MLIR, XLA

struct SerializedThunk
    f
    body::Expr
    argTypes
    IsClosure::Bool
    num_parameters::Int
    num_results::Int
    is_device_present::Bool
    num_devices::Int
    module_string::String
end

function serialize(
    filename::String, thunk::Reactant.Compiler.Thunk{FTy,tag,IsClosure,ArgTypes}
) where {FTy,tag,IsClosure,ArgTypes}
    if isempty(thunk.module_string)
        throw("To serialize a compiled thunk, ensure it is called with `serializable=true`")
    end

    serializable_thunk = SerializedThunk(
        thunk.f,
        Reactant.Compiler.__thunk_body_cache[tag],
        ArgTypes,
        IsClosure,
        thunk.exec.num_parameters,
        thunk.exec.num_outputs,
        thunk.device !== nothing,
        thunk.device !== nothing ? 1 : length(thunk.global_device_ids),
        thunk.module_string,
    )

    return JLD2.jldsave(filename; thunk=serializable_thunk)
end

function deserialize(f, filename::String; client, device, global_device_ids)
    if !isfile(filename)
        error("File $(filename) does not exist")
    end

    serialized_thunk = JLD2.jldopen(filename, "r") do file
        file["thunk"]
    end

    mod = MLIR.IR.with_context() do ctx
        parse(MLIR.IR.Module, serialized_thunk.module_string)
    end
    modop = MLIR.IR.Operation(mod)

    # We always insert these attributes
    num_replicas = Int(MLIR.IR.attr(modop, "mhlo.num_replicas"))
    num_partitions = Int(MLIR.IR.attr(modop, "mhlo.num_partitions"))
    is_sharded = num_replicas * num_partitions > 1
    use_shardy_partitioner = false

    if !serialized_thunk.is_device_present
        @assert serialized_thunk.num_devices == length(global_device_ids)
    end

    exec = XLA.compile(
        client,
        device,
        mod;
        num_outputs=serialized_thunk.num_results,
        num_parameters=serialized_thunk.num_parameters,
        is_sharded,
        global_device_ids,
        num_replicas,
        num_partitions,
        use_shardy_partitioner,
    )

    fname = gensym(Symbol(Symbol(f), :_reactant))
    Reactant.Compiler.__thunk_body_cache[fname] = serialized_thunk.body
    thunk = thunk_from_serialized_thunk(
        f,
        serialized_thunk,
        exec,
        fname,
        client,
        global_device_ids,
        device,
        serialized_thunk.module_string,
    )

    return thunk
end

function thunk_from_serialized_thunk(
    f::F,
    serialized_thunk::SerializedThunk,
    exec,
    tag,
    client,
    global_device_ids,
    device,
    module_string,
) where {F}
    return Reactant.Compiler.Thunk{
        F,
        tag,
        serialized_thunk.IsClosure,
        serialized_thunk.argTypes,
        typeof(exec),
        typeof(device),
        typeof(client),
        typeof(global_device_ids),
    }(
        f, exec, device, module_string, client, global_device_ids
    )
end

end

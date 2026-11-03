module TFSavedModel

using ProtoBuf: OneOf
using ..Serialization: NUMPY_SIMPLE_TYPES
import ..Serialization: serialization_supported
using ..Reactant: Reactant, Compiler, MLIR

include("tf_proto.jl")
include("tf_tensor_bundle.jl")

using .TFProto:
    AttrListValue,
    AttrValue,
    BoolValue,
    CollectionDef,
    CollectionNodeList,
    DT_BFLOAT16,
    DT_BOOL,
    DT_COMPLEX128,
    DT_COMPLEX64,
    DT_DOUBLE,
    DT_FLOAT,
    DT_HALF,
    DT_INT16,
    DT_INT32,
    DT_INT64,
    DT_INT8,
    DT_INVALID,
    DT_RESOURCE,
    DT_STRING,
    DT_UINT16,
    DT_UINT32,
    DT_UINT64,
    DT_UINT8,
    FunctionArgAttrs,
    FunctionDef,
    FunctionDefLibrary,
    FunctionSpec,
    GraphDef,
    MetaGraphDef,
    MetaInfoDef,
    NameAttrList,
    NodeDef,
    ObjectReference,
    OpArgDef,
    OpDef,
    PairValue,
    ResourceDtypeAndShape,
    SAVER_V2,
    SavedBareConcreteFunction,
    SavedConcreteFunction,
    SavedFunction,
    SavedModel,
    SavedObject,
    SavedObjectGraph,
    SavedUserObject,
    SavedVariable,
    SaverDef,
    SerializedTensor,
    SignatureDef,
    StructuredValue,
    TFDataType,
    TensorInfo,
    TensorProto,
    TensorShapeProto,
    TensorSpecProto,
    TrackableObject,
    TrackableObjectGraph,
    VersionDef,
    attr_b,
    attr_bytes,
    attr_func,
    attr_i,
    attr_list,
    attr_s,
    attr_shape,
    attr_tensor,
    attr_type,
    proto_to_bytes,
    sv_dict,
    sv_list,
    sv_namedtuple,
    sv_none,
    sv_spec,
    sv_string,
    sv_tuple
using .TFTensorBundle: BundleTensorSpec, write_tensor_bundle

serialization_supported(::Val{:SavedModel}) = true

struct VariableSignature
    shape::Vector{Int}
    dtype::Symbol
end

abstract type VariableType end

struct InputArgument <: VariableType
    position::Int
end

struct Parameter <: VariableType
    name::String
end

struct ReactantFunctionSpec
    input_signature::Vector{VariableSignature}
    output_signature::Vector{VariableSignature}
    input_locations::Vector{<:VariableType}
    bytecode::Base.CodeUnits{UInt8,String}
    state_dict::Dict
end

const SYMBOL_TO_TF_DTYPE = Dict{Symbol,TFDataType}(
    :bool => DT_BOOL,
    :int8 => DT_INT8,
    :int16 => DT_INT16,
    :int32 => DT_INT32,
    :int64 => DT_INT64,
    :uint8 => DT_UINT8,
    :uint16 => DT_UINT16,
    :uint32 => DT_UINT32,
    :uint64 => DT_UINT64,
    :float16 => DT_HALF,
    :bfloat16 => DT_BFLOAT16,
    :float32 => DT_FLOAT,
    :float64 => DT_DOUBLE,
    :complex64 => DT_COMPLEX64,
    :complex128 => DT_COMPLEX128,
)

function tf_dtype(sym::Symbol)::TFDataType
    dt = get(SYMBOL_TO_TF_DTYPE, sym, nothing)
    isnothing(dt) &&
        throw(ArgumentError("Unsupported TensorFlow SavedModel dtype: $(repr(sym))"))
    return dt
end

function julia_type_to_dtype_symbol(T::Type)::Symbol
    sym = get(NUMPY_SIMPLE_TYPES, T, nothing)
    isnothing(sym) &&
        throw(ArgumentError("Unsupported Julia element type for SavedModel export: $(T)"))
    return sym
end

function export_as_saved_model(
    thunk::Compiler.Thunk,
    saved_model_path::String,
    target_version::VersionNumber,
    input_locations::Vector=[],
    state_dict::Dict=Dict(),
)
    isempty(thunk.module_string) && error(
        "To export a thunk, ensure that it has been compiled with `serializable=true`."
    )

    if !serialization_supported(Val(:SavedModel))
        error("Serialization to SavedModel is not supported.")
    end

    MLIR.IR.@dispose ctx = Reactant.ReactantContext() begin
        MLIR.IR.activate(ctx)
        try
            mlir_mod = parse(MLIR.IR.Module, thunk.module_string)
            ftype = MLIR.IR.FunctionType(first(MLIR.IR.body(mlir_mod)))

            input_signature = [
                VariableSignature(
                    collect(Int64, size(MLIR.IR.input(ftype, i))),
                    julia_type_to_dtype_symbol(
                        MLIR.IR.julia_type(eltype(MLIR.IR.input(ftype, i)))
                    ),
                ) for i in 1:MLIR.IR.ninputs(ftype)
            ]

            output_signature = [
                VariableSignature(
                    collect(Int64, size(MLIR.IR.result(ftype, i))),
                    julia_type_to_dtype_symbol(
                        MLIR.IR.julia_type(eltype(MLIR.IR.result(ftype, i)))
                    ),
                ) for i in 1:MLIR.IR.nresults(ftype)
            ]

            if isempty(input_locations)
                input_locations = [InputArgument(i) for i in 1:length(input_signature)]
            end

            @assert length(input_locations) == length(input_signature) "The number of input \
                                                                        locations must match the \
                                                                        number of input signatures."

            c_print_callback = @cfunction(
                MLIR.IR.print_callback, Cvoid, (MLIR.API.MlirStringRef, Any)
            )
            ref = Ref(IOBuffer())
            result = MLIR.IR.LogicalResult(
                MLIR.API.stablehloSerializePortableArtifactFromModule(
                    mlir_mod, string(target_version), c_print_callback, ref, true
                ),
            )
            MLIR.IR.isfailure(result) && error("Couldn't serialize the module")
            serialized_module = codeunits(String(take!(ref[])))

            return to_tf_saved_model(
                ReactantFunctionSpec(
                    input_signature,
                    output_signature,
                    input_locations,
                    serialized_module,
                    Dict(k => Array(v) for (k, v) in state_dict),
                ),
                saved_model_path,
            )
        finally
            MLIR.IR.deactivate(ctx)
        end
    end
end

function build_fullargspec(;
    args::Vector{String}=String[], kwonlyargs::Vector{String}=String[]
)::StructuredValue
    return sv_namedtuple(
        "FullArgSpec",
        PairValue[
            PairValue("args", sv_list(StructuredValue[sv_string(a) for a in args])),
            PairValue("varargs", sv_none()),
            PairValue("varkw", sv_none()),
            PairValue("defaults", sv_none()),
            PairValue(
                "kwonlyargs", sv_list(StructuredValue[sv_string(k) for k in kwonlyargs])
            ),
            PairValue("kwonlydefaults", sv_none()),
            PairValue("annotations", sv_dict()),
        ],
    )
end

function build_inference_f(
    fn_spec::ReactantFunctionSpec,
    user_in_dtypes::Vector{TFDataType},
    user_in_shapes::Vector{TensorShapeProto},
    var_dtypes::Vector{TFDataType},
    var_shapes::Vector{TensorShapeProto},
    param_idx_map::Dict{String,Int},
    out_dtypes::Vector{TFDataType},
    out_shapes::Vector{TensorShapeProto},
)::FunctionDef
    K = length(user_in_dtypes)
    P = length(var_dtypes)
    M = length(out_dtypes)

    input_args = OpArgDef[]
    for i0 in 0:(K - 1)
        push!(input_args, OpArgDef("args_$(i0)", user_in_dtypes[i0 + 1]))
    end
    for idx0 in 0:(P - 1)
        push!(
            input_args,
            OpArgDef(
                "resource_$(idx0)",
                DT_RESOURCE,
                [ResourceDtypeAndShape(var_dtypes[idx0 + 1], var_shapes[idx0 + 1])],
            ),
        )
    end

    output_args = OpArgDef[
        OpArgDef(j0 == 0 ? "identity" : "identity_$(j0)", out_dtypes[j0 + 1]) for
        j0 in 0:(M - 1)
    ]

    control_outputs = vcat(
        String["XlaCallModule"], String["ReadVariableOp_$(idx0)" for idx0 in 0:(P - 1)]
    )

    nodes = NodeDef[]
    for idx0 in 0:(P - 1)
        push!(
            nodes,
            NodeDef(;
                name="ReadVariableOp_$(idx0)",
                op="ReadVariableOp",
                input=String["resource_$(idx0)"],
                attr=Dict{String,AttrValue}(
                    "dtype" => attr_type(var_dtypes[idx0 + 1]),
                    "_output_shapes" =>
                        attr_list(AttrListValue(; shape=[var_shapes[idx0 + 1]])),
                ),
            ),
        )
    end

    xla_inputs = String[]
    xla_tin = TFDataType[]
    for (loc, sig) in zip(fn_spec.input_locations, fn_spec.input_signature)
        push!(xla_tin, tf_dtype(sig.dtype))
        if loc isa InputArgument
            push!(xla_inputs, "args_$(loc.position - 1)")
        elseif loc isa Parameter
            idx0 = param_idx_map[loc.name] - 1
            push!(xla_inputs, "ReadVariableOp_$(idx0):value:0")
        else
            throw(ArgumentError("Unsupported VariableType: $(typeof(loc))"))
        end
    end

    push!(
        nodes,
        NodeDef(;
            name="XlaCallModule",
            op="XlaCallModule",
            input=xla_inputs,
            attr=Dict{String,AttrValue}(
                "version" => attr_i(5),
                "module" => attr_bytes(fn_spec.bytecode),
                "Tin" => attr_list(AttrListValue(; type=xla_tin)),
                "Tout" => attr_list(AttrListValue(; type=out_dtypes)),
                "Sout" => attr_list(AttrListValue(; shape=out_shapes)),
                "_output_shapes" => attr_list(AttrListValue(; shape=out_shapes)),
            ),
        ),
    )

    noop_inputs = String["^$(c)" for c in control_outputs]
    push!(
        nodes,
        NodeDef(;
            name="NoOp",
            op="NoOp",
            input=noop_inputs,
            attr=Dict{String,AttrValue}("_output_shapes" => attr_list(AttrListValue())),
        ),
    )

    ret = Dict{String,String}()
    for j0 in 0:(M - 1)
        id_name = j0 == 0 ? "Identity" : "Identity_$(j0)"
        out_name = j0 == 0 ? "identity" : "identity_$(j0)"
        push!(
            nodes,
            NodeDef(;
                name=id_name,
                op="Identity",
                input=String["XlaCallModule:output:$(j0)", "^NoOp"],
                attr=Dict{String,AttrValue}(
                    "T" => attr_type(out_dtypes[j0 + 1]),
                    "_output_shapes" =>
                        attr_list(AttrListValue(; shape=[out_shapes[j0 + 1]])),
                ),
            ),
        )
        ret[out_name] = "$(id_name):output:0"
    end

    arg_attr = Dict{UInt32,FunctionArgAttrs}()
    for i0 in 0:(K - 1)
        arg_attr[UInt32(i0)] = FunctionArgAttrs(
            Dict{String,AttrValue}(
                "_user_specified_name" => attr_s("args_$(i0)"),
                "_output_shapes" =>
                    attr_list(AttrListValue(; shape=[user_in_shapes[i0 + 1]])),
            ),
        )
    end
    for idx0 in 0:(P - 1)
        arg_attr[UInt32(K + idx0)] = FunctionArgAttrs(
            Dict{String,AttrValue}("_user_specified_name" => attr_s("resource_$(idx0)"))
        )
    end

    all_input_shapes = vcat(user_in_shapes, [TensorShapeProto() for _ in 1:P])

    return FunctionDef(;
        signature=OpDef(;
            name="__inference_f",
            input_arg=input_args,
            output_arg=output_args,
            is_stateful=true,
            control_output=control_outputs,
        ),
        node_def=nodes,
        ret=ret,
        control_ret=Dict{String,String}(c => c for c in control_outputs),
        attr=Dict{String,AttrValue}(
            "_construction_context" => attr_s("kEagerRuntime"),
            "_input_shapes" => attr_list(AttrListValue(; shape=all_input_shapes)),
        ),
        arg_attr=arg_attr,
    )
end

function build_signature_wrapper(
    user_in_dtypes::Vector{TFDataType},
    user_in_shapes::Vector{TensorShapeProto},
    var_dtypes::Vector{TFDataType},
    var_shapes::Vector{TensorShapeProto},
    out_dtypes::Vector{TFDataType},
    out_shapes::Vector{TensorShapeProto},
)::FunctionDef
    K = length(user_in_dtypes)
    P = length(var_dtypes)
    M = length(out_dtypes)

    input_args = OpArgDef[]
    call_inputs = String[]
    call_tin = TFDataType[]
    for i0 in 0:(K - 1)
        push!(input_args, OpArgDef("args_$(i0)", user_in_dtypes[i0 + 1]))
        push!(call_inputs, "args_$(i0)")
        push!(call_tin, user_in_dtypes[i0 + 1])
    end
    for idx0 in 0:(P - 1)
        push!(
            input_args,
            OpArgDef(
                "resource_$(idx0)",
                DT_RESOURCE,
                [ResourceDtypeAndShape(var_dtypes[idx0 + 1], var_shapes[idx0 + 1])],
            ),
        )
        push!(call_inputs, "resource_$(idx0)")
        push!(call_tin, DT_RESOURCE)
    end

    output_args = OpArgDef[
        OpArgDef(j0 == 0 ? "identity" : "identity_$(j0)", out_dtypes[j0 + 1]) for
        j0 in 0:(M - 1)
    ]

    nodes = NodeDef[
        NodeDef(;
            name="StatefulPartitionedCall",
            op="StatefulPartitionedCall",
            input=call_inputs,
            attr=Dict{String,AttrValue}(
                "f" => attr_func(NameAttrList("__inference_f")),
                "Tin" => attr_list(AttrListValue(; type=call_tin)),
                "Tout" => attr_list(AttrListValue(; type=out_dtypes)),
                "_output_shapes" => attr_list(AttrListValue(; shape=out_shapes)),
                "_collective_manager_ids" => attr_list(AttrListValue()),
                "_read_only_resource_inputs" =>
                    attr_list(AttrListValue(; i=Int64[K + idx0 for idx0 in 0:(P - 1)])),
            ),
        ),
        NodeDef(;
            name="NoOp",
            op="NoOp",
            input=String["^StatefulPartitionedCall"],
            attr=Dict{String,AttrValue}("_output_shapes" => attr_list(AttrListValue())),
        ),
    ]

    ret = Dict{String,String}()
    for j0 in 0:(M - 1)
        id_name = j0 == 0 ? "Identity" : "Identity_$(j0)"
        out_name = j0 == 0 ? "identity" : "identity_$(j0)"
        push!(
            nodes,
            NodeDef(;
                name=id_name,
                op="Identity",
                input=String["StatefulPartitionedCall:output:$(j0)", "^NoOp"],
                attr=Dict{String,AttrValue}(
                    "T" => attr_type(out_dtypes[j0 + 1]),
                    "_output_shapes" =>
                        attr_list(AttrListValue(; shape=[out_shapes[j0 + 1]])),
                ),
            ),
        )
        ret[out_name] = "$(id_name):output:0"
    end

    arg_attr = Dict{UInt32,FunctionArgAttrs}()
    for i0 in 0:(K - 1)
        arg_attr[UInt32(i0)] = FunctionArgAttrs(
            Dict{String,AttrValue}(
                "_user_specified_name" => attr_s("args_$(i0)"),
                "_output_shapes" =>
                    attr_list(AttrListValue(; shape=[user_in_shapes[i0 + 1]])),
            ),
        )
    end
    for idx0 in 0:(P - 1)
        arg_attr[UInt32(K + idx0)] = FunctionArgAttrs(
            Dict{String,AttrValue}("_user_specified_name" => attr_s("resource_$(idx0)"))
        )
    end

    all_input_shapes = vcat(user_in_shapes, [TensorShapeProto() for _ in 1:P])

    return FunctionDef(;
        signature=OpDef(;
            name="__inference_signature_wrapper",
            input_arg=input_args,
            output_arg=output_args,
            is_stateful=true,
            control_output=String["StatefulPartitionedCall"],
        ),
        node_def=nodes,
        ret=ret,
        control_ret=Dict{String,String}(
            "StatefulPartitionedCall" => "StatefulPartitionedCall"
        ),
        attr=Dict{String,AttrValue}(
            "_construction_context" => attr_s("kEagerRuntime"),
            "_input_shapes" => attr_list(AttrListValue(; shape=all_input_shapes)),
        ),
        arg_attr=arg_attr,
    )
end

function build_traced_restore(
    param_names::Vector{String},
    var_dtypes::Vector{TFDataType},
    var_shapes::Vector{TensorShapeProto},
)::FunctionDef
    P = length(param_names)
    ckpt_keys = vcat(
        String["_variables/$(idx0)/.ATTRIBUTES/VARIABLE_VALUE" for idx0 in 0:(P - 1)],
        String["_CHECKPOINTABLE_OBJECT_GRAPH"],
    )
    num_ckpt = P + 1
    restore_dtypes = vcat(var_dtypes, TFDataType[DT_STRING])

    input_args = OpArgDef[OpArgDef("file_prefix", DT_STRING)]
    for idx0 in 0:(P - 1)
        push!(
            input_args,
            OpArgDef(
                "var_res_$(idx0)",
                DT_RESOURCE,
                [ResourceDtypeAndShape(var_dtypes[idx0 + 1], var_shapes[idx0 + 1])],
            ),
        )
    end

    names_shape = TensorShapeProto([num_ckpt])
    nodes = NodeDef[
        NodeDef(;
            name="RestoreV2/tensor_names",
            op="Const",
            device="/device:CPU:0",
            attr=Dict{String,AttrValue}(
                "dtype" => attr_type(DT_STRING),
                "value" => attr_tensor(
                    TensorProto(;
                        dtype=DT_STRING,
                        tensor_shape=names_shape,
                        string_val=[Vector{UInt8}(codeunits(k)) for k in ckpt_keys],
                    ),
                ),
                "_output_shapes" => attr_list(AttrListValue(; shape=[names_shape])),
            ),
        ),
        NodeDef(;
            name="RestoreV2/shape_and_slices",
            op="Const",
            device="/device:CPU:0",
            attr=Dict{String,AttrValue}(
                "dtype" => attr_type(DT_STRING),
                "value" => attr_tensor(
                    TensorProto(;
                        dtype=DT_STRING,
                        tensor_shape=names_shape,
                        string_val=[UInt8[] for _ in 1:num_ckpt],
                    ),
                ),
                "_output_shapes" => attr_list(AttrListValue(; shape=[names_shape])),
            ),
        ),
        NodeDef(;
            name="RestoreV2",
            op="RestoreV2",
            input=String[
                "file_prefix",
                "RestoreV2/tensor_names:output:0",
                "RestoreV2/shape_and_slices:output:0",
            ],
            device="/device:CPU:0",
            attr=Dict{String,AttrValue}(
                "dtypes" => attr_list(AttrListValue(; type=restore_dtypes)),
                "_output_shapes" => attr_list(
                    AttrListValue(;
                        shape=[TensorShapeProto(; unknown_rank=true) for _ in 1:num_ckpt],
                    ),
                ),
            ),
        ),
    ]

    assign_names = String[]
    for idx0 in 0:(P - 1)
        aname = "AssignVariableOp_$(idx0)"
        push!(assign_names, aname)
        push!(
            nodes,
            NodeDef(;
                name=aname,
                op="AssignVariableOp",
                input=String["var_res_$(idx0)", "RestoreV2:tensors:$(idx0)"],
                device="/device:CPU:0",
                attr=Dict{String,AttrValue}(
                    "dtype" => attr_type(var_dtypes[idx0 + 1]),
                    "_has_manual_control_dependencies" => attr_b(true),
                    "_output_shapes" => attr_list(AttrListValue()),
                ),
            ),
        )
    end

    noop_ctrl = vcat(String["^RestoreV2"], String["^$(a)" for a in assign_names])
    push!(
        nodes,
        NodeDef(;
            name="NoOp",
            op="NoOp",
            input=noop_ctrl,
            device="/device:CPU:0",
            attr=Dict{String,AttrValue}("_output_shapes" => attr_list(AttrListValue())),
        ),
    )
    push!(
        nodes,
        NodeDef(;
            name="Identity_out",
            op="Identity",
            input=String["file_prefix", "^NoOp"],
            attr=Dict{String,AttrValue}(
                "T" => attr_type(DT_STRING),
                "_output_shapes" => attr_list(AttrListValue(; shape=[TensorShapeProto()])),
            ),
        ),
    )

    arg_attr = Dict{UInt32,FunctionArgAttrs}(
        UInt32(0) => FunctionArgAttrs(
            Dict{String,AttrValue}(
                "_user_specified_name" => attr_s("file_prefix"),
                "_output_shapes" =>
                    attr_list(AttrListValue(; shape=[TensorShapeProto()])),
            ),
        ),
    )
    for idx0 in 0:(P - 1)
        arg_attr[UInt32(idx0 + 1)] = FunctionArgAttrs(
            Dict{String,AttrValue}("_user_specified_name" => attr_s(param_names[idx0 + 1]))
        )
    end

    return FunctionDef(;
        signature=OpDef(;
            name="__inference__traced_restore",
            input_arg=input_args,
            output_arg=OpArgDef[OpArgDef("identity_out", DT_STRING)],
            is_stateful=true,
            control_output=assign_names,
        ),
        node_def=nodes,
        ret=Dict{String,String}("identity_out" => "Identity_out:output:0"),
        control_ret=Dict{String,String}(a => a for a in assign_names),
        attr=Dict{String,AttrValue}(
            "_construction_context" => attr_s("kEagerRuntime"),
            "_input_shapes" =>
                attr_list(AttrListValue(; shape=[TensorShapeProto() for _ in 1:(P + 1)])),
        ),
        arg_attr=arg_attr,
    )
end

function to_tf_saved_model(fn_spec::ReactantFunctionSpec, path::String)
    if !serialization_supported(Val(:SavedModel))
        error("Serialization to SavedModel is not supported.")
    end

    length(fn_spec.input_locations) == length(fn_spec.input_signature) || throw(
        ArgumentError(
            "The number of input locations ($(length(fn_spec.input_locations))) must match \
             the number of input signatures ($(length(fn_spec.input_signature))).",
        ),
    )

    input_pos_to_sig = Dict{Int,VariableSignature}()
    param_names = String[]
    param_sigs = VariableSignature[]
    param_arrays = Array[]
    param_idx_map = Dict{String,Int}()

    for (loc, sig) in zip(fn_spec.input_locations, fn_spec.input_signature)
        if loc isa InputArgument
            loc.position >= 1 || throw(
                ArgumentError("InputArgument position must be >= 1, got $(loc.position)"),
            )
            if haskey(input_pos_to_sig, loc.position)
                existing = input_pos_to_sig[loc.position]
                (existing.shape == sig.shape && existing.dtype == sig.dtype) || throw(
                    ArgumentError(
                        "Conflicting signatures for InputArgument($(loc.position))"
                    ),
                )
            else
                input_pos_to_sig[loc.position] = sig
            end
        elseif loc isa Parameter
            haskey(fn_spec.state_dict, loc.name) ||
                throw(ArgumentError("State dictionary does not contain key: $(loc.name)"))
            arr = Array(fn_spec.state_dict[loc.name])
            arr_dtype = julia_type_to_dtype_symbol(eltype(arr))
            arr_dtype === sig.dtype || throw(
                ArgumentError(
                    "Parameter $(repr(loc.name)) has dtype $(arr_dtype), expected $(sig.dtype)",
                ),
            )
            arr_tf_shape = reverse(collect(Int64, size(arr)))
            arr_tf_shape == sig.shape || throw(
                ArgumentError(
                    "Parameter $(repr(loc.name)) has TF shape $(arr_tf_shape) (Julia size $(size(arr))), expected $(sig.shape)",
                ),
            )
            if !haskey(param_idx_map, loc.name)
                push!(param_names, loc.name)
                push!(param_sigs, sig)
                push!(param_arrays, arr)
                param_idx_map[loc.name] = length(param_names)
            end
        else
            throw(ArgumentError("Unsupported VariableType: $(typeof(loc))"))
        end
    end

    K = length(input_pos_to_sig)
    for i in 1:K
        haskey(input_pos_to_sig, i) ||
            throw(ArgumentError("Missing InputArgument($(i)) in input_locations"))
    end
    P = length(param_names)
    M = length(fn_spec.output_signature)

    user_in_dtypes = TFDataType[tf_dtype(input_pos_to_sig[i].dtype) for i in 1:K]
    user_in_shapes = TensorShapeProto[
        TensorShapeProto(input_pos_to_sig[i].shape) for i in 1:K
    ]
    var_dtypes = TFDataType[tf_dtype(s.dtype) for s in param_sigs]
    var_shapes = TensorShapeProto[TensorShapeProto(s.shape) for s in param_sigs]
    out_dtypes = TFDataType[tf_dtype(s.dtype) for s in fn_spec.output_signature]
    out_shapes = TensorShapeProto[
        TensorShapeProto(s.shape) for s in fn_spec.output_signature
    ]

    # 1. Build top-level GraphDef nodes and FunctionDefLibrary
    top_nodes = NodeDef[]
    var_node_names = String[]
    for idx0 in 0:(P - 1)
        vname = "param_$(idx0)"
        push!(var_node_names, vname)
        push!(
            top_nodes,
            NodeDef(;
                name=vname,
                op="VarHandleOp",
                attr=Dict{String,AttrValue}(
                    "dtype" => attr_type(var_dtypes[idx0 + 1]),
                    "shape" => attr_shape(var_shapes[idx0 + 1]),
                    "shared_name" => attr_s(vname),
                    "_output_shapes" =>
                        attr_list(AttrListValue(; shape=[TensorShapeProto()])),
                ),
            ),
        )
    end

    placeholder_names = String[]
    for i0 in 0:(K - 1)
        pname = "serving_default_args_$(i0)"
        push!(placeholder_names, pname)
        push!(
            top_nodes,
            NodeDef(;
                name=pname,
                op="Placeholder",
                attr=Dict{String,AttrValue}(
                    "dtype" => attr_type(user_in_dtypes[i0 + 1]),
                    "shape" => attr_shape(user_in_shapes[i0 + 1]),
                    "_output_shapes" =>
                        attr_list(AttrListValue(; shape=[user_in_shapes[i0 + 1]])),
                ),
            ),
        )
    end

    push!(
        top_nodes,
        NodeDef(;
            name="StatefulPartitionedCall",
            op="StatefulPartitionedCall",
            input=vcat(placeholder_names, var_node_names),
            attr=Dict{String,AttrValue}(
                "f" => attr_func(NameAttrList("__inference_signature_wrapper")),
                "Tin" => attr_list(
                    AttrListValue(; type=vcat(user_in_dtypes, fill(DT_RESOURCE, P)))
                ),
                "Tout" => attr_list(AttrListValue(; type=out_dtypes)),
                "_output_shapes" => attr_list(AttrListValue(; shape=out_shapes)),
                "_collective_manager_ids" => attr_list(AttrListValue()),
                "_read_only_resource_inputs" =>
                    attr_list(AttrListValue(; i=Int64[K + idx0 for idx0 in 0:(P - 1)])),
            ),
        ),
    )

    push!(top_nodes, NodeDef(; name="NoOp", op="NoOp"))
    push!(
        top_nodes,
        NodeDef(;
            name="saver_filename",
            op="Placeholder",
            attr=Dict{String,AttrValue}(
                "dtype" => attr_type(DT_STRING),
                "shape" => attr_shape(TensorShapeProto()),
                "_output_shapes" => attr_list(AttrListValue(; shape=[TensorShapeProto()])),
            ),
        ),
    )
    push!(
        top_nodes,
        NodeDef(;
            name="StatefulPartitionedCall_1",
            op="StatefulPartitionedCall",
            input=vcat(String["saver_filename"], var_node_names),
            attr=Dict{String,AttrValue}(
                "f" => attr_func(NameAttrList("__inference__traced_restore")),
                "Tin" => attr_list(
                    AttrListValue(; type=vcat(TFDataType[DT_STRING], fill(DT_RESOURCE, P))),
                ),
                "Tout" => attr_list(AttrListValue(; type=TFDataType[DT_STRING])),
                "_output_shapes" => attr_list(AttrListValue(; shape=[TensorShapeProto()])),
                "_collective_manager_ids" => attr_list(AttrListValue()),
                "_read_only_resource_inputs" => attr_list(AttrListValue()),
            ),
        ),
    )

    fn_lib = FunctionDefLibrary(
        FunctionDef[
            build_inference_f(
                fn_spec,
                user_in_dtypes,
                user_in_shapes,
                var_dtypes,
                var_shapes,
                param_idx_map,
                out_dtypes,
                out_shapes,
            ),
            build_signature_wrapper(
                user_in_dtypes,
                user_in_shapes,
                var_dtypes,
                var_shapes,
                out_dtypes,
                out_shapes,
            ),
            build_traced_restore(param_names, var_dtypes, var_shapes),
        ],
    )

    graph_def = GraphDef(top_nodes, fn_lib, VersionDef(; producer=1882, min_consumer=12))

    # 2. Build SavedObjectGraph & TrackableObjectGraph
    # Node IDs:
    #   0: root (_generic_user_object)
    #   1: _variables (trackable_list_wrapper)
    #   2: f (SavedFunction)
    #   3: signatures (signature_map)
    #   4..(3+P): SavedVariable for each parameter
    #   4+P: __inference_f (SavedBareConcreteFunction)
    #   5+P: __inference_signature_wrapper (SavedBareConcreteFunction)
    fn_f_node_id = 4 + P
    fn_sig_node_id = 5 + P
    obj_ver = VersionDef(; producer=1, min_consumer=1)

    arg_names = String["args_$(i0)" for i0 in 0:(K - 1)]
    pos_arg_names = String["a$(i0)" for i0 in 0:(K - 1)]
    user_specs = TensorSpecProto[
        TensorSpecProto("args_$(i0)", user_in_shapes[i0 + 1], user_in_dtypes[i0 + 1]) for
        i0 in 0:(K - 1)
    ]

    f_fullargspec = build_fullargspec(; args=pos_arg_names)
    sig_fullargspec = build_fullargspec(; kwonlyargs=arg_names)

    saved_nodes = SavedObject[
        SavedObject(;
            children=ObjectReference[
                ObjectReference(1, "_variables"),
                ObjectReference(2, "f"),
                ObjectReference(3, "signatures"),
            ],
            kind=OneOf(:user_object, SavedUserObject("_generic_user_object", obj_ver)),
        ),
        SavedObject(;
            children=ObjectReference[ObjectReference(3 + idx, "$(idx - 1)") for idx in 1:P],
            kind=OneOf(:user_object, SavedUserObject("trackable_list_wrapper", obj_ver)),
        ),
        SavedObject(;
            children=ObjectReference[ObjectReference(fn_f_node_id, "trace_0")],
            dependencies=ObjectReference[ObjectReference(fn_f_node_id, "trace_0")],
            kind=OneOf(
                :function,
                SavedFunction(
                    String["__inference_f"],
                    FunctionSpec(
                        f_fullargspec,
                        false,
                        sv_tuple(StructuredValue[sv_spec(s) for s in user_specs]),
                    ),
                ),
            ),
        ),
        SavedObject(;
            children=ObjectReference[ObjectReference(fn_sig_node_id, "serving_default")],
            kind=OneOf(:user_object, SavedUserObject("signature_map", obj_ver)),
        ),
    ]

    has_vars = P > 0
    trackable_nodes = TrackableObject[
        TrackableObject(;
            children=ObjectReference[
                ObjectReference(1, "_variables"),
                ObjectReference(2, "f"),
                ObjectReference(3, "signatures"),
            ],
            has_checkpoint_values=BoolValue(has_vars),
        ),
        TrackableObject(;
            children=ObjectReference[ObjectReference(3 + idx, "$(idx - 1)") for idx in 1:P],
            has_checkpoint_values=BoolValue(has_vars),
        ),
        TrackableObject(;
            children=ObjectReference[ObjectReference(fn_f_node_id, "trace_0")],
            has_checkpoint_values=BoolValue(false),
        ),
        TrackableObject(;
            children=ObjectReference[ObjectReference(fn_sig_node_id, "serving_default")],
            has_checkpoint_values=BoolValue(false),
        ),
    ]

    for idx in 1:P
        idx0 = idx - 1
        push!(
            saved_nodes,
            SavedObject(;
                kind=OneOf(
                    :variable,
                    SavedVariable(;
                        dtype=var_dtypes[idx],
                        shape=var_shapes[idx],
                        name=param_names[idx],
                        trainable=false,
                    ),
                ),
            ),
        )
        push!(
            trackable_nodes,
            TrackableObject(;
                attributes=SerializedTensor[SerializedTensor(
                    "VARIABLE_VALUE",
                    param_names[idx],
                    "_variables/$(idx0)/.ATTRIBUTES/VARIABLE_VALUE",
                ),],
                has_checkpoint_values=BoolValue(true),
            ),
        )
    end

    push!(
        saved_nodes,
        SavedObject(;
            kind=OneOf(
                :bare_concrete_function,
                SavedBareConcreteFunction(
                    "__inference_f",
                    arg_names,
                    Int64(K),
                    FunctionSpec(f_fullargspec, false, sv_none()),
                ),
            ),
        ),
    )
    push!(
        saved_nodes,
        SavedObject(;
            kind=OneOf(
                :bare_concrete_function,
                SavedBareConcreteFunction(
                    "__inference_signature_wrapper",
                    arg_names,
                    Int64(0),
                    FunctionSpec(sig_fullargspec, false, sv_none()),
                ),
            ),
        ),
    )
    push!(trackable_nodes, TrackableObject(; has_checkpoint_values=BoolValue(false)))
    push!(trackable_nodes, TrackableObject(; has_checkpoint_values=BoolValue(false)))

    bound_inputs = Int32[3 + idx for idx in 1:P]
    concrete_functions = Dict{String,SavedConcreteFunction}(
        "__inference_f" => SavedConcreteFunction(
            bound_inputs,
            sv_tuple(
                StructuredValue[
                    sv_tuple(StructuredValue[sv_spec(s) for s in user_specs]), sv_dict()
                ],
            ),
            sv_list(
                StructuredValue[
                    sv_spec(
                        TensorSpecProto(
                            "tensor_$(j0)", out_shapes[j0 + 1], out_dtypes[j0 + 1]
                        ),
                    ) for j0 in 0:(M - 1)
                ],
            ),
        ),
        "__inference_signature_wrapper" => SavedConcreteFunction(
            bound_inputs,
            sv_tuple(
                StructuredValue[
                    sv_tuple(),
                    sv_dict(
                        Dict{String,StructuredValue}(
                            "args_$(i0)" => sv_spec(user_specs[i0 + 1]) for i0 in 0:(K - 1)
                        ),
                    ),
                ],
            ),
            sv_dict(
                Dict{String,StructuredValue}(
                    "output_$(j0)" => sv_spec(
                        TensorSpecProto(
                            "output_$(j0)", out_shapes[j0 + 1], out_dtypes[j0 + 1]
                        ),
                    ) for j0 in 0:(M - 1)
                ),
            ),
        ),
    )

    object_graph_def = SavedObjectGraph(saved_nodes, concrete_functions)
    trackable_graph = TrackableObjectGraph(trackable_nodes)

    # 3. Build MetaGraphDef & SavedModel
    signature_def = Dict{String,SignatureDef}(
        "__saved_model_init_op" => SignatureDef(;
            outputs=Dict{String,TensorInfo}(
                "__saved_model_init_op" => TensorInfo(
                    "NoOp", DT_INVALID, TensorShapeProto(; unknown_rank=true)
                ),
            ),
        ),
        "serving_default" => SignatureDef(;
            inputs=Dict{String,TensorInfo}(
                "args_$(i0)" => TensorInfo(
                    "serving_default_args_$(i0):0",
                    user_in_dtypes[i0 + 1],
                    user_in_shapes[i0 + 1],
                ) for i0 in 0:(K - 1)
            ),
            outputs=Dict{String,TensorInfo}(
                "output_$(j0)" => TensorInfo(
                    "StatefulPartitionedCall:$(j0)",
                    out_dtypes[j0 + 1],
                    out_shapes[j0 + 1],
                ) for j0 in 0:(M - 1)
            ),
            method_name="tensorflow/serving/predict",
        ),
    )

    meta_graph = MetaGraphDef(
        MetaInfoDef(
            String["serve"], "2.17.0", true, Dict{String,String}("__inference_f" => "")
        ),
        graph_def,
        SaverDef(
            "saver_filename:0", "saver_filename:0", "StatefulPartitionedCall_1", SAVER_V2
        ),
        Dict{String,CollectionDef}(
            "saved_model_main_op" => CollectionDef(CollectionNodeList(String["NoOp"]))
        ),
        signature_def,
        object_graph_def,
    )

    saved_model = SavedModel(Int64(1), MetaGraphDef[meta_graph])

    # 4. Write TensorBundle checkpoint and saved_model.pb
    mkpath(path)
    variables_dir = joinpath(path, "variables")
    bundle_tensors = BundleTensorSpec[BundleTensorSpec(
        "_CHECKPOINTABLE_OBJECT_GRAPH",
        DT_STRING,
        Int64[],
        nothing,
        proto_to_bytes(trackable_graph),
    ),]
    for idx in 1:P
        idx0 = idx - 1
        push!(
            bundle_tensors,
            BundleTensorSpec(
                "_variables/$(idx0)/.ATTRIBUTES/VARIABLE_VALUE",
                var_dtypes[idx],
                Int64[d for d in param_sigs[idx].shape],
                param_arrays[idx],
                nothing,
            ),
        )
    end

    write_tensor_bundle(variables_dir, bundle_tensors)
    write(joinpath(path, "saved_model.pb"), proto_to_bytes(saved_model))
    return nothing
end

end # module TFSavedModel

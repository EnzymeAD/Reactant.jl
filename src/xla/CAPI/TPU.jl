using CEnum: CEnum, @cenum

mutable struct TSL_Status end

const TF_Status = TSL_Status

@cenum TpuCoreTypeEnum::UInt32 begin
    kTensorCore = 0
    kEmbeddingV1 = 1
    kEmbeddingV2 = 2
end

@cenum TpuVersionEnum::UInt32 begin
    kUnknownTpuVersion = 0
    kTpuV2 = 1
    kTpuV3 = 2
    kTpuV4 = 3
    kTpuV5 = 4
end

struct TpuRuntimeVersion
    version::NTuple{3,Cint}
    metadata::Cstring
    metadata_size::Csize_t
end

struct TpuSerializedProto
    bytes::Cstring
    size::Csize_t
end

struct IntList
    data::NTuple{32,UInt8}
end

function Base.getproperty(x::Ptr{IntList}, f::Symbol)
    f === :heap && return Ptr{Ptr{Cint}}(x + 0)
    f === :inlined && return Ptr{NTuple{6,Cint}}(x + 0)
    f === :size && return Ptr{Int64}(x + 24)
    return getfield(x, f)
end

function Base.getproperty(x::IntList, f::Symbol)
    r = Ref{IntList}(x)
    ptr = Base.unsafe_convert(Ptr{IntList}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{IntList}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::IntList, private::Bool=false)
    return (:heap, :inlined, :size, if private
        fieldnames(typeof(x))
    else
        ()
    end...)
end

struct Int64List
    data::NTuple{56,UInt8}
end

function Base.getproperty(x::Ptr{Int64List}, f::Symbol)
    f === :heap && return Ptr{Ptr{Int64}}(x + 0)
    f === :inlined && return Ptr{NTuple{6,Int64}}(x + 0)
    f === :size && return Ptr{Int64}(x + 48)
    return getfield(x, f)
end

function Base.getproperty(x::Int64List, f::Symbol)
    r = Ref{Int64List}(x)
    ptr = Base.unsafe_convert(Ptr{Int64List}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{Int64List}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::Int64List, private::Bool=false)
    return (:heap, :inlined, :size, if private
        fieldnames(typeof(x))
    else
        ()
    end...)
end

struct FloatList
    data::NTuple{32,UInt8}
end

function Base.getproperty(x::Ptr{FloatList}, f::Symbol)
    f === :heap && return Ptr{Ptr{Cfloat}}(x + 0)
    f === :inlined && return Ptr{NTuple{6,Cfloat}}(x + 0)
    f === :size && return Ptr{Int64}(x + 24)
    return getfield(x, f)
end

function Base.getproperty(x::FloatList, f::Symbol)
    r = Ref{FloatList}(x)
    ptr = Base.unsafe_convert(Ptr{FloatList}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{FloatList}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::FloatList, private::Bool=false)
    return (:heap, :inlined, :size, if private
        fieldnames(typeof(x))
    else
        ()
    end...)
end

struct BoolList
    data::NTuple{16,UInt8}
end

function Base.getproperty(x::Ptr{BoolList}, f::Symbol)
    f === :heap && return Ptr{Ptr{Bool}}(x + 0)
    f === :inlined && return Ptr{NTuple{6,Bool}}(x + 0)
    f === :size && return Ptr{Int64}(x + 8)
    return getfield(x, f)
end

function Base.getproperty(x::BoolList, f::Symbol)
    r = Ref{BoolList}(x)
    ptr = Base.unsafe_convert(Ptr{BoolList}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{BoolList}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::BoolList, private::Bool=false)
    return (:heap, :inlined, :size, if private
        fieldnames(typeof(x))
    else
        ()
    end...)
end

struct FloatListRef
    ptr::Ptr{Cfloat}
    size::Int64
end

struct TpuEmbeddingEngineParameters
    parameters::NTuple{8,Ptr{Ptr{FloatListRef}}}
    num_tables::Csize_t
end

struct XLA_Tile
    data::NTuple{56,UInt8}
end

function Base.getproperty(x::Ptr{XLA_Tile}, f::Symbol)
    f === :dimensions && return Ptr{Int64List}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::XLA_Tile, f::Symbol)
    r = Ref{XLA_Tile}(x)
    ptr = Base.unsafe_convert(Ptr{XLA_Tile}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{XLA_Tile}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::XLA_Tile, private::Bool=false)
    return (:dimensions, if private
        fieldnames(typeof(x))
    else
        ()
    end...)
end

struct TileList
    data::NTuple{344,UInt8}
end

function Base.getproperty(x::Ptr{TileList}, f::Symbol)
    f === :heap && return Ptr{Ptr{XLA_Tile}}(x + 0)
    f === :inlined && return Ptr{NTuple{6,XLA_Tile}}(x + 0)
    f === :size && return Ptr{Int64}(x + 336)
    return getfield(x, f)
end

function Base.getproperty(x::TileList, f::Symbol)
    r = Ref{TileList}(x)
    ptr = Base.unsafe_convert(Ptr{TileList}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{TileList}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::TileList, private::Bool=false)
    return (:heap, :inlined, :size, if private
        fieldnames(typeof(x))
    else
        ()
    end...)
end

struct XLA_Layout
    data::NTuple{440,UInt8}
end

function Base.getproperty(x::Ptr{XLA_Layout}, f::Symbol)
    f === :minor_to_major && return Ptr{Int64List}(x + 0)
    f === :tiles && return Ptr{TileList}(x + 56)
    f === :index_primitive_type && return Ptr{Cint}(x + 400)
    f === :pointer_primitive_type && return Ptr{Cint}(x + 404)
    f === :element_size_in_bits && return Ptr{Int64}(x + 408)
    f === :memory_space && return Ptr{Int64}(x + 416)
    f === :dynamic_shape_metadata_prefix_bytes && return Ptr{Int64}(x + 424)
    f === :tail_padding_alignment_in_elements && return Ptr{Int64}(x + 432)
    return getfield(x, f)
end

function Base.getproperty(x::XLA_Layout, f::Symbol)
    r = Ref{XLA_Layout}(x)
    ptr = Base.unsafe_convert(Ptr{XLA_Layout}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{XLA_Layout}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::XLA_Layout, private::Bool=false)
    return (
        :minor_to_major,
        :tiles,
        :index_primitive_type,
        :pointer_primitive_type,
        :element_size_in_bits,
        :memory_space,
        :dynamic_shape_metadata_prefix_bytes,
        :tail_padding_alignment_in_elements,
        if private
            fieldnames(typeof(x))
        else
            ()
        end...,
    )
end

struct XLA_Shape
    data::NTuple{536,UInt8}
end

function Base.getproperty(x::Ptr{XLA_Shape}, f::Symbol)
    f === :element_type && return Ptr{Cint}(x + 0)
    f === :dimensions && return Ptr{Int64List}(x + 8)
    f === :dynamic_dimensions && return Ptr{BoolList}(x + 64)
    f === :tuple_shapes && return Ptr{Ptr{XLA_Shape}}(x + 80)
    f === :ntuple_shapes && return Ptr{Cint}(x + 88)
    f === :has_layout && return Ptr{Bool}(x + 92)
    f === :layout && return Ptr{XLA_Layout}(x + 96)
    return getfield(x, f)
end

function Base.getproperty(x::XLA_Shape, f::Symbol)
    r = Ref{XLA_Shape}(x)
    ptr = Base.unsafe_convert(Ptr{XLA_Shape}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{XLA_Shape}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::XLA_Shape, private::Bool=false)
    return (
        :element_type,
        :dimensions,
        :dynamic_dimensions,
        :tuple_shapes,
        :ntuple_shapes,
        :has_layout,
        :layout,
        if private
            fieldnames(typeof(x))
        else
            ()
        end...,
    )
end

struct XLA_Literal
    data::NTuple{560,UInt8}
end

function Base.getproperty(x::Ptr{XLA_Literal}, f::Symbol)
    f === :buffers && return Ptr{Ptr{Cstring}}(x + 0)
    f === :sizes && return Ptr{Ptr{Csize_t}}(x + 8)
    f === :count && return Ptr{Csize_t}(x + 16)
    f === :shape && return Ptr{XLA_Shape}(x + 24)
    return getfield(x, f)
end

function Base.getproperty(x::XLA_Literal, f::Symbol)
    r = Ref{XLA_Literal}(x)
    ptr = Base.unsafe_convert(Ptr{XLA_Literal}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{XLA_Literal}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::XLA_Literal, private::Bool=false)
    return (:buffers, :sizes, :count, :shape, if private
        fieldnames(typeof(x))
    else
        ()
    end...)
end

struct XLA_ShapeIndex
    indices::NTuple{8,Int64}
    count::Int64
end

struct XLA_ComputationLayout
    data::NTuple{552,UInt8}
end

function Base.getproperty(x::Ptr{XLA_ComputationLayout}, f::Symbol)
    f === :parameter_count && return Ptr{Cint}(x + 0)
    f === :parameter_layouts && return Ptr{Ptr{XLA_Shape}}(x + 8)
    f === :result_layout && return Ptr{XLA_Shape}(x + 16)
    return getfield(x, f)
end

function Base.getproperty(x::XLA_ComputationLayout, f::Symbol)
    r = Ref{XLA_ComputationLayout}(x)
    ptr = Base.unsafe_convert(Ptr{XLA_ComputationLayout}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{XLA_ComputationLayout}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::XLA_ComputationLayout, private::Bool=false)
    return (
        :parameter_count,
        :parameter_layouts,
        :result_layout,
        if private
            fieldnames(typeof(x))
        else
            ()
        end...,
    )
end

struct XLA_HloModuleConfig
    data::NTuple{784,UInt8}
end

function Base.getproperty(x::Ptr{XLA_HloModuleConfig}, f::Symbol)
    f === :seed && return Ptr{UInt64}(x + 0)
    f === :launch_id && return Ptr{Int32}(x + 8)
    f === :replica_count && return Ptr{Int64}(x + 16)
    f === :num_partitions && return Ptr{Int64}(x + 24)
    f === :use_spmd_partitioning && return Ptr{Bool}(x + 32)
    f === :use_auto_spmd_partitioning && return Ptr{Bool}(x + 33)
    f === :auto_spmd_partitioning_mesh_shape && return Ptr{Int64List}(x + 40)
    f === :auto_spmd_partitioning_mesh_ids && return Ptr{Int64List}(x + 96)
    f === :debug_options && return Ptr{TpuSerializedProto}(x + 152)
    f === :has_static_device_assignment && return Ptr{Bool}(x + 168)
    f === :static_device_assignment && return Ptr{TpuSerializedProto}(x + 176)
    f === :has_entry_computation_layout && return Ptr{Bool}(x + 192)
    f === :entry_computation_layout && return Ptr{XLA_ComputationLayout}(x + 200)
    f === :allow_spmd_sharding_propagation_to_parameters && return Ptr{BoolList}(x + 752)
    f === :allow_spmd_sharding_propagation_to_output && return Ptr{BoolList}(x + 768)
    return getfield(x, f)
end

function Base.getproperty(x::XLA_HloModuleConfig, f::Symbol)
    r = Ref{XLA_HloModuleConfig}(x)
    ptr = Base.unsafe_convert(Ptr{XLA_HloModuleConfig}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{XLA_HloModuleConfig}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::XLA_HloModuleConfig, private::Bool=false)
    return (
        :seed,
        :launch_id,
        :replica_count,
        :num_partitions,
        :use_spmd_partitioning,
        :use_auto_spmd_partitioning,
        :auto_spmd_partitioning_mesh_shape,
        :auto_spmd_partitioning_mesh_ids,
        :debug_options,
        :has_static_device_assignment,
        :static_device_assignment,
        :has_entry_computation_layout,
        :entry_computation_layout,
        :allow_spmd_sharding_propagation_to_parameters,
        :allow_spmd_sharding_propagation_to_output,
        if private
            fieldnames(typeof(x))
        else
            ()
        end...,
    )
end

mutable struct SE_HloExecutionProfile end

struct XLA_HloModuleGroup
    proto::TpuSerializedProto
    module_config::Ptr{XLA_HloModuleConfig}
end

struct XLA_HloModule
    data::NTuple{800,UInt8}
end

function Base.getproperty(x::Ptr{XLA_HloModule}, f::Symbol)
    f === :proto && return Ptr{TpuSerializedProto}(x + 0)
    f === :module_config && return Ptr{XLA_HloModuleConfig}(x + 16)
    return getfield(x, f)
end

function Base.getproperty(x::XLA_HloModule, f::Symbol)
    r = Ref{XLA_HloModule}(x)
    ptr = Base.unsafe_convert(Ptr{XLA_HloModule}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{XLA_HloModule}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::XLA_HloModule, private::Bool=false)
    return (:proto, :module_config, if private
        fieldnames(typeof(x))
    else
        ()
    end...)
end

mutable struct XLA_TpuMeshState end

# typedef void ( * XLA_CallbackFn ) ( void * )
const XLA_CallbackFn = Ptr{Cvoid}

# typedef void ( * XLA_StatusCallbackFn ) ( void * , TF_Status * )
const XLA_StatusCallbackFn = Ptr{Cvoid}

mutable struct SE_TpuTopology end

mutable struct SE_TpuTopology_Core end

const SE_TpuTopology_Host = SE_TpuTopology_Core

mutable struct SE_OutsideCompilationParams end

function TfTpu_Initialize(init_library, num_args, args)
    @ccall libxla.TfTpu_Initialize(
        init_library::Bool, num_args::Cint, args::Ptr{Cstring}
    )::Cvoid
end

struct TfTpu_BaseFn
    TfTpu_InitializeFn::Ptr{Cvoid}
end

const TPU_C_API_MAX_INLINED = 6

module Plugin

# When dealing with memory ownership at the Julia FFI boundary, there are two different designs that can be followed:
# - rely on Julia's GC, which requires to...
#   - globally mark/root objects
#   - refer to them via `pointer_from_objref`, `unsafe_pointer_to_objref`
#   - use `mutable` objects only (immutable objects do not have a stable memory address)
# - ...or manually allocate/free memory, which requires to...
#   - use `Libc`
#   - only work with `isbits` objects (i.e. not mutable nor immutable with mutable fields)
#
# After experimenting with both designs, the "globally GC rooted" design was ultimately chosen because...
# - it's simpler and easier to read
# - it's easier to extend from the Julia side
# - safer against memory bugs (as memory is tracked by the GC)
# - almost no overhead compared to manual allocations (only a dynamic dispatch for offering a Julia interface)
#
# This code is designed in such a way that PJRT plugins developed in Julia do not need to interface with the FFI
# boundary. All interaction with the PJRT FFI should be located here and abstract away through an interface that Julia
# plugins can implement elsewhere.
#
# When extending the following code or implementing the interface, take these points into considerations:
# - All objects passed through the XLA/PJRT boundary need to be GC rooted (i.e. use `(un)root!`), which requires objects
#   to be `mutable` in order to have a stable pointer address. Objects are GC rooted/unrooted here and should be
#   transparent to the user.
# - Some Julia objects require as a first field a pointer to their vtable (e.g. Error, Memory, ...). This is required by
#   the PJRT Plugin API. Keep in mind that it should point to the correct implementation for your plugin.
# - If for some PJRT object, the API requires it to return a string that is auto-generated (not a string literal),
#   store the string inside the object so that it's correctly freed (owner is the object).
# - If for some PJRT object, the API requires it to return a list, store the list as a `Vector{X}` where `X` is an
#   abstract type.
#   - The Julia array ABI layout states that an array with abstract eltype is represented as a pointer of pointers
#     (no inline storage). Check https://docs.julialang.org/en/v1/manual/embedding/#Working-with-Arrays
#   - The stored pointers are of type `jl_value_t*` which points to the object data (the type tag is stored before that
#     pointer address). Check https://docs.julialang.org/en/v1/devdocs/object/#Object-layout-(jl_value_t)
#   - Once the object is created, DO NOT resize the array.
# - DO NOT use moving GC (e.g. MMTk) as it would break `pointer_from_objref`.
# - The following objects do not need to be reimplemented on different accelerators: Error, Event.

using Base: unsafe_convert
using ..Reactant: Reactant, WeakKeyIdDict
using Reactant.XLA.CAPI: CAPI, csizeof
using Reactant.MLIR: IR, API

macro pjrt_catch_return(fdef)
    Meta.isexpr(fdef, :function) ||
        error("@pjrt_catch_return must be applied to a `function` definition")

    err = gensym(:err)
    fdef.args[2] = quote
        try
            $(fdef.args[2])
        catch $err
            if $err isa $Error
                return $pointer_from_objref($root!($err))
            else
                # we could return an "internal" error, but throwing makes it easier for Julia REPL sessions
                # return $pointer_from_objref($root!($Error($(CAPI.PJRT_Error_Code_INTERNAL), $sprint($showerror, $err))))
                $rethrow()
            end
        end
        return $ok()
    end
    return esc(fdef)
end

# interface functions
function add_metadata! end
function addressable_device_logical_ids end
function addressable_devices end
function addressable_memories end
function alias_buffer end
function async_tracking_event end
function attributes end
function bitcast! end
function buffercount end
function buffersize end
function canonical_shape end
function clear_memory_stats! end
function codesize end
function compile end
function compile_options end
function compiled_memory_stats end
function copyto end
function cost_analysis end
function debug_string end
function decref! end
function default_device_assignment end
function default_memory end
function default_memory_description end
function description end
function device_assignment end
function device end
function device_descriptions end
function devices end
function dma_map! end
function dma_unmap! end
function donate! end
function dynamic_dims end
function executable end
function execute end
function fingerprint end
function granulesize end
function id end
function incref! end
function initialize end
function isaddressable end
function isdeleted end
function isoncpu end
function kind_id end
function kind end
function load end
function local_hardware_id end
function lookup_addressable_device end
function lookup_device end
function memory end
function memory_descriptions end
function memory_layout end
function memory_space_kind_ids end
function name end
function noutputs end
function npartitions end
function nreplicas end
function opaque_pointer end
function optimized_program end
function output_eltypes end
function output_memory_kinds end
function output_pjrt_eltypes end
function output_sizes end
function parameter_memory_kinds end
function platform_name end
function platform_version end
function poison_execution! end
function process_index end
function ready_event end
function retrieve_buffer! end
function serialize end
function set_buffer_error! end
function to_host_buffer! end
function topology_description end
function total_bytes end
function transfer_data! end
function transfer_literal! end
function unpadded_size end
function update_global_process_info! end

export unimplemented, invalid_argument, resource_exhausted

# GC roots of objects whose pointers are handed to C; objects stay alive until they are unmarked
const gc_roots = IdSet{Any}()
const gc_roots_lock = ReentrantLock()

function root!(obj::T) where {T}
    ismutabletype(T) || error("Only mutable types can be GC rooted")
    @lock gc_roots_lock push!(gc_roots, obj)
    # return Ptr{T}(pointer_from_objref(obj))
    return obj
end

unroot!(ptr::Ptr) = unroot!(unsafe_pointer_to_objref(ptr))
function unroot!(obj)
    @lock gc_roots_lock delete!(gc_roots, obj)
    return nothing
end

function pjrt_serialized_bytes_deleter(ptr::Ptr{Cvoid})::Cvoid
    ptr != C_NULL && unroot!(ptr)
    return nothing
end

## vtables
const vtable_error = Ref{CAPI.PJRT_Error_FunctionTable}()
const vtable_memory = Ref{CAPI.PJRT_Memory_FunctionTable}()

## helpers
const ProcessState = CAPI.PJRT_ProcessState
const ProcessInfo = CAPI.PJRT_ProcessInfo

function pjrt_elsize(T::CAPI.PJRT_Buffer_Type)
    if T in (
        CAPI.PJRT_Buffer_Type_PRED,
        CAPI.PJRT_Buffer_Type_S8,
        CAPI.PJRT_Buffer_Type_U8,
        CAPI.PJRT_Buffer_Type_F8E5M2,
        CAPI.PJRT_Buffer_Type_F8E4M3FN,
        CAPI.PJRT_Buffer_Type_F8E4M3B11FNUZ,
        CAPI.PJRT_Buffer_Type_F8E5M2FNUZ,
        CAPI.PJRT_Buffer_Type_F8E4M3FNUZ,
        CAPI.PJRT_Buffer_Type_F8E4M3,
        CAPI.PJRT_Buffer_Type_F8E3M4,
        CAPI.PJRT_Buffer_Type_F8E8M0FNU,
    )
        1
    elseif T in (
        CAPI.PJRT_Buffer_Type_S16,
        CAPI.PJRT_Buffer_Type_U16,
        CAPI.PJRT_Buffer_Type_F16,
        CAPI.PJRT_Buffer_Type_BF16,
    )
        2
    elseif T in
        (CAPI.PJRT_Buffer_Type_S32, CAPI.PJRT_Buffer_Type_U32, CAPI.PJRT_Buffer_Type_F32)
        4
    elseif T in (
        CAPI.PJRT_Buffer_Type_S64,
        CAPI.PJRT_Buffer_Type_U64,
        CAPI.PJRT_Buffer_Type_F64,
        CAPI.PJRT_Buffer_Type_C64,
    )
        8
    elseif T == CAPI.PJRT_Buffer_Type_C128
        16
    else
        return nothing
    end
end

# TODO move `pjrt_eltype` to specialized pkg exts
pjrt_eltype(T::Type) = error("PJRT does not have a type mapping for $T")
pjrt_eltype(::Type{Bool}) = CAPI.PJRT_Buffer_Type_PRED
pjrt_eltype(::Type{Int8}) = CAPI.PJRT_Buffer_Type_S8
pjrt_eltype(::Type{UInt8}) = CAPI.PJRT_Buffer_Type_U8
# pjrt_eltype(::Type{}) = CAPI.PJRT_Buffer_Type_F8E5M2
# pjrt_eltype(::Type{}) = CAPI.PJRT_Buffer_Type_F8E4M3FN
# pjrt_eltype(::Type{}) = CAPI.PJRT_Buffer_Type_F8E4M3B11FNUZ
# pjrt_eltype(::Type{}) = CAPI.PJRT_Buffer_Type_F8E5M2FNUZ
# pjrt_eltype(::Type{}) = CAPI.PJRT_Buffer_Type_F8E4M3FNUZ
# pjrt_eltype(::Type{}) = CAPI.PJRT_Buffer_Type_F8E4M3
# pjrt_eltype(::Type{}) = CAPI.PJRT_Buffer_Type_F8E3M4
# pjrt_eltype(::Type{}) = CAPI.PJRT_Buffer_Type_F8E8M0FNU
pjrt_eltype(::Type{Int16}) = CAPI.PJRT_Buffer_Type_S16
pjrt_eltype(::Type{UInt16}) = CAPI.PJRT_Buffer_Type_U16
pjrt_eltype(::Type{Float16}) = CAPI.PJRT_Buffer_Type_F16
# pjrt_eltype(::Type{}) = CAPI.PJRT_Buffer_Type_BF16
pjrt_eltype(::Type{Int32}) = CAPI.PJRT_Buffer_Type_S32
pjrt_eltype(::Type{UInt32}) = CAPI.PJRT_Buffer_Type_U32
pjrt_eltype(::Type{Float32}) = CAPI.PJRT_Buffer_Type_F32
pjrt_eltype(::Type{Int64}) = CAPI.PJRT_Buffer_Type_S64
pjrt_eltype(::Type{UInt64}) = CAPI.PJRT_Buffer_Type_U64
pjrt_eltype(::Type{Float64}) = CAPI.PJRT_Buffer_Type_F64
pjrt_eltype(::Type{ComplexF32}) = CAPI.PJRT_Buffer_Type_C64
pjrt_eltype(::Type{ComplexF64}) = CAPI.PJRT_Buffer_Type_C128

## Named values
# read a list of `PJRT_NamedValue`s owned by C into a `Dict`, copying their contents
function unsafe_load_pjrt_named_values(ptr::Ptr{CAPI.PJRT_NamedValue}, n::Integer)
    values = Dict{String,Any}()
    for i in 1:n
        nv = ptr + (i - 1) * sizeof(CAPI.PJRT_NamedValue)
        name = unsafe_string(Ptr{UInt8}(unsafe_load(nv.name)), unsafe_load(nv.name_size))
        type = unsafe_load(nv.type)
        size = unsafe_load(nv.value_size)
        values[name] = if type == CAPI.PJRT_NamedValue_kString
            unsafe_string(Ptr{UInt8}(unsafe_load(nv.string_value)), size)
        elseif type == CAPI.PJRT_NamedValue_kInt64
            unsafe_load(nv.int64_value)
        elseif type == CAPI.PJRT_NamedValue_kInt64List
            copy(unsafe_wrap(Vector{Int64}, unsafe_load(nv.int64_array_value), size))
        elseif type == CAPI.PJRT_NamedValue_kFloat
            unsafe_load(nv.float_value)
        elseif type == CAPI.PJRT_NamedValue_kBool
            unsafe_load(nv.bool_value)
        else
            throw(invalid_argument("named value `$name` has unknown type $type"))
        end
    end
    return values
end

# build the `PJRT_NamedValue`s of the `name => value` pairs in `pairs`; also return the Julia objects they point to,
# which must be kept alive for as long as C uses them
function pjrt_named_values(pairs)
    roots = Any[]
    values = map(collect(pairs)) do (name, value)
        name = String(name)
        push!(roots, name)
        r = Ref{CAPI.PJRT_NamedValue}()
        GC.@preserve r begin
            nv = Base.unsafe_convert(Ptr{CAPI.PJRT_NamedValue}, r)
            unsafe_store!(nv.struct_size, csizeof(CAPI.PJRT_NamedValue))
            unsafe_store!(nv.extension_start, C_NULL)
            unsafe_store!(nv.name, Cstring(pointer(name)))
            unsafe_store!(nv.name_size, sizeof(name))
            if value isa AbstractString
                str = String(value)
                push!(roots, str)
                unsafe_store!(nv.type, CAPI.PJRT_NamedValue_kString)
                unsafe_store!(nv.string_value, Cstring(pointer(str)))
                unsafe_store!(nv.value_size, sizeof(str))
            elseif value isa Bool # before `Integer`, since `Bool <: Integer`
                unsafe_store!(nv.type, CAPI.PJRT_NamedValue_kBool)
                unsafe_store!(nv.bool_value, value)
                unsafe_store!(nv.value_size, 1)
            elseif value isa Integer
                unsafe_store!(nv.type, CAPI.PJRT_NamedValue_kInt64)
                unsafe_store!(nv.int64_value, Int64(value))
                unsafe_store!(nv.value_size, 1)
            elseif value isa AbstractVector{<:Integer}
                list = Vector{Int64}(value)
                push!(roots, list)
                unsafe_store!(nv.type, CAPI.PJRT_NamedValue_kInt64List)
                unsafe_store!(nv.int64_array_value, pointer(list))
                unsafe_store!(nv.value_size, length(list))
            elseif value isa AbstractFloat
                unsafe_store!(nv.type, CAPI.PJRT_NamedValue_kFloat)
                unsafe_store!(nv.float_value, Float32(value))
                unsafe_store!(nv.value_size, 1)
            else
                msg = "value of `$name` has type $(typeof(value)), which can't be a `PJRT_NamedValue`"
                throw(Error(CAPI.PJRT_Error_Code_INTERNAL, msg))
            end
        end
        r[]
    end
    return values, roots
end

include("Error.jl")
include("Event.jl")
include("AbstractPlugin.jl")
include("Memory.jl")
include("DeviceDescription.jl")
include("Device.jl")
include("MemoryLayout.jl")
include("Buffer.jl")
include("Executable.jl")
include("ExecuteContext.jl")
include("LoadedExecutable.jl")
include("Client.jl")
include("TopologyDescription.jl")
include("AsyncHostToDeviceTransferManager.jl")
include("CopyToDeviceStream.jl")
include("Extensions/Extension.jl")

unimplemented_c_hook() = Ptr{CAPI.PJRT_Error}(pointer_from_objref(root!(unimplemented())))

"""
    PluginApi

A `PJRT_Api` function table bound to a specific [`AbstractPlugin`](@ref) and set of [`Extension`](@ref)s. Create it
with [`make_api`](@ref).

!!! warning
It must be kept alive for as long as any client or topology created through it exists.
"""
struct PluginApi
    api::Base.RefValue{CAPI.PJRT_Api}

    # plugin entrypoints
    plugin_initialize::Base.CFunction
    plugin_attributes::Base.CFunction
    client_create::Base.CFunction
    topology_create::Base.CFunction
    topology_deserialize::Base.CFunction

    # keepalive
    extensions::Vector{Extension}
end

function Base.unsafe_convert(::Type{Ptr{CAPI.PJRT_Api}}, x::PluginApi)
    return unsafe_convert(Ptr{CAPI.PJRT_Api}, x.api)
end

function __init__()
    #! format: off
    vtable_error[]= CAPI.PJRT_Error_FunctionTable(
        #= struct_size =# csizeof(CAPI.PJRT_Error_FunctionTable),
        #= instance_size =# csizeof(CAPI.PJRT_Error),
        #= extension_start =# C_NULL,
        #= destroy =# @cfunction(pjrt_error_destroy_obj, Cvoid, (Ptr{Error},)),
        #= message =# @cfunction(pjrt_error_message_obj, Cvoid, (Ptr{Error}, Ptr{Cstring}, Ptr{Csize_t})),
        #= get_code =# @cfunction(pjrt_error_getcode_obj, CAPI.PJRT_Error_Code, (Ptr{Error},)),
        #= for_each_payload =# @cfunction(pjrt_error_for_each_payload_obj, Cvoid, (Ptr{Error}, CAPI.PJRT_Error_PayloadVisitor, Ptr{Cvoid})),
    )

    vtable_memory[] = CAPI.PJRT_Memory_FunctionTable(
        #= struct_size =# csizeof(CAPI.PJRT_Memory_FunctionTable),
        #= extension_start =# C_NULL,
        #= instance_struct_size =# csizeof(CAPI.PJRT_Memory),
        #= get_user_data =# C_NULL,
        #= set_user_data =# C_NULL,
    )
    #! format: on
    return nothing
end

"""
    make_api(plugin::AbstractPlugin; extensions=()) -> PluginApi

Build the `PJRT_Api` function table of `plugin`.

`extensions` is a collection of [`Extension`](@ref) types, which are built as `E(plugin, next)` and advertised to XLA
through the extension chain of the table, in the given order, e.g.
```julia
make_api(MyPlugin(); extensions=[MemoryDescriptionsExtension, FFIExtension])
```
Each extension type can be given only once.

Store the result in a global (e.g. a `const Ref`) set from the plugin's `__init__`, and pass
`unsafe_convert(Ptr{CAPI.PJRT_Api}, api)` to XLA.
"""
function make_api(plugin::P; extensions=()) where {P<:AbstractPlugin}
    extensions = collect(Type{<:Extension}, extensions)
    types = map(extension_type, extensions)
    allunique(types) ||
        throw(ArgumentError("each extension type can be given only once, got $types"))
    Base.issingletontype(P) || throw(
        ArgumentError("plugin must be stateless (a `struct` without fields), got a `$P`"),
    )

    # build the chain back to front, so that each extension gets the pointer to the next one
    exts = Vector{Extension}(undef, length(extensions))
    extension_start = Ptr{CAPI.PJRT_Extension_Base}(C_NULL)
    for i in reverse(eachindex(extensions))
        E = extensions[i]
        ext = E(plugin, extension_start)
        ext isa E || throw(ArgumentError("constructor of $E returned a `$(typeof(ext))`"))
        ismutable(ext) || throw(ArgumentError("$E must be a mutable struct"))
        exts[i] = ext
        extension_start = Ptr{CAPI.PJRT_Extension_Base}(pointer_from_objref(ext))
    end

    # generic handler for unimplemented PJRT API functions
    c_unimplemented = @cfunction(unimplemented_c_hook, Ptr{CAPI.PJRT_Error}, ())
    c_plugin_initialize = @cfunction(
        $(PluginInitialize{P}()),
        Ptr{CAPI.PJRT_Error},
        (Ptr{CAPI.PJRT_Plugin_Initialize_Args},)
    )
    c_plugin_attributes = @cfunction(
        $(PluginAttributes{P}()),
        Ptr{CAPI.PJRT_Error},
        (Ptr{CAPI.PJRT_Plugin_Attributes_Args},)
    )
    c_client_create = @cfunction(
        $(ClientCreate{P}()), Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_Create_Args},)
    )
    c_topology_create = @cfunction(
        $(TopologyDescriptionCreate{P}()),
        Ptr{CAPI.PJRT_Error},
        (Ptr{CAPI.PJRT_TopologyDescription_Create_Args},)
    )
    c_topology_deserialize = @cfunction(
        $(TopologyDescriptionDeserialize{P}()),
        Ptr{CAPI.PJRT_Error},
        (Ptr{CAPI.PJRT_TopologyDescription_Deserialize_Args},)
    )

    #! format: off
    api = Ref(CAPI.PJRT_Api(
        #= struct_size =# csizeof(CAPI.PJRT_Api),
        #= extension_start =# extension_start,
        #= pjrt_api_version =# CAPI.PJRT_Api_Version(
            #= struct_size =# csizeof(CAPI.PJRT_Api_Version),
            #= extension_start =# C_NULL,
            #= major_version =# CAPI.PJRT_API_MAJOR,
            #= minor_version =# CAPI.PJRT_API_MINOR,
        ),
        #= PJRT_Error_Destroy =# @cfunction(pjrt_error_destroy, Cvoid, (Ptr{CAPI.PJRT_Error_Destroy_Args},)),
        #= PJRT_Error_Message =# @cfunction(pjrt_error_message, Cvoid, (Ptr{CAPI.PJRT_Error_Message_Args},)),
        #= PJRT_Error_GetCode =# @cfunction(pjrt_error_getcode, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Error_GetCode_Args},)),
        #= PJRT_Plugin_Initialize =# unsafe_convert(Ptr{Cvoid}, c_plugin_initialize),
        #= PJRT_Plugin_Attributes =# unsafe_convert(Ptr{Cvoid}, c_plugin_attributes),
        #= PJRT_Event_Destroy =# @cfunction(pjrt_event_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Event_Destroy_Args},)),
        #= PJRT_Event_IsReady =# @cfunction(pjrt_event_is_ready, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Event_IsReady_Args},)),
        #= PJRT_Event_Error =# @cfunction(pjrt_event_error, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Event_Error_Args},)),
        #= PJRT_Event_Await =# @cfunction(pjrt_event_await, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Event_Await_Args},)),
        #= PJRT_Event_OnReady =# @cfunction(pjrt_event_on_ready, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Event_OnReady_Args},)),
        #= PJRT_Client_Create =# unsafe_convert(Ptr{Cvoid}, c_client_create),
        #= PJRT_Client_Destroy =# @cfunction(pjrt_client_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_Destroy_Args},)),
        #= PJRT_Client_PlatformName =# @cfunction(pjrt_client_platform_name, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_PlatformName_Args},)),
        #= PJRT_Client_ProcessIndex =# @cfunction(pjrt_client_process_index, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_ProcessIndex_Args},)),
        #= PJRT_Client_PlatformVersion =# @cfunction(pjrt_client_platform_version, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_PlatformVersion_Args},)),
        #= PJRT_Client_Devices =# @cfunction(pjrt_client_devices, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_Devices_Args},)),
        #= PJRT_Client_AddressableDevices =# @cfunction(pjrt_client_addressable_devices, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_AddressableDevices_Args},)),
        #= PJRT_Client_LookupDevice =# @cfunction(pjrt_client_lookup_device, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_LookupDevice_Args},)),
        #= PJRT_Client_LookupAddressableDevice =# @cfunction(pjrt_client_lookup_addressable_device, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_LookupAddressableDevice_Args},)),
        #= PJRT_Client_AddressableMemories =# @cfunction(pjrt_client_addressable_memories, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_AddressableMemories_Args},)),
        #= PJRT_Client_Compile =# @cfunction(pjrt_client_compile, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_Compile_Args},)),
        #= PJRT_Client_DefaultDeviceAssignment =# @cfunction(pjrt_client_default_device_assignment, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_DefaultDeviceAssignment_Args},)),
        #= PJRT_Client_BufferFromHostBuffer =# @cfunction(pjrt_client_buffer_from_host_buffer, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_BufferFromHostBuffer_Args},)),
        #= PJRT_DeviceDescription_Id =# @cfunction(pjrt_device_description_id, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_DeviceDescription_Id_Args},)),
        #= PJRT_DeviceDescription_ProcessIndex =# @cfunction(pjrt_device_description_process_index, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_DeviceDescription_ProcessIndex_Args},)),
        #= PJRT_DeviceDescription_Attributes =# @cfunction(pjrt_device_description_attributes, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_DeviceDescription_Attributes_Args},)),
        #= PJRT_DeviceDescription_Kind =# @cfunction(pjrt_device_description_kind, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_DeviceDescription_Kind_Args},)),
        #= PJRT_DeviceDescription_DebugString =# @cfunction(pjrt_device_description_debug_string, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_DeviceDescription_DebugString_Args},)),
        #= PJRT_DeviceDescription_ToString =# @cfunction(pjrt_device_description_to_string, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_DeviceDescription_ToString_Args},)),
        #= PJRT_Device_GetDescription =# @cfunction(pjrt_device_get_description, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_GetDescription_Args},)),
        #= PJRT_Device_IsAddressable =# @cfunction(pjrt_device_is_addressable, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_IsAddressable_Args},)),
        #= PJRT_Device_LocalHardwareId =# @cfunction(pjrt_device_local_hardware_id, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_LocalHardwareId_Args},)),
        #= PJRT_Device_AddressableMemories =# @cfunction(pjrt_device_addressable_memories, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_AddressableMemories_Args},)),
        #= PJRT_Device_DefaultMemory =# @cfunction(pjrt_device_default_memory, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_DefaultMemory_Args},)),
        #= PJRT_Device_MemoryStats =# @cfunction(pjrt_device_memory_stats, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_MemoryStats_Args},)),
        #= PJRT_Memory_Id =# @cfunction(pjrt_memory_id, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Memory_Id_Args},)),
        #= PJRT_Memory_Kind =# @cfunction(pjrt_memory_kind, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Memory_Kind_Args},)),
        #= PJRT_Memory_DebugString =# @cfunction(pjrt_memory_debug_string, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Memory_DebugString_Args},)),
        #= PJRT_Memory_ToString =# @cfunction(pjrt_memory_to_string, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Memory_ToString_Args},)),
        #= PJRT_Memory_AddressableByDevices =# @cfunction(pjrt_memory_addressable_by_devices, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Memory_AddressableByDevices_Args},)),
        #= PJRT_Executable_Destroy =# @cfunction(pjrt_executable_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_Destroy_Args},)),
        #= PJRT_Executable_Name =# @cfunction(pjrt_executable_name, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_Name_Args},)),
        #= PJRT_Executable_NumReplicas =# @cfunction(pjrt_executable_num_replicas, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_NumReplicas_Args},)),
        #= PJRT_Executable_NumPartitions =# @cfunction(pjrt_executable_num_partitions, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_NumPartitions_Args},)),
        #= PJRT_Executable_NumOutputs =# @cfunction(pjrt_executable_num_outputs, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_NumOutputs_Args},)),
        #= PJRT_Executable_SizeOfGeneratedCodeInBytes =# @cfunction(pjrt_executable_size_of_generated_code_in_bytes, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_SizeOfGeneratedCodeInBytes_Args},)),
        #= PJRT_Executable_GetCostAnalysis =# @cfunction(pjrt_executable_get_cost_analysis, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_GetCostAnalysis_Args},)),
        #= PJRT_Executable_OutputMemoryKinds =# @cfunction(pjrt_executable_output_memory_kinds, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_OutputMemoryKinds_Args},)),
        #= PJRT_Executable_OptimizedProgram =# @cfunction(pjrt_executable_optimized_program, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_OptimizedProgram_Args},)),
        #= PJRT_Executable_Serialize =# @cfunction(pjrt_executable_serialize, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_Serialize_Args},)),
        #= PJRT_LoadedExecutable_Destroy =# @cfunction(pjrt_loaded_executable_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_LoadedExecutable_Destroy_Args},)),
        #= PJRT_LoadedExecutable_GetExecutable =# @cfunction(pjrt_loaded_executable_get_executable, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_LoadedExecutable_GetExecutable_Args},)),
        #= PJRT_LoadedExecutable_AddressableDevices =# @cfunction(pjrt_loaded_executable_addressable_devices, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_LoadedExecutable_AddressableDevices_Args},)),
        #= PJRT_LoadedExecutable_Delete =# @cfunction(pjrt_loaded_executable_delete, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_LoadedExecutable_Delete_Args},)),
        #= PJRT_LoadedExecutable_IsDeleted =# @cfunction(pjrt_loaded_executable_is_deleted, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_LoadedExecutable_IsDeleted_Args},)),
        #= PJRT_LoadedExecutable_Execute =# @cfunction(pjrt_loaded_executable_execute, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_LoadedExecutable_Execute_Args},)),
        #= PJRT_Executable_DeserializeAndLoad =# @cfunction(pjrt_executable_deserialize_and_load, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_DeserializeAndLoad_Args},)),
        #= PJRT_LoadedExecutable_Fingerprint =# c_unimplemented, # deprecated
        #= PJRT_Buffer_Destroy =# @cfunction(pjrt_buffer_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_Destroy_Args},)),
        #= PJRT_Buffer_ElementType =# @cfunction(pjrt_buffer_element_type, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_ElementType_Args},)),
        #= PJRT_Buffer_Dimensions =# @cfunction(pjrt_buffer_dimensions, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_Dimensions_Args},)),
        #= PJRT_Buffer_UnpaddedDimensions =# @cfunction(pjrt_buffer_unpadded_dimensions, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_UnpaddedDimensions_Args},)),
        #= PJRT_Buffer_DynamicDimensionIndices =# @cfunction(pjrt_buffer_dynamic_dimension_indices, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_DynamicDimensionIndices_Args},)),
        #= PJRT_Buffer_GetMemoryLayout =# @cfunction(pjrt_buffer_get_memory_layout, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_GetMemoryLayout_Args},)),
        #= PJRT_Buffer_OnDeviceSizeInBytes =# @cfunction(pjrt_buffer_on_device_size_in_bytes, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_OnDeviceSizeInBytes_Args},)),
        #= PJRT_Buffer_Device =# @cfunction(pjrt_buffer_device, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_Device_Args},)),
        #= PJRT_Buffer_Memory =# @cfunction(pjrt_buffer_memory, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_Memory_Args},)),
        #= PJRT_Buffer_Delete =# @cfunction(pjrt_buffer_delete, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_Delete_Args},)),
        #= PJRT_Buffer_IsDeleted =# @cfunction(pjrt_buffer_is_deleted, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_IsDeleted_Args},)),
        #= PJRT_Buffer_CopyToDevice =# @cfunction(pjrt_buffer_copy_to_device, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_CopyToDevice_Args},)),
        #= PJRT_Buffer_ToHostBuffer =# @cfunction(pjrt_buffer_to_host_buffer, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_ToHostBuffer_Args},)),
        #= PJRT_Buffer_IsOnCpu =# @cfunction(pjrt_buffer_is_on_cpu, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_IsOnCpu_Args},)),
        #= PJRT_Buffer_ReadyEvent =# @cfunction(pjrt_buffer_ready_event, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_ReadyEvent_Args},)),
        #= PJRT_Buffer_UnsafePointer =# @cfunction(pjrt_buffer_unsafe_pointer, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_UnsafePointer_Args},)),
        #= PJRT_Buffer_IncreaseExternalReferenceCount =# @cfunction(pjrt_buffer_increase_external_reference_count, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_IncreaseExternalReferenceCount_Args},)),
        #= PJRT_Buffer_DecreaseExternalReferenceCount =# @cfunction(pjrt_buffer_decrease_external_reference_count, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_DecreaseExternalReferenceCount_Args},)),
        #= PJRT_Buffer_OpaqueDeviceMemoryDataPointer =# @cfunction(pjrt_buffer_opaque_device_memory_data_pointer, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args},)),
        #= PJRT_CopyToDeviceStream_Destroy =# @cfunction(pjrt_copy_to_device_stream_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_CopyToDeviceStream_Destroy_Args},)),
        #= PJRT_CopyToDeviceStream_AddChunk =# @cfunction(pjrt_copy_to_device_stream_add_chunk, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_CopyToDeviceStream_AddChunk_Args},)),
        #= PJRT_CopyToDeviceStream_TotalBytes =# @cfunction(pjrt_copy_to_device_stream_total_bytes, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_CopyToDeviceStream_TotalBytes_Args},)),
        #= PJRT_CopyToDeviceStream_GranuleSize =# @cfunction(pjrt_copy_to_device_stream_granule_size, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_CopyToDeviceStream_GranuleSize_Args},)),
        #= PJRT_CopyToDeviceStream_CurrentBytes =# @cfunction(pjrt_copy_to_device_stream_current_bytes, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_CopyToDeviceStream_CurrentBytes_Args},)),
        #= PJRT_TopologyDescription_Create =# unsafe_convert(Ptr{Cvoid}, c_topology_create),
        #= PJRT_TopologyDescription_Destroy =# @cfunction(pjrt_topology_description_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_TopologyDescription_Destroy_Args},)),
        #= PJRT_TopologyDescription_PlatformName =# @cfunction(pjrt_topology_description_platform_name, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_TopologyDescription_PlatformName_Args},)),
        #= PJRT_TopologyDescription_PlatformVersion =# @cfunction(pjrt_topology_description_platform_version, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_TopologyDescription_PlatformVersion_Args},)),
        #= PJRT_TopologyDescription_GetDeviceDescriptions =# @cfunction(pjrt_topology_description_get_device_descriptions, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_TopologyDescription_GetDeviceDescriptions_Args},)),
        #= PJRT_TopologyDescription_Serialize =# @cfunction(pjrt_topology_description_serialize, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_TopologyDescription_Serialize_Args},)),
        #= PJRT_TopologyDescription_Attributes =# @cfunction(pjrt_topology_description_attributes, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_TopologyDescription_Attributes_Args},)),
        #= PJRT_Compile =# @cfunction(pjrt_compile, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Compile_Args},)),
        #= PJRT_Executable_OutputElementTypes =# @cfunction(pjrt_executable_output_element_types, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_OutputElementTypes_Args},)),
        #= PJRT_Executable_OutputDimensions =# @cfunction(pjrt_executable_output_dimensions, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_OutputDimensions_Args},)),
        #= PJRT_Buffer_CopyToMemory =# @cfunction(pjrt_buffer_copy_to_memory, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_CopyToMemory_Args},)),
        #= PJRT_Client_CreateViewOfDeviceBuffer =# c_unimplemented,
        #= PJRT_Executable_Fingerprint =# @cfunction(pjrt_executable_fingerprint, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_Fingerprint_Args},)),
        #= PJRT_Client_TopologyDescription =# @cfunction(pjrt_client_topology_description, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_TopologyDescription_Args},)),
        #= PJRT_Executable_GetCompiledMemoryStats =# @cfunction(pjrt_executable_get_compiled_memory_stats, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_GetCompiledMemoryStats_Args},)),
        #= PJRT_Memory_Kind_Id =# @cfunction(pjrt_memory_kind_id, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Memory_Kind_Id_Args},)),
        #= PJRT_ExecuteContext_Create =# @cfunction(pjrt_execute_context_create, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_ExecuteContext_Create_Args},)),
        #= PJRT_ExecuteContext_Destroy =# @cfunction(pjrt_execute_context_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_ExecuteContext_Destroy_Args},)),
        #= PJRT_Buffer_CopyRawToHost =# @cfunction(pjrt_buffer_copy_raw_to_host, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_CopyRawToHost_Args},)),
        #= PJRT_AsyncHostToDeviceTransferManager_Destroy =# @cfunction(pjrt_async_host_to_device_transfer_manager_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_Destroy_Args},)),
        #= PJRT_AsyncHostToDeviceTransferManager_TransferData =# @cfunction(pjrt_async_host_to_device_transfer_manager_transfer_data, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_TransferData_Args},)),
        #= PJRT_Client_CreateBuffersForAsyncHostToDevice =# @cfunction(pjrt_client_create_buffers_for_async_host_to_device, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_CreateBuffersForAsyncHostToDevice_Args},)),
        #= PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer =# @cfunction(pjrt_async_host_to_device_transfer_manager_retrieve_buffer, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args},)),
        #= PJRT_AsyncHostToDeviceTransferManager_Device =# @cfunction(pjrt_async_host_to_device_transfer_manager_device, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_Device_Args},)),
        #= PJRT_AsyncHostToDeviceTransferManager_BufferCount =# @cfunction(pjrt_async_host_to_device_transfer_manager_buffer_count, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args},)),
        #= PJRT_AsyncHostToDeviceTransferManager_BufferSize =# @cfunction(pjrt_async_host_to_device_transfer_manager_buffer_size, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args},)),
        #= PJRT_AsyncHostToDeviceTransferManager_SetBufferError =# @cfunction(pjrt_async_host_to_device_transfer_manager_set_buffer_error, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args},)),
        #= PJRT_AsyncHostToDeviceTransferManager_AddMetadata =# @cfunction(pjrt_async_host_to_device_transfer_manager_add_metadata, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args},)),
        #= PJRT_Client_DmaMap =# @cfunction(pjrt_client_dma_map, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_DmaMap_Args},)),
        #= PJRT_Client_DmaUnmap =# @cfunction(pjrt_client_dma_unmap, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_DmaUnmap_Args},)),
        #= PJRT_Client_CreateUninitializedBuffer =# @cfunction(pjrt_client_create_uninitialized_buffer, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_CreateUninitializedBuffer_Args},)),
        #= PJRT_Client_UpdateGlobalProcessInfo =# @cfunction(pjrt_client_update_global_process_info, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_UpdateGlobalProcessInfo_Args},)),
        #= PJRT_TopologyDescription_Deserialize =# unsafe_convert(Ptr{Cvoid}, c_topology_deserialize),
        #= PJRT_Client_CreateAliasBuffer =# @cfunction(pjrt_client_create_alias_buffer, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_CreateAliasBuffer_Args},)),
        #= PJRT_Client_FulfillAliasBuffer =# @cfunction(pjrt_client_fulfill_alias_buffer, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_FulfillAliasBuffer_Args},)),
        #= PJRT_LoadedExecutable_GetDeviceAssignment =# @cfunction(pjrt_loaded_executable_get_device_assignment, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_LoadedExecutable_GetDeviceAssignment_Args},)),
        #= PJRT_Client_CreateErrorBuffer =# @cfunction(pjrt_client_create_error_buffer, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_CreateErrorBuffer_Args},)),
        #= PJRT_AsyncHostToDeviceTransferManager_TransferLiteral =# @cfunction(pjrt_async_host_to_device_transfer_manager_transfer_literal, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args},)),
        #= PJRT_Buffer_CopyRawToHostFuture =# @cfunction(pjrt_buffer_copy_raw_to_host_future, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_CopyRawToHostFuture_Args},)),
        #= PJRT_Device_PoisonExecution =# @cfunction(pjrt_device_poison_execution, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_PoisonExecution_Args},)),
        #= PJRT_Device_CreateAsyncTrackingEvent =# @cfunction(pjrt_device_create_async_tracking_event, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_CreateAsyncTrackingEvent_Args},)),
        #= PJRT_AsyncTrackingEvent_Destroy =# @cfunction(pjrt_async_tracking_event_destroy, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_AsyncTrackingEvent_Destroy_Args},)),
        #= PJRT_Executable_GetCompileOptions =# @cfunction(pjrt_executable_get_compile_options, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_GetCompileOptions_Args},)),
        #= PJRT_Buffer_DonateWithControlDependency =# @cfunction(pjrt_buffer_donate_with_control_dependency, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_DonateWithControlDependency_Args},)),
        #= PJRT_Event_Create =# @cfunction(pjrt_event_create, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Event_Create_Args},)),
        #= PJRT_Event_Set =# @cfunction(pjrt_event_set, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Event_Set_Args},)),
        #= PJRT_Device_GetAttributes =# @cfunction(pjrt_device_get_attributes, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_GetAttributes_Args},)),
        #= PJRT_Client_Load =# @cfunction(pjrt_client_load, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Client_Load_Args},)),
        #= PJRT_LoadedExecutable_AddressableDeviceLogicalIds =# @cfunction(pjrt_loaded_executable_addressable_device_logical_ids, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args},)),
        #= PJRT_Buffer_Bitcast =# @cfunction(pjrt_buffer_bitcast, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Buffer_Bitcast_Args},)),
        #= PJRT_Error_ForEachPayload =# @cfunction(pjrt_error_for_each_payload, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Error_ForEachPayload_Args},)),
        #= PJRT_TopologyDescription_Fingerprint =# @cfunction(pjrt_topology_description_fingerprint, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_TopologyDescription_Fingerprint_Args},)),
        #= PJRT_Executable_ParameterMemoryKinds =# @cfunction(pjrt_executable_parameter_memory_kinds, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Executable_ParameterMemoryKinds_Args},)),
        #= PJRT_Device_ClearMemoryStats =# @cfunction(pjrt_device_clear_memory_stats, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_Device_ClearMemoryStats_Args},)),
        #= PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace =# @cfunction(pjrt_topology_description_make_canonical_shape_for_memory_space, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace_Args},)),
        #= PJRT_TopologyDescription_GetMemorySpaceKindIds =# @cfunction(pjrt_topology_description_get_memory_space_kind_ids, Ptr{CAPI.PJRT_Error}, (Ptr{CAPI.PJRT_TopologyDescription_GetMemorySpaceKindIds_Args},)),
    ))
    #! format: on

    return PluginApi(
        api,
        c_plugin_initialize,
        c_plugin_attributes,
        c_client_create,
        c_topology_create,
        c_topology_deserialize,
        exts,
    )
end

end

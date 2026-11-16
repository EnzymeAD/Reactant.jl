"""
    Profiler

A profiler of a plugin, through which XLA collects the activity of the plugin in a profiling session (e.g.
`Reactant.Profiler.with_profiler`). XLA creates one per profiling session, and then starts it, stops it, collects its
data and destroys it, in that order.

A `PLUGIN_Profiler*` points to a `Profiler`, which is GC rooted from its creation until `PLUGIN_Profiler_Destroy`.

### Interface

- [`Profiler(plugin, options)`](@ref Profiler(::AbstractPlugin, ::Reactant.Proto.tensorflow.ProfileOptions))
- [`start!`](@ref)
- [`stop!`](@ref)
- [`collect_data`](@ref collect_data(::Profiler))
- [`consume!`](@ref consume!(::Profiler))
- [`serialize`](@ref serialize(::Profiler, data)) (required if `consume!` is implemented)
"""
abstract type Profiler end

# profiler errors are `Error`s
function profiler_error_destroy(
    args::Ptr{CAPI.PLUGIN_Profiler_Error_Destroy_Args}
)::Cvoid
    pjrt_error_destroy_obj(Ptr{Error}(unsafe_load(args.error)))
    return
end

function profiler_error_message(
    args::Ptr{CAPI.PLUGIN_Profiler_Error_Message_Args}
)::Cvoid
    pjrt_error_message_obj(
        Ptr{Error}(unsafe_load(args.error)), args.message, args.message_size
    )
    return
end

function profiler_error_get_code(
    args::Ptr{CAPI.PLUGIN_Profiler_Error_GetCode_Args}
)::Ptr{CAPI.PLUGIN_Profiler_Error}
    code = pjrt_error_getcode_obj(Ptr{Error}(unsafe_load(args.error)))
    unsafe_store!(args.code, Cint(code))
    return ok()
end

"""
    Profiler(plugin::AbstractPlugin, options::Reactant.Proto.tensorflow.ProfileOptions) -> Profiler

Return a new profiler of `plugin` for a profiling session with `options` (e.g. `options.device_tracer_level`, which is
`0` if device tracing is disabled).

Backs `PLUGIN_Profiler_Create`.
"""
Profiler(::AbstractPlugin, ::Reactant.Proto.tensorflow.ProfileOptions)

struct ProfilerCreate{P<:AbstractPlugin} end

@pjrt_catch_return function (::ProfilerCreate{P})(
    args::Ptr{CAPI.PLUGIN_Profiler_Create_Args}
)::Ptr{CAPI.PLUGIN_Profiler_Error} where {P}
    bytes = unsafe_wrap(
        Vector{UInt8}, Ptr{UInt8}(unsafe_load(args.options)), unsafe_load(args.options_size)
    )
    options = Reactant.ProtoUtils.proto_from_bytes(
        Reactant.Proto.tensorflow.ProfileOptions, bytes
    )
    profiler = root!(Profiler(P(), options)::Profiler)
    unsafe_store!(args.profiler, Ptr{CAPI.PLUGIN_Profiler}(pointer_from_objref(profiler)))
    return ok()
end

@pjrt_catch_return function profiler_destroy(
    args::Ptr{CAPI.PLUGIN_Profiler_Destroy_Args}
)::Ptr{CAPI.PLUGIN_Profiler_Error}
    ptr = unsafe_load(args.profiler)
    ptr != C_NULL && unroot!(ptr)
    return ok()
end

"""
    start!(profiler::Profiler)

Start recording the activity of the plugin.

Backs `PLUGIN_Profiler_Start`.
"""
function start! end

@pjrt_catch_return function profiler_start(
    args::Ptr{CAPI.PLUGIN_Profiler_Start_Args}
)::Ptr{CAPI.PLUGIN_Profiler_Error}
    profiler = unsafe_pointer_to_objref(unsafe_load(args.profiler))::Profiler
    start!(profiler)
    return ok()
end

"""
    stop!(profiler::Profiler)

Stop recording the activity of the plugin.

Backs `PLUGIN_Profiler_Stop`.
"""
function stop! end

@pjrt_catch_return function profiler_stop(
    args::Ptr{CAPI.PLUGIN_Profiler_Stop_Args}
)::Ptr{CAPI.PLUGIN_Profiler_Error}
    profiler = unsafe_pointer_to_objref(unsafe_load(args.profiler))::Profiler
    stop!(profiler)
    return ok()
end

"""
    collect_data(profiler::Profiler) -> Reactant.Proto.tensorflow.profiler.XSpace

Return the activity recorded by `profiler` between [`start!`](@ref) and [`stop!`](@ref), whose planes XLA adds to the
`XSpace` of the profiling session (e.g. a `/device:...` plane with an `XLA Ops` line holding an event per op).

It is called at most once per profiler, after `stop!`.

Defaults to throwing an `UNIMPLEMENTED` [`Error`](@ref), which XLA logs without failing the profiling session. Since
XLA creates the profiler in every profiling session once a client of the plugin exists, a missing method would
otherwise throw through XLA and break profiling for every backend.

Backs `PLUGIN_Profiler_CollectData`.
"""
collect_data(::Profiler) =
    throw(unimplemented("`collect_data` is not implemented by this profiler"))

# C reads the serialized data after the call returns, so it is kept for as long as the profiler is alive
const profiler_data_cache = WeakKeyIdDict{Profiler,Vector{UInt8}}()

# XLA reads the buffer returned by the first call (with a null `buffer`), but the C API documents a second call with a
# buffer of `buffer_size_in_bytes` bytes for the data to be copied into, so both are supported
@pjrt_catch_return function profiler_collect_data(
    args::Ptr{CAPI.PLUGIN_Profiler_CollectData_Args}
)::Ptr{CAPI.PLUGIN_Profiler_Error}
    profiler = unsafe_pointer_to_objref(unsafe_load(args.profiler))::Profiler
    bytes = get!(profiler_data_cache, profiler) do
        xspace = collect_data(profiler)::Reactant.Proto.tensorflow.profiler.XSpace
        Reactant.ProtoUtils.proto_to_bytes(xspace)
    end
    buffer = unsafe_load(args.buffer)
    if buffer == C_NULL
        unsafe_store!(args.buffer, pointer(bytes))
    elseif buffer != pointer(bytes)
        unsafe_copyto!(buffer, pointer(bytes), length(bytes))
    end
    unsafe_store!(args.buffer_size_in_bytes, length(bytes))
    return ok()
end

"""
    consume!(profiler::Profiler) -> Any

Take the activity recorded by `profiler` since it started or since the last call, without stopping it, and return it in
any form (e.g. the raw events) for [`serialize(profiler, data)`](@ref serialize) to turn into an `XSpace` later. Used
in continuous profiling, which consumes the data periodically from a background thread.

Defaults to throwing an `UNIMPLEMENTED` [`Error`](@ref), which XLA takes as the plugin not supporting it.

Backs `PLUGIN_Profiler_Consume`.
"""
function consume!(::Profiler)
    throw(unimplemented("`consume!` is not implemented by this profiler"))
end

# A `PLUGIN_Profiler_ConsumeResult*` points to a `Ref{Any}` holding the data returned by `consume!`, which is GC rooted
# until `PLUGIN_Profiler_ConsumeResult_Destroy`
@pjrt_catch_return function profiler_consume(
    args::Ptr{CAPI.PLUGIN_Profiler_Consume_Args}
)::Ptr{CAPI.PLUGIN_Profiler_Error}
    profiler = unsafe_pointer_to_objref(unsafe_load(args.profiler))::Profiler
    result = root!(Ref{Any}(consume!(profiler)))
    unsafe_store!(
        args.result, Ptr{CAPI.PLUGIN_Profiler_ConsumeResult}(pointer_from_objref(result))
    )
    return ok()
end

function profiler_consume_result_destroy(
    args::Ptr{CAPI.PLUGIN_Profiler_ConsumeResult_Destroy_Args}
)::Cvoid
    ptr = unsafe_load(args.consume_result)
    ptr != C_NULL && unroot!(ptr)
    return
end

"""
    serialize(profiler::Profiler, data) -> Reactant.Proto.tensorflow.profiler.XSpace

Return the activity in `data`, as returned by [`consume!(profiler)`](@ref consume!), as an `XSpace` whose planes XLA
adds to the `XSpace` of the profiling session (like the one of [`collect_data`](@ref)).

It is called at most once per `data`, possibly from a different thread than `consume!`.

Backs `PLUGIN_Profiler_Serialize`.
"""
serialize(::Profiler, data)

# C reads the serialized data after the call returns, so it is kept for as long as the consume result is alive
const consume_result_data_cache = WeakKeyIdDict{Base.RefValue{Any},Vector{UInt8}}()

@pjrt_catch_return function profiler_serialize(
    args::Ptr{CAPI.PLUGIN_Profiler_Serialize_Args}
)::Ptr{CAPI.PLUGIN_Profiler_Error}
    profiler = unsafe_pointer_to_objref(unsafe_load(args.profiler))::Profiler
    result = unsafe_pointer_to_objref(unsafe_load(args.consume_result))::Base.RefValue{Any}
    bytes = get!(consume_result_data_cache, result) do
        xspace = serialize(profiler, result[])::Reactant.Proto.tensorflow.profiler.XSpace
        Reactant.ProtoUtils.proto_to_bytes(xspace)
    end
    unsafe_store!(args.serialized_bytes, pointer(bytes))
    unsafe_store!(args.serialized_size, length(bytes))
    return ok()
end

"""
    ProfilerExtension(plugin::AbstractPlugin, next::Ptr{CAPI.PJRT_Extension_Base} = C_NULL)

Advertises `PJRT_Profiler_Extension`, through which XLA creates a [`Profiler`](@ref) of the plugin in every profiling
session and adds the data it collects to the session. Pass it to [`make_api`](@ref).

`plugin` must implement `Profiler(plugin, options)`, and its [`Profiler`](@ref)s `start!`, `stop!` and `collect_data`
(and optionally `consume!` and `serialize`).

!!! note
XLA registers the profiler of the plugin when a client is created from its `PJRT_Api`, so profiling sessions started
before then don't include it.
"""
mutable struct ProfilerExtension <: Extension
    const ftable::CAPI.PJRT_Profiler_Extension

    # keepalive of the `PLUGIN_Profiler_Api` that `ftable` points to, and of the `@cfunction` in it
    const api::Base.RefValue{CAPI.PLUGIN_Profiler_Api}
    const create::Base.CFunction

    function ProfilerExtension(
        ::P, next::Ptr{CAPI.PJRT_Extension_Base}=C_NULL
    ) where {P<:AbstractPlugin}
        create = @cfunction(
            $(ProfilerCreate{P}()),
            Ptr{CAPI.PLUGIN_Profiler_Error},
            (Ptr{CAPI.PLUGIN_Profiler_Create_Args},)
        )

        #! format: off
        api = Ref(CAPI.PLUGIN_Profiler_Api(
            #= struct_size =# csizeof(CAPI.PLUGIN_Profiler_Api),
            #= priv =# C_NULL,
            #= PLUGIN_Profiler_Error_Destroy =# @cfunction(profiler_error_destroy, Cvoid, (Ptr{CAPI.PLUGIN_Profiler_Error_Destroy_Args},)),
            #= PLUGIN_Profiler_Error_Message =# @cfunction(profiler_error_message, Cvoid, (Ptr{CAPI.PLUGIN_Profiler_Error_Message_Args},)),
            #= PLUGIN_Profiler_Error_GetCode =# @cfunction(profiler_error_get_code, Ptr{CAPI.PLUGIN_Profiler_Error}, (Ptr{CAPI.PLUGIN_Profiler_Error_GetCode_Args},)),
            #= PLUGIN_Profiler_Create =# unsafe_convert(Ptr{Cvoid}, create),
            #= PLUGIN_Profiler_Destroy =# @cfunction(profiler_destroy, Ptr{CAPI.PLUGIN_Profiler_Error}, (Ptr{CAPI.PLUGIN_Profiler_Destroy_Args},)),
            #= PLUGIN_Profiler_Start =# @cfunction(profiler_start, Ptr{CAPI.PLUGIN_Profiler_Error}, (Ptr{CAPI.PLUGIN_Profiler_Start_Args},)),
            #= PLUGIN_Profiler_Stop =# @cfunction(profiler_stop, Ptr{CAPI.PLUGIN_Profiler_Error}, (Ptr{CAPI.PLUGIN_Profiler_Stop_Args},)),
            #= PLUGIN_Profiler_CollectData =# @cfunction(profiler_collect_data, Ptr{CAPI.PLUGIN_Profiler_Error}, (Ptr{CAPI.PLUGIN_Profiler_CollectData_Args},)),
            #= PLUGIN_Profiler_Consume =# @cfunction(profiler_consume, Ptr{CAPI.PLUGIN_Profiler_Error}, (Ptr{CAPI.PLUGIN_Profiler_Consume_Args},)),
            #= PLUGIN_Profiler_ConsumeResult_Destroy =# @cfunction(profiler_consume_result_destroy, Cvoid, (Ptr{CAPI.PLUGIN_Profiler_ConsumeResult_Destroy_Args},)),
            #= PLUGIN_Profiler_Serialize =# @cfunction(profiler_serialize, Ptr{CAPI.PLUGIN_Profiler_Error}, (Ptr{CAPI.PLUGIN_Profiler_Serialize_Args},)),
        ))
        ftable = CAPI.PJRT_Profiler_Extension(
            CAPI.PJRT_Extension_Base(csizeof(CAPI.PJRT_Profiler_Extension), extension_type(ProfilerExtension), next),
            unsafe_convert(Ptr{CAPI.PLUGIN_Profiler_Api}, api),
            #= traceme_context_id =# 0,
        )
        #! format: on
        return new(ftable, api, create)
    end
end

extension_type(::Type{ProfilerExtension}) = CAPI.PJRT_Extension_Type_Profiler

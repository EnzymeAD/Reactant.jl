module Metal

# Apple GPUs through the `jax-graft` PJRT plugin (https://github.com/dfm/jax-graft), which
# runs XLA's GPU compiler with Metal as its target. The plugin library ships in the
# `metal-pjrt-core` wheel; the rest of `jax-graft` is the JAX frontend, whose job is done
# here (client creation options) and in `Compiler.Metal.legalize!`, which rewrites the ops
# that XLA:GPU can't compile on Metal into the plugin's custom calls.

using Reactant: Reactant
using Scratch: @get_scratch!
using Downloads: Downloads
using p7zip_jll: p7zip
using FileWatching: mkpidlock
using Libdl: Libdl

using ..Registration: register_backend

# The PJRT platform name reported by the plugin, i.e. `XLA.platform_name(client)`.
const PLATFORM_NAME = "mtl"

const METAL_PJRT_CORE_VERSION = "0.0.5"
const METAL_PJRT_CORE_WHEEL = "https://files.pythonhosted.org/packages/10/4e/3fb98e68eaf3bef237b594831623d75503e419b716591d5fed94c80f2f92/metal_pjrt_core-0.0.5-py3-none-macosx_26_0_arm64.whl"
# `metal_pjrt_frontend_abi_version()`: the client options and FFI targets used by Reactant
const METAL_PJRT_FRONTEND_ABI_VERSION = 2

const metal_pjrt_plugin_dir = Ref{Union{Nothing,String}}(nothing)
const metal_pjrt_plugin_name = "pjrt_c_api_mtl_plugin.dylib"

function __init__()
    if Sys.isapple() && Sys.ARCH === :aarch64 && !Reactant.precompiling()
        # Below the CPU: Metal has no Float64, so it is opt-in through
        # `Reactant.set_default_backend("metal")` (or `"gpu"`).
        register_backend(
            "metal";
            priority=50,
            pjrt_initialize_function=make_pjrt_client,
            ifrt_initialize_function=make_ifrt_client,
            preinitialize_setup_function=setup_metal_pjrt_plugin!,
        )
    end
end

function setup_metal_pjrt_plugin!()
    plugin_dir_from_env = get(ENV, "METAL_PJRT_PLUGIN_DIR", nothing)
    if plugin_dir_from_env !== nothing && ispath(plugin_dir_from_env)
        metal_pjrt_plugin_dir[] = plugin_dir_from_env
    else
        metal_pjrt_plugin_dir[] = @get_scratch!(
            "pjrt_plugin_metal_$(METAL_PJRT_CORE_VERSION)"
        )
    end
    download_metal_pjrt_plugin_if_needed(metal_pjrt_plugin_dir[])
    return nothing
end

get_metal_pjrt_plugin_dir() = metal_pjrt_plugin_dir[]

function get_metal_pjrt_plugin_path()
    return joinpath(get_metal_pjrt_plugin_dir(), metal_pjrt_plugin_name)
end

function download_metal_pjrt_plugin_if_needed(dir=nothing)
    dir === nothing && (dir = get_metal_pjrt_plugin_dir())
    @assert dir !== nothing "metal_pjrt_plugin_dir is not set!"

    metal_pjrt_plugin_path = joinpath(dir, metal_pjrt_plugin_name)
    if isfile(metal_pjrt_plugin_path)
        @debug "Metal PJRT plugin already found in '$(metal_pjrt_plugin_path)', nothing \
                to do"
    else
        mkpidlock(joinpath(dir, "download_metal_pjrt_plugin.lock")) do
            if !isfile(metal_pjrt_plugin_path)
                @debug "Downloading the Metal PJRT plugin from '$(METAL_PJRT_CORE_WHEEL)'"
                mktempdir() do tmp_dir
                    zip_file_path = joinpath(tmp_dir, "metal-pjrt-core.zip")
                    Downloads.download(METAL_PJRT_CORE_WHEEL, zip_file_path)
                    run(
                        pipeline(
                            `$(p7zip()) x -tzip -o$(tmp_dir) -- $(zip_file_path)`, devnull
                        ),
                    )
                    mv(
                        joinpath(tmp_dir, "metal_pjrt_core", metal_pjrt_plugin_name),
                        metal_pjrt_plugin_path,
                    )
                end
            end
        end
        @assert isfile(metal_pjrt_plugin_path)
    end
end

# The plugin is XLA's GPU PJRT client; these are jax-graft's client creation options (see
# `jax_graft.initialize`): XLA's pass-through allocator over the runtime's buffer cache
# instead of a preallocated BFC pool, and no staging of host transfers (unified memory).
const CLIENT_CREATE_OPTIONS = (
    "platform_name" => "METAL",
    "allocator" => "platform",
    "preallocate" => false,
    "memory_fraction" => 1.0f0,
    "visible_devices" => Int64[0],
    "should_stage_host_to_device_transfers" => false,
)

const metal_pjrt_api = Ref{Ptr{Cvoid}}(C_NULL)
const plugin_client_create = Ref{Ptr{Cvoid}}(C_NULL)
const client_create_options = Ref{Any}(nothing)

function named_value!(ptr::Ptr, name::String, value)
    CAPI = Reactant.XLA.CAPI
    ptr.struct_size = CAPI.csizeof(CAPI.PJRT_NamedValue)
    ptr.extension_start = C_NULL
    ptr.name = Base.unsafe_convert(Cstring, name)
    ptr.name_size = sizeof(name)
    if value isa String
        ptr.type = CAPI.PJRT_NamedValue_kString
        ptr.string_value = Base.unsafe_convert(Cstring, value)
        ptr.value_size = sizeof(value)
    elseif value isa Bool
        ptr.type = CAPI.PJRT_NamedValue_kBool
        ptr.bool_value = value
        ptr.value_size = 1
    elseif value isa Float32
        ptr.type = CAPI.PJRT_NamedValue_kFloat
        ptr.float_value = value
        ptr.value_size = 1
    elseif value isa Vector{Int64}
        ptr.type = CAPI.PJRT_NamedValue_kInt64List
        ptr.int64_array_value = pointer(value)
        ptr.value_size = length(value)
    else
        error("Unsupported PJRT_NamedValue: $(typeof(value))")
    end
    return nothing
end

# `PJRT_Client_Create` of the plugin with `CLIENT_CREATE_OPTIONS`, as Reactant creates
# plugin clients without options
function client_create(args::Ptr{Cvoid})
    T = Reactant.XLA.CAPI.PJRT_Client_Create_Args
    num_options = Ptr{Csize_t}(args + fieldoffset(T, Base.fieldindex(T, :num_options)))
    if unsafe_load(num_options) == 0
        options = client_create_options[]
        create_options = Ptr{Ptr{Cvoid}}(
            args + fieldoffset(T, Base.fieldindex(T, :create_options))
        )
        unsafe_store!(create_options, pointer(options))
        unsafe_store!(num_options, length(options))
    end
    return ccall(plugin_client_create[], Ptr{Cvoid}, (Ptr{Cvoid},), args)
end

function load_metal_pjrt_api()
    metal_pjrt_api[] == C_NULL || return metal_pjrt_api[]

    handle = Libdl.dlopen(get_metal_pjrt_plugin_path())
    abi_version = ccall(Libdl.dlsym(handle, :metal_pjrt_frontend_abi_version), Cint, ())
    @assert abi_version == METAL_PJRT_FRONTEND_ABI_VERSION "The Metal PJRT plugin in \
                                                            $(get_metal_pjrt_plugin_dir()) \
                                                            has frontend ABI version \
                                                            $(abi_version), expected \
                                                            $(METAL_PJRT_FRONTEND_ABI_VERSION)."

    CAPI = Reactant.XLA.CAPI
    options = Vector{CAPI.PJRT_NamedValue}(undef, length(CLIENT_CREATE_OPTIONS))
    for (i, (name, value)) in enumerate(CLIENT_CREATE_OPTIONS)
        named_value!(pointer(options, i), name, value)
    end
    client_create_options[] = options

    # Copy the plugin's `PJRT_Api` table and route `PJRT_Client_Create` through
    # `client_create`
    api = ccall(Libdl.dlsym(handle, :GetPjrtApi), Ptr{Cvoid}, ())
    api_size = unsafe_load(Ptr{Csize_t}(api))
    api_copy = Libc.malloc(api_size)
    unsafe_copyto!(Ptr{UInt8}(api_copy), Ptr{UInt8}(api), api_size)
    offset = fieldoffset(CAPI.PJRT_Api, Base.fieldindex(CAPI.PJRT_Api, :PJRT_Client_Create))
    plugin_client_create[] = unsafe_load(Ptr{Ptr{Cvoid}}(api + offset))
    unsafe_store!(
        Ptr{Ptr{Cvoid}}(api_copy + offset),
        @cfunction(client_create, Ptr{Cvoid}, (Ptr{Cvoid},)),
    )

    metal_pjrt_api[] = api_copy
    return api_copy
end

function make_pjrt_client(;
    node_id::Integer=0,
    num_nodes::Integer=1,
    distributed_runtime_client=nothing,
    allowed_devices::Union{Nothing,Vector{Int}}=nothing,
)
    @assert node_id == 0 "`make_pjrt_client` does not support node_id"
    @assert num_nodes == 1 "`make_pjrt_client` does not support num_nodes > 1"
    @assert distributed_runtime_client === nothing "`make_pjrt_client` does not \
                                                    support distributed_runtime_client"

    if allowed_devices !== nothing
        @debug "MetalClient doesn't support allowed_devices. Ignoring the kwarg."
    end

    return Reactant.XLA.PJRT.MakeClientFromApi(load_metal_pjrt_api(), PLATFORM_NAME)
end

function make_ifrt_client(;
    node_id::Integer=0,
    num_nodes::Integer=1,
    distributed_runtime_client=nothing,
    allowed_devices::Union{Nothing,Vector{Int}}=nothing,
)
    if allowed_devices !== nothing
        @debug "MetalClient doesn't support allowed_devices. Ignoring the kwarg."
    end

    return Reactant.XLA.IFRT.MakeIFRTPJRTClient(
        Reactant.XLA.PJRT.MakeClientFromApi(load_metal_pjrt_api(), PLATFORM_NAME),
        PLATFORM_NAME;
        node_id,
        num_nodes,
        distributed_runtime_client,
    )
end

end

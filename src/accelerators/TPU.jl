module TPU

using Reactant: Reactant
using EnumX: @enumx
using Scratch: @get_scratch!
using HTTP: HTTP
using Downloads: Downloads
using p7zip_jll: p7zip

const libtpu_dir = Ref{Union{Nothing,String}}(nothing)
const RUNNING_IN_CLOUD_TPU_VM = Ref(false)

const LIBTPU_VERSION = "0.0.28.dev20251027"
const LIBTPU_SO = "libtpu-$(replace(string(LIBTPU_VERSION), '.' => '_')).so"

function __init__()
    @static if !Sys.isapple()
        if !Reactant.precompiling() && has_tpu()
            setup_libtpu!()
            cloud_tpu_init!()
        end
    end
end

function setup_libtpu!()
    path_from_env = get(ENV, "TPU_LIBRARY_PATH", nothing)
    if path_from_env !== nothing && ispath(path_from_env)
        libtpu_dir[] = path_from_env
    else
        libtpu_dir[] = @get_scratch!("libtpu")
    end
    download_libtpu_if_needed(libtpu_dir[])
    return nothing
end

get_libtpu_dir() = libtpu_dir[]

get_libtpu_path() = joinpath(get_libtpu_dir(), LIBTPU_SO)

function download_libtpu_if_needed(path=nothing)
    path === nothing && (path = get_libtpu_dir())
    @assert path !== nothing "libtpu_dir is not set!"

    libtpu_path = joinpath(path, LIBTPU_SO)
    if !isfile(libtpu_path)
        zip_file_path = joinpath(path, "tpu.zip")
        tmp_dir = joinpath(path, "tmp")
        Downloads.download(
            "https://storage.googleapis.com/libtpu-nightly-releases/wheels/libtpu/libtpu-0.0.33.dev20251224+nightly-cp314-cp314-manylinux_2_31_x86_64.whl",
            zip_file_path,
        )
        run(pipeline(`$(p7zip()) x -tzip -o$(tmp_dir) -- $(zip_file_path)`, devnull))
        mv(joinpath(tmp_dir, "libtpu", "libtpu.so"), libtpu_path)
        rm(tmp_dir; recursive=true)
        rm(zip_file_path; recursive=true)
    end
end

force_tpu_init() = haskey(ENV, "REACTANT_FORCE_TPU_INIT")

# https://github.com/jax-ml/jax/blob/152099ee0ef31119f16f4c2dac50d84fcb1575ef/jax/_src/hardware_utils.py#L19-L55
const _GOOGLE_PCI_VENDOR_ID = "0x1ae0"

@enumx TPUVersion begin
    Unknown
    v2
    v3
    plc
    v4
    v5p
    v5e
    v6e
end

const _TPU_PCI_DEVICE_IDS = Dict(
    "0x0027" => TPUVersion.v2,
    "0x0056" => TPUVersion.plc,
    "0x005e" => TPUVersion.v4,
    "0x0062" => TPUVersion.v5p,
    "0x0063" => TPUVersion.v5e,
    "0x006f" => TPUVersion.v6e,
)

has_tpu() = first(num_available_tpu_chips_and_device_id()) > 0

function num_available_tpu_chips_and_device_id()
    Sys.islinux() || return 0, TPUVersion.Unknown

    devices_dir = "/sys/bus/pci/devices/"
    isdir(devices_dir) || return 0, TPUVersion.Unknown

    try
        num_chips = 0
        tpu_version = TPUVersion.Unknown
        for path in readdir(devices_dir; join=true, sort=false)
            if strip(read(joinpath(path, "vendor"), String)) != _GOOGLE_PCI_VENDOR_ID
                continue
            end

            device_id = strip(read(joinpath(path, "device"), String))
            if haskey(_TPU_PCI_DEVICE_IDS, device_id)
                num_chips += 1
                tpu_version = _TPU_PCI_DEVICE_IDS[device_id]
            end
        end
        return num_chips, tpu_version
    catch ex
        @warn "failed to query PCI device information" maxlog = 1 exception = (
            ex, catch_backtrace()
        )
    end

    return 0, TPUVersion.Unknown
end

function transparent_hugepages_enabled()
    # See https://docs.kernel.org/admin-guide/mm/transhuge.html for more
    # information about transparent huge pages.
    path = "/sys/kernel/mm/transparent_hugepage/enabled"
    return ispath(path) && strip(read(path, String)) == "[always] madvise never"
end

function cloud_tpu_init!()
    libtpu_dir = get_libtpu_dir()
    num_tpu_chips, tpu_version = num_available_tpu_chips_and_device_id()
    if tpu_version != TPUVersion.Unknown &&
        tpu_version ≥ TPUVersion.v5e &&
        !transparent_hugepages_enabled()
        @warn "Transparent hugepages are not enabled. TPU runtime startup and \
               shutdown time should be significantly improved on TPU v5e and newer. \
               If not already set, you may need to enable transparent hugepages in \
               your VM image (sudo sh -c \"echo always > \
               /sys/kernel/mm/transparent_hugepage/enabled\")"
    end

    if (libtpu_dir === nothing || num_tpu_chips == 0) && !force_tpu_init()
        return nothing
    end

    RUNNING_IN_CLOUD_TPU_VM[] = true

    # Set environment variables
    ENV["GRPC_VERBOSITY"] = get(ENV, "GRPC_VERBOSITY", "ERROR")
    ENV["TPU_ML_PLATFORM"] = get(ENV, "TPU_ML_PLATFORM", "Reactant")
    ENV["TPU_ML_PLATFORM_VERSION"] = get(
        ENV, "TPU_ML_PLATFORM_VERSION", pkgversion(Reactant)
    )
    ENV["ENABLE_RUNTIME_UPTIME_TELEMETRY"] = get(
        ENV, "ENABLE_RUNTIME_UPTIME_TELEMETRY", "1"
    )

    # Check and update LIBTPU_INIT_ARGS
    if !occursin("--xla_tpu_use_enhanced_launch_barrier", get(ENV, "LIBTPU_INIT_ARGS", ""))
        ENV["LIBTPU_INIT_ARGS"] =
            get(ENV, "LIBTPU_INIT_ARGS", "") * " --xla_tpu_use_enhanced_launch_barrier=true"
    end

    # Set tensorstore variables
    ENV["TENSORSTORE_CURL_LOW_SPEED_TIME_SECONDS"] = get(
        ENV, "TENSORSTORE_CURL_LOW_SPEED_TIME_SECONDS", "60"
    )
    ENV["TENSORSTORE_CURL_LOW_SPEED_LIMIT_BYTES"] = get(
        ENV, "TENSORSTORE_CURL_LOW_SPEED_LIMIT_BYTES", "256"
    )
    return nothing
end

const _TPU_METADATA_RESPONSE_CODE_SUCCESS = 200

function get_metadata(key)
    # Based on https://github.com/tensorflow/tensorflow/pull/40317
    gce_metadata_endpoint =
        "http://" * get(ENV, "GCE_METADATA_IP", "metadata.google.internal")
    retry_count = 0
    retry_seconds = 0.500
    api_resp = nothing

    while retry_count < 6
        try
            api_resp = HTTP.get(
                "$(gce_metadata_endpoint)/computeMetadata/v1/instance/attributes/$(key)",
                ["Metadata-Flavor" => "Google"];
                connect_timeout=60,
                readtimeout=60,
            )

            HTTP.status(api_resp) == _TPU_METADATA_RESPONSE_CODE_SUCCESS && break
        catch err
            @warn "Error while trying to get metadata['$(key)']. Tried \
                   [$(retry_count) / 6] times" err
        end

        retry_count += 1
        sleep(retry_seconds)
    end

    if api_resp === nothing
        throw(ErrorException("Getting metadata['$(key)'] failed for 6 tries"))
    end

    return String(api_resp.body), HTTP.status(api_resp)
end

function get_tpu_env_value(key)
    haskey(ENV, key) && return ENV[key]

    tpu_env_data = first(get_metadata("tpu-env"))
    key_value_pairs = split(tpu_env_data, "\n")
    for key_value_pair in key_value_pairs
        # Typical line is MEGASCALE_NUM_SLICES: '2'
        if contains(key_value_pair, ':')
            row_key, value = split(key_value_pair, ':'; limit=2)
            strip(row_key) == key && return strip(strip(value), '\'')
        end
    end
    return nothing
end

has_megascale_address() = get_tpu_env_value("MEGASCALE_COORDINATOR_ADDRESS") !== nothing

end

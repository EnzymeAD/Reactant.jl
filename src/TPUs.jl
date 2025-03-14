module TPUUtils

using Reactant: Reactant
using EnumX: @enumx
using Scratch: @get_scratch!

const libtpu_dir = Ref{Union{Nothing,String}}(nothing)
const RUNNING_IN_CLOUD_TPU_VM = Ref(false)

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

function download_libtpu_if_needed(path)
    @assert path !== nothing "libtpu_dir is not set!"
    if !isfile(path * "/libtpu.so")
        Downloads.download(
            "https://storage.googleapis.com/libtpu-nightly-releases/wheels/libtpu-nightly/libtpu_nightly-0.1.dev20250313+nightly-py3-none-manylinux_2_31_x86_64.whl",
            path * "/tpu.zip",
        )
        run(`unzip -qq $(path*"/tpu.zip") -d $(path)/tmp`)
        run(`mv $(path)/tmp/libtpu/libtpu.so $(path)/libtpu.so`)
        rm(path * "/tmp"; recursive=true)
        rm(path * "/tpu.zip"; recursive=true)
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
        ENV, "TPU_ML_PLATFORM_VERSION", version.__version__
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

end

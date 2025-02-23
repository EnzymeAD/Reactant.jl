module Distributed

using ..Reactant: Reactant

const initialized = Ref(false)

function initialize(;
    coordinator_address::Union{Nothing,String}=nothing,
    num_processes::Union{Nothing,Integer}=nothing,
    process_id::Union{Nothing,Integer}=nothing,
    local_gpu_device_ids::Union{Nothing,Vector{Int}}=nothing,
    initialization_timeout_in_seconds::Integer=300,
    kwargs...,
)
    @assert !initialized[] "`Distributed.initialize` has already been called"

    (coordinator_address, num_processes, process_id, local_gpu_device_ids) = auto_detect_unset_distributed_params(;
        coordinator_address,
        num_processes,
        process_id,
        local_gpu_device_ids,
        initialization_timeout_in_seconds,
    )

    @debug "Detected Reactant distributed params" coordinator_address num_processes process_id local_gpu_device_ids

    Reactant.XLA.update_global_state!(;
        coordinator_address, num_processes, process_id, local_gpu_device_ids, kwargs...
    )

    @debug "New Global State" Reactant.XLA.global_state

    initialized[] = true
    return nothing
end

abstract type AbstractClusterEnvDetector end

abstract type AbstractOMPIClusterEnvDetector <: AbstractClusterEnvDetector end

struct OpenMPIORTEEnvDetector <: AbstractOMPIClusterEnvDetector end
struct OpenMPIPMIXEnvDetector <: AbstractOMPIClusterEnvDetector end

struct MPIEnvDetector <: AbstractClusterEnvDetector end

# Based on https://github.com/jax-ml/jax/blob/b0117366686ab084d38ad2657d9a2ae3a581ca7e/jax/_src/clusters/cluster.py

is_env_present(::AbstractClusterEnvDetector) = false

function get_coordinator_address end
function get_process_count end
function get_process_id end
function get_local_process_id end

function auto_detect_unset_distributed_params(;
    detector_list=[OpenMPIORTEEnvDetector(), OpenMPIPMIXEnvDetector(), MPIEnvDetector()],
    coordinator_address::Union{Nothing,String}=nothing,
    num_processes::Union{Nothing,Integer}=nothing,
    process_id::Union{Nothing,Integer}=nothing,
    local_gpu_device_ids::Union{Nothing,Vector{Int}}=nothing,
    initialization_timeout_in_seconds::Integer=300,
)
    if all(
        Base.Fix2(!==, nothing),
        (coordinator_address, num_processes, process_id, local_gpu_device_ids),
    )
        return coordinator_address, num_processes, process_id, local_gpu_device_ids
    end

    idx = findfirst(is_env_present, detector_list)
    if idx === nothing
        error("Couldn't find a functional cluster environment detector. Attempted to use: \
               $(detector_list)")
    end

    detector = detector_list[idx]

    @debug "Detected cluster environment" detector

    if coordinator_address === nothing
        coordinator_address = get_coordinator_address(
            detector, initialization_timeout_in_seconds
        )
    end

    if num_processes === nothing
        num_processes = get_process_count(detector)
    end

    if process_id === nothing
        process_id = get_process_id(detector)
    end

    if local_gpu_device_ids === nothing
        local_gpu_device_ids = [get_local_process_id(detector)]
    end

    return coordinator_address, num_processes, process_id, local_gpu_device_ids
end

# OpenMPIORTEEnvDetector & OpenMPIPMIXEnvDetector
# Based on https://github.com/jax-ml/jax/blob/b0117366686ab084d38ad2657d9a2ae3a581ca7e/jax/_src/clusters/ompi_cluster.py and adapted for latest OpenMPI versions
const _ORTE_URI = "OMPI_MCA_orte_hnp_uri"
const _PMIX_SERVER_URI = (
    "PMIX_SERVER_URI2",
    "PMIX_SERVER_URI3",
    "PMIX_SERVER_URI4",
    "PMIX_SERVER_URI41",
    "PMIX_SERVER_URI21",
)
const _OMPI_PROCESS_COUNT = "OMPI_COMM_WORLD_SIZE"
const _OMPI_PROCESS_ID = "OMPI_COMM_WORLD_RANK"
const _OMPI_LOCAL_PROCESS_ID = "OMPI_COMM_WORLD_LOCAL_RANK"

is_env_present(::OpenMPIORTEEnvDetector) = haskey(ENV, _ORTE_URI)
is_env_present(::OpenMPIPMIXEnvDetector) = any(Base.Fix1(haskey, ENV), _PMIX_SERVER_URI)

function get_coordinator_address(::OpenMPIORTEEnvDetector, ::Integer)
    orte_uri = ENV[_ORTE_URI]

    job_id = parse(Int, split(orte_uri, '.'; limit=2)[1])
    port = job_id % 2^12 + (65535 - 2^12 + 1)

    launcher_ip_match = match(r"tcp://(.+?)[,:]|tcp6://\[(.+?)[,\]]", orte_uri)

    @assert launcher_ip_match !== nothing "Could not parse coordinator IP address from \
                                           Open MPI environment."

    launcher_ip = launcher_ip_match.captures[findfirst(
        !isnothing, launcher_ip_match.captures
    )]
    return "$(launcher_ip):$(port)"
end

function get_coordinator_address(::OpenMPIPMIXEnvDetector, ::Integer)
    varname = findfirst(Base.Fix1(haskey, ENV), _PMIX_SERVER_URI)
    pmix_uri = ENV[_PMIX_SERVER_URI[varname]]

    job_id = parse(Int, split(split(pmix_uri, '-'; limit=3)[3], "@"; limit=2)[1])
    port = job_id % 2^12 + (65535 - 2^12 + 1)

    launcher_ip_match = match(r"tcp4://(.+?):|tcp6://\[(.+?)\]", pmix_uri)

    @assert launcher_ip_match !== nothing "Could not parse coordinator IP address from \
                                           Open MPI environment."

    launcher_ip = launcher_ip_match.captures[findfirst(
        !isnothing, launcher_ip_match.captures
    )]

    return "$(launcher_ip):$(port)"
end

get_process_count(::AbstractOMPIClusterEnvDetector) = parse(Int, ENV[_OMPI_PROCESS_COUNT])

get_process_id(::AbstractOMPIClusterEnvDetector) = parse(Int, ENV[_OMPI_PROCESS_ID])

function get_local_process_id(::AbstractOMPIClusterEnvDetector)
    return parse(Int, ENV[_OMPI_LOCAL_PROCESS_ID])
end

end

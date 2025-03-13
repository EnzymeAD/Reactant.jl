module Distributed

using ..Reactant: Reactant

const initialized = Ref(false)

function initialize(;
    coordinator_address::Union{Nothing,String}=nothing,
    num_processes::Union{Nothing,Integer}=nothing,
    process_id::Union{Nothing,Integer}=nothing,
    single_gpu_per_process::Bool=true,
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
        single_gpu_per_process,
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

struct SlurmEnvDetector <: AbstractClusterEnvDetector end

# Based on https://github.com/jax-ml/jax/blob/b0117366686ab084d38ad2657d9a2ae3a581ca7e/jax/_src/clusters/cluster.py

is_env_present(::AbstractClusterEnvDetector) = false

function get_coordinator_address end
function get_process_count end
function get_process_id end
function get_local_process_id end

function auto_detect_unset_distributed_params(;
    detector_list=[
        SlurmEnvDetector(),
        OpenMPIORTEEnvDetector(),
        MPIEnvDetector(),
        # Keep this at the end since parsing for this is a bit flaky
        OpenMPIPMIXEnvDetector(),
    ],
    coordinator_address::Union{Nothing,String}=nothing,
    num_processes::Union{Nothing,Integer}=nothing,
    process_id::Union{Nothing,Integer}=nothing,
    local_gpu_device_ids::Union{Nothing,Vector{Int}}=nothing,
    initialization_timeout_in_seconds::Integer=300,
    single_gpu_per_process::Bool=true,
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

    if local_gpu_device_ids === nothing && single_gpu_per_process
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
const _PMIX_NAMESPACE = "PMIX_NAMESPACE"
const _PRTERUN = "PRTE_LAUNCHED"
const _PMIX_VERSION = "PMIX_VERSION"
const _OMPI_PROCESS_COUNT = "OMPI_COMM_WORLD_SIZE"
const _OMPI_PROCESS_ID = "OMPI_COMM_WORLD_RANK"
const _OMPI_LOCAL_PROCESS_ID = "OMPI_COMM_WORLD_LOCAL_RANK"

is_env_present(::OpenMPIORTEEnvDetector) = haskey(ENV, _ORTE_URI)
is_env_present(::OpenMPIPMIXEnvDetector) = haskey(ENV, _PMIX_NAMESPACE)

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

function _throw_pmix_env_error(msg)
    msg = msg * " Open an issue on Reactant with the relevant PMIX Enviroment Variables \
                 (you might want to obfuscate identifiable variables from this log \
                 before opening an issue)\n\n"
    for (var, val) in [var => val for (var, val) in ENV if startswith(var, "PMIX")]
        msg *= "    * $var => $val.\n"
    end
    return error(msg)
end

function get_coordinator_address(::OpenMPIPMIXEnvDetector, ::Integer)
    pmix_version = parse(VersionNumber, ENV[_PMIX_VERSION])
    pmix_uri = ENV[_PMIX_SERVER_URI[findfirst(Base.Fix1(haskey, ENV), _PMIX_SERVER_URI)]]
    @debug "PMIX VERSION: $(pmix_version)"
    if v"5" ≤ pmix_version < v"6"
        return get_coordinator_address_pmixv5(pmix_uri)
    elseif v"2" ≤ pmix_version < v"4"
        return get_coordinator_address_pmixv2_or_3(pmix_uri)
    else
        _throw_pmix_env_error("Unsupported PMIX version: $(pmix_version).")
    end
end

function get_coordinator_address_pmixv2_or_3(pmix_uri)
    pre_semicolon = first(split(pmix_uri, ";"))
    if startswith(pre_semicolon, "pmix-server.")
        job_id = parse(Int, first(split(last(split(pre_semicolon, '.'; limit=2)))))
    elseif contains(pre_semicolon, ".")
        job_id = parse(Int, first(split(pre_semicolon, '.')))
    else
        _throw_pmix_env_error("Could not parse coordinator address from Open MPI \
                               environment.")
    end
    return get_coordinator_address_from_pmix_uri(pmix_uri, job_id)
end

function get_coordinator_address_pmixv5(pmix_uri)
    job_id = parse(Int, first(split(last(split(pmix_uri, '-'; limit=3)), "@"; limit=2)))
    return get_coordinator_address_from_pmix_uri(pmix_uri, job_id)
end

function get_coordinator_address_from_pmix_uri(pmix_uri, job_id)
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

# SlurmEnvDetector
# Based on https://github.com/jax-ml/jax/blob/d89835acbacec938971400d6fa54ea6dd5efe76c/jax/_src/clusters/slurm_cluster.py#L3
const _SLURM_JOB_ID = "SLURM_JOB_ID"
const _SLURM_NODELIST = "SLURM_STEP_NODELIST"
const _SLURM_PROCESS_COUNT = "SLURM_NTASKS"
const _SLURM_PROCESS_ID = "SLURM_PROCID"
const _SLURM_LOCAL_PROCESS_ID = "SLURM_LOCALID"
const _SLURM_NUM_NODES = "SLURM_STEP_NUM_NODES"

is_env_present(::SlurmEnvDetector) = haskey(ENV, _SLURM_JOB_ID)

function get_coordinator_address(::SlurmEnvDetector, ::Integer)
    port = parse(Int, ENV[_SLURM_JOB_ID]) % 2^12 + (65535 - 2^12 + 1)

    # Parse the first hostname of the job
    # If we are looking for 'node001',
    # node_list potential formats are 'node001', 'node001,host2',
    # 'node[001-0015],host2', and 'node[001,007-015],host2'.
    node_list = ENV[_SLURM_NODELIST]
    ind = findfirst(Base.Fix2(in, (',', '[')), node_list)
    ind = isnothing(ind) ? length(node_list) + 1 : ind

    if ind == length(node_list) + 1 || node_list[ind] == ','
        # 'node001' or 'node001,host2'
        return "$(node_list[1:ind-1]):$(port)"
    else
        # 'node[001-0015],host2' or 'node[001,007-015],host2'
        prefix = node_list[1:(ind - 1)]
        suffix = node_list[(ind + 1):end]
        ind2 = findfirst(Base.Fix2(in, (',', '-')), suffix)
        ind2 = isnothing(ind2) ? length(suffix) : ind2
        return "$(prefix)$(suffix[1:ind2-1]):$(port)"
    end
end

get_process_count(::SlurmEnvDetector) = parse(Int, ENV[_SLURM_PROCESS_COUNT])

get_process_id(::SlurmEnvDetector) = parse(Int, ENV[_SLURM_PROCESS_ID])

get_local_process_id(::SlurmEnvDetector) = parse(Int, ENV[_SLURM_LOCAL_PROCESS_ID])

end

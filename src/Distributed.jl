module Distributed

using ..Reactant: Reactant, Accelerators, Sharding, XLA
using ReactantCore: ReactantCore
using Sockets: Sockets, IPv4, getaddrinfo

const initialized = Ref(false)

"""
    local_rank()

Returns the local rank of the current process.
"""
local_rank() = Reactant.XLA.global_state.process_id

"""
    num_processes()

Returns the number of processes.
"""
num_processes() = Reactant.XLA.global_state.num_processes

"""
    is_initialized()

Returns `true` if the distributed environment has been initialized.
"""
is_initialized() = initialized[]

function initialize(;
    coordinator_address::Union{Nothing,String}=nothing,
    num_processes::Union{Nothing,Integer}=nothing,
    process_id::Union{Nothing,Integer}=nothing,
    single_gpu_per_process::Bool=true,
    local_gpu_device_ids::Union{Nothing,Vector{Int}}=nothing,
    initialization_timeout_in_seconds::Integer=300,
    kwargs...,
)
    if Reactant.XLA.runtime() isa Val{:PJRT}
        @warn "Attempting to using Reactant Distributed functionality with PJRT runtime. \
               This will never be properly supported. Switch to using IFRT runtime by \
               adding a `xla_runtime` preference with value \"IFRT\""
    end

    if isinteractive()
        @warn "Reactant.Distributed.initialize() should not be called in interactive mode. \
               Use Reactant.Distributed.initialize() in a script instead."
    end

    @assert !initialized[] "`Distributed.initialize` has already been called"

    (coordinator_address, num_processes, process_id, local_gpu_device_ids) = auto_detect_unset_distributed_params(;
        coordinator_address,
        num_processes,
        process_id,
        local_gpu_device_ids,
        initialization_timeout_in_seconds,
        single_gpu_per_process,
    )

    Reactant.XLA.claim_gpu_device_mapping!(:distributed)

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

abstract type AbstractCloudTPUEnvDetector <: AbstractClusterEnvDetector end

struct GceTPUCluster <: AbstractCloudTPUEnvDetector end
struct GkeTPUCluster <: AbstractCloudTPUEnvDetector end

# Based on https://github.com/jax-ml/jax/blob/b0117366686ab084d38ad2657d9a2ae3a581ca7e/jax/_src/clusters/cluster.py

is_env_present(::AbstractClusterEnvDetector) = false

function get_coordinator_address end
function get_process_count end
function get_process_id end

get_local_process_id(::AbstractClusterEnvDetector) = nothing

function auto_detect_unset_distributed_params(;
    detector_list=[
        SlurmEnvDetector(),
        OpenMPIORTEEnvDetector(),
        MPIEnvDetector(),
        # Keep this at the end since parsing for this is a bit flaky
        OpenMPIPMIXEnvDetector(),
        # Cloud TPU environments
        GkeTPUCluster(),
        GceTPUCluster(),
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
        detected_local_process_id = get_local_process_id(detector)
        if detected_local_process_id !== nothing
            local_gpu_device_ids = [detected_local_process_id]
        end
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

# TPU Environment Detectors
# Based on https://github.com/jax-ml/jax/blob/d89835acbacec938971400d6fa54ea6dd5efe76c/jax/_src/clusters/cloud_tpu_cluster.py

const _TPU_COORDINATOR_PORT = "8476"

function get_coordinator_address(
    env::AbstractCloudTPUEnvDetector, timeout_in_seconds::Integer
)
    coordinator_address = if Accelerators.TPU.has_megascale_address()
        Accelerators.TPU.get_tpu_env_value("MEGASCALE_COORDINATOR_ADDRESS")
    else
        first(_get_worker_list_in_slice(env))
    end
    coordinator_address = split(coordinator_address, ':')[1]
    @debug "TPU Cluster using coordinator address: $(coordinator_address)"
    _wait_for_coordinator(env, coordinator_address, timeout_in_seconds)
    return "$(coordinator_address):$(_TPU_COORDINATOR_PORT)"
end

function _wait_for_coordinator(
    env::AbstractCloudTPUEnvDetector,
    coordinator_address::AbstractString,
    timeout_in_seconds::Integer,
)
    coordinator_found = false
    max_time = time() + timeout_in_seconds
    coordinator_retry_secs = 5
    pid = get_process_id(env)
    while !coordinator_found && time() < max_time
        try
            ip_address = getaddrinfo(coordinator_address, IPv4)
            @debug "[PID $(pid)] Found coordinator with address $(coordinator_address)"
            return nothing
        catch err
            @debug "[PID $(pid)] Error while trying to connect to coordinator_address \
                    $(coordinator_address). Retrying in $(coordinator_retry_secs) \
                    seconds." err
            sleep(coordinator_retry_secs)
        end
    end
    return error(
        "Failed to recognize coordinator_address $(coordinator_address) \
        after $(timeout_in_seconds) seconds. Please check if the address is correct."
    )
end

function get_process_count(env::AbstractCloudTPUEnvDetector)
    processes_per_slice = length(_get_worker_list_in_slice(env))
    num_slices = _get_num_slices(env)
    total_process_count = processes_per_slice * num_slices
    @debug "Total process count of $(total_process_count) = $(processes_per_slice) \
            processes per slice and $(num_slices) slices"
    return total_process_count
end

function get_process_id(env::AbstractCloudTPUEnvDetector)
    process_id_in_slice = _get_process_id_in_slice(env)
    slice_id = _get_slice_id(env)
    processes_per_slice = length(_get_worker_list_in_slice(env))
    process_id = process_id_in_slice + slice_id * processes_per_slice
    @debug "Process ID of $(process_id) generated by within-slice \
            id $(process_id_in_slice) and slice id $(slice_id)"
    return process_id
end

function _get_num_slices(::AbstractCloudTPUEnvDetector)
    Accelerators.TPU.has_megascale_address() || return 1
    return parse(Int, Accelerators.TPU.get_tpu_env_value("MEGASCALE_NUM_SLICES"))
end

function _get_slice_id(::AbstractCloudTPUEnvDetector)
    Accelerators.TPU.has_megascale_address() || return 0
    return parse(Int, Accelerators.TPU.get_tpu_env_value("MEGASCALE_SLICE_ID"))
end

function _get_process_id_in_slice end
function _get_worker_list_in_slice end

## GceTPUCluster

function is_env_present(::GceTPUCluster)
    if !Accelerators.TPU.has_tpu()
        @debug "Did not detect cloud TPU VM"
        return false
    end

    if Accelerators.TPU.skip_mds_query()
        @debug "TPU_SKIP_MDS_QUERY is set to True, so it's probably not a GCE TPU cluster."
        return false
    end

    metadata_response, metadata_code = Accelerators.TPU.get_metadata("agent-worker-number")
    if metadata_code == Accelerators.TPU._TPU_METADATA_RESPONSE_CODE_SUCCESS
        @debug "Gce Tpu Cluster detected for Reactant Distributed System"
        return true
    else
        @debug "Did not detect Gce Tpu Cluster since agent-worker-number is not set in \
                metadata"
        @debug "Metadata code: $metadata_code"
        @debug "Metadata response: $metadata_response"
        return false
    end
end

function _get_process_id_in_slice(::GceTPUCluster)
    return parse(Int, first(Accelerators.TPU.get_metadata("agent-worker-number")))
end

function _get_worker_list_in_slice(::GceTPUCluster)
    workers = split(first(Accelerators.TPU.get_metadata("worker-network-endpoints")), ',')
    return [split(w, ':')[3] for w in workers]
end

## GkeTPUCluster

function is_env_present(::GkeTPUCluster)
    if (
        Accelerators.TPU.has_tpu() &&
        (haskey(ENV, "TPU_WORKER_HOSTNAMES") || haskey(ENV, "TPU_PROCESS_ADDRESSES"))
    )
        @debug "Detected GKE TPU cluster for Reactant Distributed System"
        return true
    end

    if !Accelerators.TPU.has_tpu()
        @debug "Did not detect cloud TPU VM"
        return false
    end

    @debug "TPU_WORKER_HOSTNAMES or TPU_PROCESS_ADDRESSES is not set, so it's not a GKE TPU cluster."
    return false
end

function _get_process_id_in_slice(::GkeTPUCluster)
    @assert haskey(ENV, "TPU_WORKER_ID") "TPU_WORKER_ID is not set in the environment."
    @debug "[GETPID $(getpid())] TPU_WORKER_ID is set: $(ENV["TPU_WORKER_ID"])"
    return parse(Int, ENV["TPU_WORKER_ID"])
end

function _get_worker_list_in_slice(::GkeTPUCluster)
    if haskey(ENV, "TPU_PROCESS_ADDRESSES")
        @debug "[GETPID $(getpid())] TPU_PROCESS_ADDRESSES is set: $(ENV["TPU_PROCESS_ADDRESSES"])"
        return split(ENV["TPU_PROCESS_ADDRESSES"], ',')
    end
    @debug "[GETPID $(getpid())] TPU_WORKER_HOSTNAMES is set: $(ENV["TPU_WORKER_HOSTNAMES"])"
    return split(ENV["TPU_WORKER_HOSTNAMES"], ',')
end

# Collective utility functions (eager / non-traced)

"""
    avg

Sentinel reduction operator for [`allreduce!`](@ref) and [`reduce!`](@ref) that averages
values across all participating processes.
"""
function avg end

function _check_not_traced(x, fn_name::Symbol)
    if ReactantCore.within_compile() ||
        x isa Union{Reactant.TracedRArray,Reactant.TracedRNumber} ||
        (x isa AbstractArray && eltype(x) <: Reactant.TracedRNumber) ||
        (x isa Base.RefValue && x[] isa Union{Reactant.TracedRArray,Reactant.TracedRNumber})
        throw(
            ArgumentError(
                "Distributed.$(fn_name) cannot be used inside traced functions or on traced values",
            ),
        )
    end
    return nothing
end

function _validate_root(root::Integer, np::Integer)
    if !(0 ≤ root < np)
        throw(
            ArgumentError(
                "Invalid root process rank $(root) for num_processes = $(np) (expected 0 ≤ root < $(np))",
            ),
        )
    end
    return Int(root)
end

_buffer_eltype(x::AbstractArray) = eltype(x)
_buffer_eltype(::Base.RefValue{T}) where {T} = T
_buffer_eltype(::Reactant.AbstractConcreteNumber{T}) where {T} = T

_buffer_size(x::AbstractArray) = size(x)
_buffer_size(::Base.RefValue) = ()
_buffer_size(::Reactant.AbstractConcreteNumber) = ()

function _check_compatible_buffers(sendbuf, recvbuf, fn_name::Symbol)
    if _buffer_eltype(sendbuf) != _buffer_eltype(recvbuf)
        throw(
            ArgumentError(
                "Distributed.$(fn_name): sendbuf eltype $(_buffer_eltype(sendbuf)) does not match recvbuf eltype $(_buffer_eltype(recvbuf))",
            ),
        )
    end
    if _buffer_size(sendbuf) != _buffer_size(recvbuf)
        throw(
            ArgumentError(
                "Distributed.$(fn_name): sendbuf size $(_buffer_size(sendbuf)) does not match recvbuf size $(_buffer_size(recvbuf))",
            ),
        )
    end
    return nothing
end

function _to_host_array(x::Array{T,N}) where {T<:Reactant.ReactantPrimitive,N}
    return x
end
function _to_host_array(x::AbstractArray{T,N}) where {T<:Reactant.ReactantPrimitive,N}
    return convert(Array{T,N}, x)
end
function _to_host_array(x::Base.RefValue{T}) where {T<:Reactant.ReactantPrimitive}
    return fill(x[])
end
function _to_host_array(
    x::Reactant.AbstractConcreteNumber{T}
) where {T<:Reactant.ReactantPrimitive}
    return fill(Reactant.to_number(x))
end

function _copy_local_buffer!(recvbuf::AbstractArray, sendbuf::AbstractArray)
    recvbuf === sendbuf && return recvbuf
    if recvbuf isa Reactant.AbstractConcreteArray && sendbuf isa Array
        copyto!(recvbuf, sendbuf)
    elseif recvbuf isa Array && sendbuf isa Reactant.AbstractConcreteArray
        copyto!(recvbuf, sendbuf)
    elseif recvbuf isa Reactant.AbstractConcreteArray ||
        sendbuf isa Reactant.AbstractConcreteArray
        copyto!(recvbuf, _to_host_array(sendbuf))
    else
        copyto!(recvbuf, sendbuf)
    end
    return recvbuf
end

function _copy_local_buffer!(recvbuf::Base.RefValue{T}, sendbuf) where {T}
    recvbuf === sendbuf && return recvbuf
    recvbuf[] = only(_to_host_array(sendbuf))
    return recvbuf
end

function _copy_local_buffer!(
    recvbuf::Reactant.ConcreteIFRTNumber{T}, sendbuf
) where {T<:Reactant.ReactantPrimitive}
    recvbuf === sendbuf && return recvbuf
    val = only(_to_host_array(sendbuf))
    tmp = Reactant.ConcreteIFRTNumber{T}(
        val; client=XLA.client(recvbuf), device=XLA.device(recvbuf), recvbuf.sharding
    )
    recvbuf.data = tmp.data
    return recvbuf
end

function _copy_local_buffer!(
    recvbuf::Reactant.ConcretePJRTNumber{T}, sendbuf
) where {T<:Reactant.ReactantPrimitive}
    recvbuf === sendbuf && return recvbuf
    val = only(_to_host_array(sendbuf))
    tmp = Reactant.ConcretePJRTNumber{T}(
        val; client=XLA.client(recvbuf), device=XLA.device(recvbuf), recvbuf.sharding
    )
    recvbuf.data = tmp.data
    return recvbuf
end

function _copy_local_buffer!(recvbuf::AbstractArray, sendbuf)
    copyto!(recvbuf, _to_host_array(sendbuf))
    return recvbuf
end

function _infer_collective_client(client, sendbuf, recvbuf)
    client !== nothing && return client
    if sendbuf isa Union{Reactant.AbstractConcreteArray,Reactant.AbstractConcreteNumber}
        return XLA.client(sendbuf)
    end
    if recvbuf isa Union{Reactant.AbstractConcreteArray,Reactant.AbstractConcreteNumber}
        return XLA.client(recvbuf)
    end
    return XLA.default_backend()
end

function _bcast_select_kernel(x::AbstractArray{T,M}, ::Val{idx}) where {T,M,idx}
    return copy(selectdim(x, M, idx))
end

function _reduce_op_kernel(x::AbstractArray{T,M}, op::F) where {T,M,F}
    return dropdims(reduce(op, x; dims=M); dims=M)
end

function _divide_by_count(s::AbstractArray{T}, n::Integer) where {T<:Integer}
    return div.(s, T(n))
end
function _divide_by_count(s::AbstractArray{T}, n::Integer) where {T}
    denom = convert(real(T), n)
    return s ./ denom
end

function _reduce_avg_kernel(x::AbstractArray{T,M}) where {T,M}
    s = dropdims(sum(x; dims=M); dims=M)
    return _divide_by_count(s, size(x, M))
end

function _assemble_process_array(
    theclient::XLA.IFRT.Client, local_arr::Array{T,N}, np::Int
) where {T<:Reactant.ReactantPrimitive,N}
    devs = sort(XLA.devices(theclient); by=XLA.device_ordinal)
    ndevs = length(devs)
    if ndevs % np != 0
        error(
            "Total IFRT device count ($(ndevs)) must be divisible by num_processes ($(np))"
        )
    end
    devs_per_proc = ndevs ÷ np
    naddr = count(XLA.is_addressable, devs)
    if naddr != devs_per_proc
        error(
            "Expected $(devs_per_proc) addressable IFRT devices per process, got $(naddr)"
        )
    end
    dev_ids = Vector{Int64}(undef, ndevs)
    for i in 1:ndevs
        dev_ids[i] = Int64(XLA.device_ordinal(devs[i]))
    end
    mesh = if devs_per_proc == 1
        Sharding.Mesh(dev_ids, (:proc,))
    else
        Sharding.Mesh(reshape(dev_ids, devs_per_proc, np), (:local_dev, :proc))
    end
    sharding = Sharding.NamedSharding(mesh, (ntuple(Returns(nothing), N)..., :proc))
    stacked = reshape(local_arr, size(local_arr)..., 1)
    global_shape = (size(local_arr)..., np)
    addr_map = Vector{Int64}[collect(Int64, 1:naddr)]
    assembled = Reactant.ConcreteIFRTArray(
        Array{T,N + 1}[stacked], global_shape, addr_map; client=theclient, sharding
    )
    return assembled, mesh
end

function _run_mesh_bcast(
    assembled::Reactant.ConcreteIFRTArray{T,M}, mesh::Sharding.Mesh, root::Int
) where {T,M}
    out_sharding = Sharding.NamedSharding(mesh, ntuple(Returns(nothing), M - 1))
    val_idx = Val(root + 1)
    fn = Reactant.Compiler.compile(
        _bcast_select_kernel,
        (assembled, val_idx);
        sync=true,
        output_shardings=Dict(1 => out_sharding),
    )
    return fn(assembled, val_idx)
end

function _run_mesh_reduce(
    assembled::Reactant.ConcreteIFRTArray{T,M}, mesh::Sharding.Mesh, op::F
) where {T,M,F}
    out_sharding = Sharding.NamedSharding(mesh, ntuple(Returns(nothing), M - 1))
    if op === avg
        fn_avg = Reactant.Compiler.compile(
            _reduce_avg_kernel,
            (assembled,);
            sync=true,
            output_shardings=Dict(1 => out_sharding),
        )
        return fn_avg(assembled)
    end
    fn = Reactant.Compiler.compile(
        _reduce_op_kernel,
        (assembled, op);
        sync=true,
        output_shardings=Dict(1 => out_sharding),
    )
    return fn(assembled, op)
end

function _write_replicated_result!(
    recvbuf::Reactant.ConcreteIFRTArray{T,N}, res::Reactant.ConcreteIFRTArray{T,N}
) where {T,N}
    shards = XLA.IFRT.disassemble_into_single_device_arrays(res.data, true)
    first_shard = first(shards)
    if recvbuf.sharding isa Sharding.NoShardInfo &&
        XLA.device(recvbuf) == XLA.device(first_shard)
        recvbuf.data = XLA.IFRT.AsyncArray(first_shard, nothing)
        return recvbuf
    end
    host_out = Array{T,N}(undef, size(res))
    XLA.to_host(first_shard, host_out, Sharding.NoSharding())
    copyto!(recvbuf, host_out)
    return recvbuf
end

function _write_replicated_result!(
    recvbuf::AbstractArray, res::Reactant.ConcreteIFRTArray{T,N}
) where {T,N}
    shards = XLA.IFRT.disassemble_into_single_device_arrays(res.data, true)
    first_shard = first(shards)
    if recvbuf isa Array{T,N}
        XLA.to_host(first_shard, recvbuf, Sharding.NoSharding())
    else
        host_out = Array{T,N}(undef, size(res))
        XLA.to_host(first_shard, host_out, Sharding.NoSharding())
        copyto!(recvbuf, host_out)
    end
    return recvbuf
end

function _write_replicated_result!(
    recvbuf::Base.RefValue{T}, res::Reactant.ConcreteIFRTArray{T,0}
) where {T}
    shards = XLA.IFRT.disassemble_into_single_device_arrays(res.data, true)
    host_out = Array{T,0}(undef)
    XLA.to_host(first(shards), host_out, Sharding.NoSharding())
    recvbuf[] = host_out[]
    return recvbuf
end

function _write_replicated_result!(
    recvbuf::Reactant.ConcreteIFRTNumber{T}, res::Reactant.ConcreteIFRTArray{T,0}
) where {T}
    shards = XLA.IFRT.disassemble_into_single_device_arrays(res.data, true)
    first_shard = first(shards)
    if recvbuf.sharding isa Sharding.NoShardInfo &&
        XLA.device(recvbuf) == XLA.device(first_shard)
        recvbuf.data = XLA.IFRT.AsyncArray(first_shard, nothing)
        return recvbuf
    end
    host_out = Array{T,0}(undef)
    XLA.to_host(first_shard, host_out, Sharding.NoSharding())
    tmp = Reactant.ConcreteIFRTNumber{T}(
        host_out[]; client=XLA.client(recvbuf), device=XLA.device(recvbuf), recvbuf.sharding
    )
    recvbuf.data = tmp.data
    return recvbuf
end

function _write_replicated_result!(
    recvbuf::Reactant.ConcretePJRTNumber{T}, res::Reactant.ConcreteIFRTArray{T,0}
) where {T}
    shards = XLA.IFRT.disassemble_into_single_device_arrays(res.data, true)
    host_out = Array{T,0}(undef)
    XLA.to_host(first(shards), host_out, Sharding.NoSharding())
    tmp = Reactant.ConcretePJRTNumber{T}(
        host_out[]; client=XLA.client(recvbuf), device=XLA.device(recvbuf), recvbuf.sharding
    )
    recvbuf.data = tmp.data
    return recvbuf
end

"""
    bcast!(sendrecvbuf; root::Integer=0, client=nothing)
    bcast!(sendbuf, recvbuf; root::Integer=0, client=nothing)

Broadcast `sendrecvbuf` (or `sendbuf` into `recvbuf`) from process `root` (0-indexed) to
all participating processes. Must be called outside traced functions.
"""
function bcast!(sendrecvbuf; root::Integer=0, client=nothing)
    return bcast!(sendrecvbuf, sendrecvbuf; root, client)
end

function bcast!(sendbuf, recvbuf; root::Integer=0, client=nothing)
    _check_not_traced(sendbuf, :bcast!)
    _check_not_traced(recvbuf, :bcast!)
    _check_compatible_buffers(sendbuf, recvbuf, :bcast!)

    np = num_processes()
    r = _validate_root(root, np)
    if np == 1
        return _copy_local_buffer!(recvbuf, sendbuf)
    end

    theclient = _infer_collective_client(client, sendbuf, recvbuf)
    if !(theclient isa XLA.IFRT.Client)
        error("Multi-process `Distributed.bcast!` requires the IFRT runtime")
    end

    local_arr = _to_host_array(sendbuf)
    assembled, mesh = _assemble_process_array(theclient, local_arr, np)
    res = _run_mesh_bcast(assembled, mesh, r)
    return _write_replicated_result!(recvbuf, res)
end

"""
    allreduce!(sendrecvbuf, op; client=nothing)
    allreduce!(sendbuf, recvbuf, op; client=nothing)

Reduce `sendrecvbuf` (or `sendbuf` into `recvbuf`) across all participating processes using
binary operator `op` (or [`Distributed.avg`](@ref)) and store the result on every process.
Must be called outside traced functions.
"""
function allreduce!(sendrecvbuf, op::F; client=nothing) where {F}
    return allreduce!(sendrecvbuf, sendrecvbuf, op; client)
end

function allreduce!(sendbuf, recvbuf, op::F; client=nothing) where {F}
    _check_not_traced(sendbuf, :allreduce!)
    _check_not_traced(recvbuf, :allreduce!)
    _check_compatible_buffers(sendbuf, recvbuf, :allreduce!)

    np = num_processes()
    if np == 1
        return _copy_local_buffer!(recvbuf, sendbuf)
    end

    theclient = _infer_collective_client(client, sendbuf, recvbuf)
    if !(theclient isa XLA.IFRT.Client)
        error("Multi-process `Distributed.allreduce!` requires the IFRT runtime")
    end

    local_arr = _to_host_array(sendbuf)
    assembled, mesh = _assemble_process_array(theclient, local_arr, np)
    res = _run_mesh_reduce(assembled, mesh, op)
    return _write_replicated_result!(recvbuf, res)
end

"""
    reduce!(sendrecvbuf, op; root::Integer=0, client=nothing)
    reduce!(sendbuf, recvbuf, op; root::Integer=0, client=nothing)

Reduce `sendrecvbuf` (or `sendbuf` into `recvbuf`) across all participating processes using
binary operator `op` (or [`Distributed.avg`](@ref)) and store the reduced result on process
`root` (0-indexed). On non-root processes `recvbuf` is left unmodified (and may be
`nothing` in the two-buffer form). Must be called outside traced functions.
"""
function reduce!(sendrecvbuf, op::F; root::Integer=0, client=nothing) where {F}
    return reduce!(sendrecvbuf, sendrecvbuf, op; root, client)
end

function reduce!(sendbuf, recvbuf, op::F; root::Integer=0, client=nothing) where {F}
    _check_not_traced(sendbuf, :reduce!)
    recvbuf === nothing || _check_not_traced(recvbuf, :reduce!)

    np = num_processes()
    r = _validate_root(root, np)
    is_root = local_rank() == r
    if is_root
        if recvbuf === nothing
            throw(
                ArgumentError(
                    "Distributed.reduce!: recvbuf cannot be `nothing` on root rank"
                ),
            )
        end
        _check_compatible_buffers(sendbuf, recvbuf, :reduce!)
    elseif recvbuf !== nothing
        _check_compatible_buffers(sendbuf, recvbuf, :reduce!)
    end

    if np == 1
        return _copy_local_buffer!(recvbuf, sendbuf)
    end

    theclient = _infer_collective_client(client, sendbuf, recvbuf)
    if !(theclient isa XLA.IFRT.Client)
        error("Multi-process `Distributed.reduce!` requires the IFRT runtime")
    end

    local_arr = _to_host_array(sendbuf)
    assembled, mesh = _assemble_process_array(theclient, local_arr, np)
    res = _run_mesh_reduce(assembled, mesh, op)
    if is_root
        return _write_replicated_result!(recvbuf, res)
    end
    Reactant.synchronize(res)
    return recvbuf
end

end

module Julia

using ..Reactant: Reactant
using Reactant.XLA: CAPI

using Reactant.XLA.PJRT.Plugin:
    Buffer,
    Client,
    Device,
    DeviceDescription,
    Event,
    Executable,
    LoadedExecutable,
    Memory,
    PluginApi,
    invalid_argument,
    make_api,
    pjrt_elsize,
    pjrt_eltype,
    resource_exhausted,
    unimplemented,
    vtable_memory

# interface functions extended by this plugin
import Reactant.XLA.PJRT.Plugin:
    addressable_device_logical_ids,
    addressable_devices,
    addressable_memories,
    buffer_from_host_buffer,
    compile!,
    compile_options,
    copy_to_device,
    copy_to_memory,
    create_uninitialized_buffer,
    debug_string,
    default_memory,
    description,
    device,
    device_assignment,
    devices,
    dimensions,
    element_type,
    executable,
    execute,
    fingerprint,
    id,
    is_addressable,
    is_on_cpu,
    isdeleted,
    kind,
    kind_id,
    local_hardware_id,
    memory,
    name,
    num_outputs,
    num_partitions,
    num_replicas,
    on_device_size_in_bytes,
    optimized_program,
    output_element_types,
    platform_name,
    platform_version,
    process_index,
    ready_event,
    serialize,
    size_of_generated_code_in_bytes,
    to_host_buffer,
    to_string,
    unsafe_pointer

using LinearAlgebra: BLAS, BlasFloat, Hermitian, cholesky, issuccess, mul!

## HostMemory
mutable struct HostMemory <: Memory
    const vtable::Ptr{CAPI.PJRT_Memory_FunctionTable}
    const id::Cint
    const kind::String
    const debug_string::String
    const to_string::String
    const devices::Vector{Device}

    function HostMemory(id, kind, devices)
        return new(
            unsafe_convert(Ptr{CAPI.PJRT_Memory_FunctionTable}, vtable_memory),
            id,
            kind,
            "Memory(id=$id, kind=$kind, platform=julia)",
            "Memory managed by Julia",
            devices,
        )
    end
end

id(mem::HostMemory) = mem.id
kind(mem::HostMemory) = mem.kind
kind_id(::HostMemory) = 0
debug_string(mem::HostMemory) = mem.debug_string
to_string(mem::HostMemory) = mem.to_string
addressable_devices(mem::HostMemory) = mem.devices

## DeviceDescription
mutable struct HostDeviceDescription <: DeviceDescription
    const id::Cint
    const pid::Cint
    const debug_string::String
    const to_string::String
end

function HostDeviceDescription()
    id = 0
    pid = 0
    return HostDeviceDescription(id, pid, "julia: id=$id", "JuliaDevice(id=$id)")
end

id(desc::HostDeviceDescription) = desc.id
process_index(desc::HostDeviceDescription) = desc.pid
kind(desc::HostDeviceDescription) = "julia-cpu"
debug_string(desc::HostDeviceDescription) = desc.debug_string
to_string(desc::HostDeviceDescription) = desc.to_string
# attributes(desc::HostDeviceDescription) = desc.

## Device
mutable struct HostDevice <: Device
    const description::DeviceDescription # TODO or HostDeviceDescription
    const local_hardware_id::Cint
    const memories::Vector{Memory}
end

description(dev::HostDevice) = dev.description
is_addressable(dev::HostDevice) = true
local_hardware_id(dev::HostDevice) = dev.local_hardware_id
addressable_memories(dev::HostDevice) = dev.memories
default_memory(dev::HostDevice) = dev.memories[1]
# memory_stats(dev::HostDevice) = 

## HostBuffer
mutable struct HostBuffer <: Buffer
    data::Ptr{Cvoid}
    const size_in_bytes::Csize_t
    const type::CAPI.PJRT_Buffer_Type
    const dims::Vector{Int64}
    const minor_to_major::Vector{Int64}
    const device::Device
    const memory::Memory

    function HostBuffer(args...)
        obj = new(args...)
        return Base.finalizer(obj) do x
            if !isdeleted(x)
                Libc.free(x.data)
            end
        end
    end
end

function Base.delete!(buffer::HostBuffer)
    if !isdeleted(buffer)
        Libc.free(buffer.data)
        buffer.data = C_NULL
    end
end

isdeleted(buffer::HostBuffer) = buffer.data == C_NULL
element_type(buffer::HostBuffer) = buffer.type
dimensions(buffer::HostBuffer) = buffer.dims
device(buffer::HostBuffer) = buffer.device
memory(buffer::HostBuffer) = buffer.memory
on_device_size_in_bytes(buffer::HostBuffer) = buffer.size_in_bytes

function copy_to_device(dev::HostDevice, buffer::HostBuffer)
    if dev === device(buffer)
        throw(invalid_argument("HostBuffer is already on HostDevice"))
    end
    new_data = Libc.malloc(buffer.size_in_bytes)
    new_data == C_NULL && throw(resource_exhausted())
    Libc.memcpy(new_data, buffer.data, buffer.size_in_bytes)
    new_buffer = HostBuffer(
        new_data,
        buffer.size_in_bytes,
        buffer.type,
        buffer.dims,
        buffer.minor_to_major,
        dev,
        default_memory(dev),
    )
    return new_buffer
end

function copy_to_memory(mem::HostMemory, buffer::HostBuffer)
    devs = addressable_devices(mem)
    isempty(devs) && throw(invalid_argument("HostMemory has no addressable devices"))
    dev = device(buffer) in devs ? device(buffer) : first(devs)
    new_data = Libc.malloc(buffer.size_in_bytes)
    new_data == C_NULL && throw(resource_exhausted())
    Libc.memcpy(new_data, buffer.data, buffer.size_in_bytes)
    return HostBuffer(
        new_data,
        buffer.size_in_bytes,
        buffer.type,
        buffer.dims,
        buffer.minor_to_major,
        dev,
        mem,
    )
end

function to_host_buffer(dst::Ptr{Cvoid}, buffer::HostBuffer)
    Libc.memcpy(dst, buffer.data, buffer.size_in_bytes)
    return nothing
end

is_on_cpu(::HostBuffer) = true

# `HostBuffer` data is written synchronously before the buffer is handed out, so it is always ready
function ready_event(::HostBuffer)
    event = Event()
    notify(event)
    return event
end

unsafe_pointer(buffer::HostBuffer) = buffer.data

## HostExecutable
mutable struct HostExecutable <: Executable
    const name::String
    const program::String
    const output_types::Vector{CAPI.PJRT_Buffer_Type}
    const options::Reactant.Proto.xla.CompileOptionsProto
end

name(exe::HostExecutable) = exe.name

function num_replicas(exe::HostExecutable)
    build_options = exe.options.executable_build_options
    return isnothing(build_options) ? 1 : max(build_options.num_replicas, 1)
end

function num_partitions(exe::HostExecutable)
    build_options = exe.options.executable_build_options
    return isnothing(build_options) ? 1 : max(build_options.num_partitions, 1)
end

num_outputs(exe::HostExecutable) = length(exe.output_types)
size_of_generated_code_in_bytes(::HostExecutable) = 0
fingerprint(::HostExecutable) = throw(unimplemented("`HostExecutable` has no fingerprint"))
output_element_types(exe::HostExecutable) = exe.output_types
optimized_program(exe::HostExecutable) = exe.program
function serialize(::HostExecutable)
    throw(unimplemented("serialization of `HostExecutable` is not supported yet"))
end
compile_options(exe::HostExecutable) = exe.options

## HostLoadedExecutable
# `Tag` is unique per compiled program, so that `compile!` can define an `execute` method for each one
mutable struct HostLoadedExecutable{Tag} <: LoadedExecutable
    const executable::HostExecutable
    const devices::Vector{Device}
    const logical_ids::Vector{CAPI.PJRT_LogicalDeviceIds}
    const device_assignment::Union{Nothing,Reactant.Proto.xla.DeviceAssignmentProto}
    deleted::Bool
end

executable(exec::HostLoadedExecutable) = exec.executable

function Base.delete!(exec::HostLoadedExecutable)
    exec.deleted = true
    return nothing
end

isdeleted(exec::HostLoadedExecutable) = exec.deleted
addressable_devices(exec::HostLoadedExecutable) = exec.devices
addressable_device_logical_ids(exec::HostLoadedExecutable) = exec.logical_ids
device_assignment(exec::HostLoadedExecutable) = exec.device_assignment

"""
    handle(::Val{opname}, operands::Tuple, attributes::NamedTuple, results::Tuple)

Run the MLIR op `opname` (e.g. `Symbol("stablehlo.add")`) and return its results.

- `operands`: Julia arrays, i.e. row-major MLIR tensors seen as column-major arrays with reversed dimensions, or
  [`Token`](@ref)s.
- `attributes`: the op attributes, decoded by `decode_attribute`.
- `results`: the type of each result, as `(; eltype, size)` with `size` in Julia order (i.e. reversed), or `Token`
  for tokens.

Like MLIR values, the return value depends on the number of results: it is ignored if there are none, it is the
result itself if there is one (`%N`), and it must be a tuple with one entry per result if there are several (`%N:2`,
bound to `%N#0` and `%N#1`).
"""
function handle end

# element-wise ops use `map` instead of broadcasting: StableHLO operands always have the same shape, and broadcasting
# would return a scalar over 0-dimensional arrays and a `BitArray` for booleans instead of an `Array`
handle(::Val{Symbol("stablehlo.add")}, (a, b), _, _) = map(+, a, b)
handle(::Val{Symbol("stablehlo.subtract")}, (a, b), _, _) = map(-, a, b)
handle(::Val{Symbol("stablehlo.multiply")}, (a, b), _, _) = map(*, a, b)
handle(::Val{Symbol("stablehlo.maximum")}, (a, b), _, _) = map(max, a, b)
handle(::Val{Symbol("stablehlo.minimum")}, (a, b), _, _) = map(min, a, b)
handle(::Val{Symbol("stablehlo.negate")}, (a,), _, _) = map(-, a)
handle(::Val{Symbol("stablehlo.abs")}, (a,), _, _) = map(abs, a)
handle(::Val{Symbol("stablehlo.floor")}, (a,), _, _) = map(floor, a)
handle(::Val{Symbol("stablehlo.and")}, (a, b), _, _) = map(&, a, b)
handle(::Val{Symbol("stablehlo.or")}, (a, b), _, _) = map(|, a, b)
handle(::Val{Symbol("stablehlo.not")}, (a,), _, _) = map(~, a)
handle(::Val{Symbol("stablehlo.xor")}, (a, b), _, _) = map(xor, a, b)
handle(::Val{Symbol("stablehlo.ceil")}, (a,), _, _) = map(ceil, a)
function handle(::Val{Symbol("stablehlo.round_nearest_even")}, (a,), _, _)
    return map(x -> round(x, RoundNearest), a)
end
function handle(::Val{Symbol("stablehlo.round_nearest_afz")}, (a,), _, _)
    return map(x -> round(x, RoundNearestTiesAway), a)
end
handle(::Val{Symbol("stablehlo.sign")}, (a,), _, _) = map(sign, a)
handle(::Val{Symbol("stablehlo.tanh")}, (a,), _, _) = map(tanh, a)
handle(::Val{Symbol("stablehlo.exponential")}, (a,), _, _) = map(exp, a)
handle(::Val{Symbol("stablehlo.exponential_minus_one")}, (a,), _, _) = map(expm1, a)
handle(::Val{Symbol("stablehlo.logistic")}, (a,), _, _) = map(x -> inv(one(x) + exp(-x)), a)
handle(::Val{Symbol("stablehlo.is_finite")}, (a,), _, _) = map(isfinite, a)
handle(::Val{Symbol("stablehlo.real")}, (a,), _, _) = map(real, a)
handle(::Val{Symbol("stablehlo.imag")}, (a,), _, _) = map(imag, a)
handle(::Val{Symbol("stablehlo.complex")}, (a, b), _, _) = map(complex, a, b)
function handle(::Val{Symbol("stablehlo.count_leading_zeros")}, (a,), _, _)
    return map(x -> oftype(x, leading_zeros(x)), a)
end
function handle(::Val{Symbol("stablehlo.popcnt")}, (a,), _, _)
    return map(x -> oftype(x, count_ones(x)), a)
end

# integer division truncates towards zero, like `div`
function handle(::Val{Symbol("stablehlo.divide")}, (a, b), _, (result,))
    return map(result.eltype <: Integer ? div : /, a, b)
end

handle(::Val{Symbol("stablehlo.constant")}, _, attributes, _) = attributes.value

# each element is its 0-based index along `iota_dimension`, an MLIR dimension, which is Julia dimension `N - d`
function handle(::Val{Symbol("stablehlo.iota")}, _, attributes, (result,))
    j = length(result.size) - attributes.iota_dimension
    return [result.eltype(I[j] - 1) for I in CartesianIndices(result.size)]
end

# Julia throws `DomainError` where StableHLO returns NaN (e.g. `sin(Inf)`, `log(-1.0)`, `sqrt(-1.0)`), so real
# inputs outside of the domain are mapped to NaN explicitly
function handle(::Val{Symbol("stablehlo.sine")}, (a,), _, _)
    return map(x -> x isa Real && isinf(x) ? oftype(x, NaN) : sin(x), a)
end

function handle(::Val{Symbol("stablehlo.cosine")}, (a,), _, _)
    return map(x -> x isa Real && isinf(x) ? oftype(x, NaN) : cos(x), a)
end

function handle(::Val{Symbol("stablehlo.log")}, (a,), _, _)
    return map(x -> x isa Real && x < 0 ? oftype(x, NaN) : log(x), a)
end

function handle(::Val{Symbol("stablehlo.sqrt")}, (a,), _, _)
    return map(x -> x isa Real && x < 0 ? oftype(x, NaN) : sqrt(x), a)
end

function handle(::Val{Symbol("stablehlo.rsqrt")}, (a,), _, _)
    return map(x -> x isa Real && x < 0 ? oftype(x, NaN) : inv(sqrt(x)), a)
end

function handle(::Val{Symbol("stablehlo.tan")}, (a,), _, _)
    return map(x -> x isa Real && isinf(x) ? oftype(x, NaN) : tan(x), a)
end

function handle(::Val{Symbol("stablehlo.log_plus_one")}, (a,), _, _)
    return map(x -> x isa Real && x < -1 ? oftype(x, NaN) : log1p(x), a)
end

# Julia has no complex `cbrt`, so complex numbers use the principal cube root like XLA
function handle(::Val{Symbol("stablehlo.cbrt")}, (a,), _, _)
    return map(x -> x isa Complex ? x^(one(real(x)) / 3) : cbrt(x), a)
end

# like XLA, integer powers with negative exponents are 0 except for bases 1 and -1 (which Julia's `^` handles), and
# negative real bases with finite non-integer exponents are NaN
function handle(::Val{Symbol("stablehlo.power")}, (a, b), _, _)
    return map(a, b) do x, y
        if x isa Integer
            y < 0 && abs(x) != 1 ? zero(x) : x^y
        elseif x isa Real && x < 0 && isfinite(y) && !isinteger(y)
            oftype(x, NaN)
        else
            x^y
        end
    end
end

# the sign of the result follows the dividend, like `rem`. like XLA, an integer remainder by zero is the dividend
function handle(::Val{Symbol("stablehlo.remainder")}, (a, b), _, _)
    return map((x, y) -> x isa Integer && iszero(y) ? x : rem(x, y), a, b)
end

# like XLA, shift amounts are read as unsigned, so negative amounts are treated as too large: logical shifts by at least
# the bit width give 0, which Julia does for unsigned amounts, and arithmetic shifts fill with the sign bit (the most
# significant bit, also for unsigned integers, hence the reinterpretation as signed)
function handle(::Val{Symbol("stablehlo.shift_left")}, (a, b), _, _)
    return map((x, y) -> x << unsigned(y), a, b)
end

function handle(::Val{Symbol("stablehlo.shift_right_logical")}, (a, b), _, _)
    return map((x, y) -> x >>> unsigned(y), a, b)
end

function handle(::Val{Symbol("stablehlo.shift_right_arithmetic")}, (a, b), _, _)
    return map((x, y) -> (signed(x) >> unsigned(y)) % typeof(x), a, b)
end

# `min` and `max` are either 0-dimensional or have the shape of `operand`, so they are broadcast into a new array
function handle(::Val{Symbol("stablehlo.clamp")}, (lo, x, hi), _, _)
    return broadcast!((l, v, h) -> min(max(v, l), h), similar(x), lo, x, hi)
end

# like XLA, integers wrap around, floats are truncated towards zero and saturate when converted to integers (NaN
# becomes 0), and complex numbers lose their imaginary part when converted to real numbers
function handle(::Val{Symbol("stablehlo.convert")}, (a,), _, (result,))
    T = result.eltype
    return map(a) do x
        if T === Bool
            !iszero(x)
        elseif x isa Complex && !(T <: Complex)
            convert(T, real(x))
        elseif T <: Integer && x isa Integer
            x % T
        elseif T <: Integer
            if isnan(x)
                zero(T)
            elseif x <= typemin(T)
                typemin(T)
            elseif x >= typemax(T)
                typemax(T)
            else
                trunc(T, x)
            end
        else
            convert(T, x)
        end
    end
end

# `compare_type` defaults to the natural order of the element type (IEEE for floats, so comparisons with NaN are false
# except for `NE`). with `TOTALORDER`, floats are compared by their bits as -NaN < -Inf < ... < -0.0 < +0.0 < ... <
# +Inf < +NaN, which is done by flipping all but the sign bit of negative floats
function handle(::Val{Symbol("stablehlo.compare")}, (a, b), attributes, _)
    direction = attributes.comparison_direction
    f = if direction === :EQ
        ==
    elseif direction === :NE
        !=
    elseif direction === :GE
        >=
    elseif direction === :GT
        >
    elseif direction === :LE
        <=
    elseif direction === :LT
        <
    else
        throw(invalid_argument("unknown comparison direction $direction"))
    end

    if get(attributes, :compare_type, :NOTYPE) === :TOTALORDER
        key =
            x -> if x isa Base.IEEEFloat
                (bits = reinterpret(Signed, x); bits < 0 ? bits ⊻ typemax(bits) : bits)
            else
                x
            end
        return map((x, y) -> f(key(x), key(y)), a, b)
    end
    return map(f, a, b)
end

# a 0-dimensional predicate selects whole operands, otherwise elements are selected one by one
function handle(::Val{Symbol("stablehlo.select")}, (pred, on_true, on_false), _, _)
    ndims(pred) == 0 && return pred[] ? on_true : on_false
    return map(ifelse, pred, on_true, on_false)
end

# the result has the batching dimensions, then the free (i.e. neither batching nor contracting) dimensions of `lhs`
# and then those of `rhs`, all in MLIR order. on the reversed arrays this is `rhs * lhs` for each batch: `rhs` is
# permuted and reshaped into a (rhs free × contracting × batch) array, `lhs` into a (contracting × lhs free × batch)
# array, and the product of each batch slice is written into a (rhs free × lhs free × batch) array.
# `dot_dimension_numbers` is decoded by `decode_attribute` with 0-based MLIR dimensions.
function handle(::Val{Symbol("stablehlo.dot_general")}, (lhs, rhs), attributes, (result,))
    dims = attributes.dot_dimension_numbers
    jdim(x, d) = ndims(x) - d

    # batching dimensions are reversed so that they flatten in the order of the result
    lb = [jdim(lhs, d) for d in reverse(dims.lhs_batching_dimensions)]
    rb = [jdim(rhs, d) for d in reverse(dims.rhs_batching_dimensions)]
    lc = [jdim(lhs, d) for d in dims.lhs_contracting_dimensions]
    rc = [jdim(rhs, d) for d in dims.rhs_contracting_dimensions]

    # free dimensions keep their MLIR order, which reversed is the increasing order of Julia dimensions
    lf = [j for j in 1:ndims(lhs) if j ∉ lb && j ∉ lc]
    rf = [j for j in 1:ndims(rhs) if j ∉ rb && j ∉ rc]

    nbatch = prod(j -> size(lhs, j), lb; init=1)
    k = prod(j -> size(lhs, j), lc; init=1)
    m = prod(j -> size(lhs, j), lf; init=1)
    n = prod(j -> size(rhs, j), rf; init=1)

    l = reshape(permutedims(lhs, [lc; lf; lb]), k, m, nbatch)
    r = reshape(permutedims(rhs, [rf; rc; rb]), n, k, nbatch)
    out = Array{result.eltype}(undef, n, m, nbatch)
    for (o, x, y) in zip(eachslice(out; dims=3), eachslice(r; dims=3), eachslice(l; dims=3))
        mul!(o, x, y)
    end
    return reshape(out, result.size)
end

# shape handlers. arrays have reversed dimensions, so MLIR dimension `d` (0-based) of a rank-`N` tensor is Julia
# dimension `N - d`, and attributes indexed by MLIR dimension (e.g. `permutation`) are read in reverse

# reshapes follow the row-major order of MLIR tensors, which is the column-major order of the reversed arrays
function handle(::Val{Symbol("stablehlo.reshape")}, (a,), _, (result,))
    return reshape(a, result.size)
end

function handle(::Val{Symbol("stablehlo.transpose")}, (a,), attributes, _)
    perm = attributes.permutation
    N = length(perm)
    return permutedims(a, ntuple(j -> N - perm[N - j + 1], N))
end

# `start_indices` are 0-based and `limit_indices` are exclusive
function handle(::Val{Symbol("stablehlo.slice")}, (a,), attributes, _)
    (; start_indices, limit_indices, strides) = attributes
    N = length(start_indices)
    ranges = ntuple(N) do j
        d = N - j + 1
        (start_indices[d] + 1):strides[d]:limit_indices[d]
    end
    # indexing a 0-dimensional array with no indices would return a scalar
    return copy(view(a, ranges...))
end

# like in StableHLO, start indices are clamped so that the slice is always within bounds
function handle(::Val{Symbol("stablehlo.dynamic_slice")}, operands, attributes, _)
    a = first(operands)
    starts = Base.tail(operands)
    sizes = attributes.slice_sizes
    N = length(sizes)
    ranges = ntuple(N) do j
        d = N - j + 1
        start = clamp(Int(starts[d][]), 0, size(a, j) - sizes[d])
        (start + 1):(start + sizes[d])
    end
    return copy(view(a, ranges...))
end

function handle(::Val{Symbol("stablehlo.dynamic_update_slice")}, operands, _, _)
    a, update = operands[1], operands[2]
    starts = operands[3:end]
    N = ndims(a)
    ranges = ntuple(N) do j
        start = clamp(Int(starts[N - j + 1][]), 0, size(a, j) - size(update, j))
        (start + 1):(start + size(update, j))
    end

    # the operand may be an input or a constant, so the update goes into a copy
    result = copy(a)
    view(result, ranges...) .= update
    return result
end

function handle(
    ::Val{Symbol("stablehlo.dynamic_iota")}, (output_shape,), attributes, (result,)
)
    dims = ntuple(j -> Int(output_shape[end - j + 1]), Val(length(result.size)))
    j = length(dims) - attributes.iota_dimension
    return [result.eltype(I[j] - 1) for I in CartesianIndices(dims)]
end

function handle(::Val{Symbol("stablehlo.dynamic_reshape")}, (a, output_shape), _, (result,))
    return reshape(a, ntuple(j -> Int(output_shape[end - j + 1]), Val(length(result.size))))
end

# MLIR dimension `i` of the operand is MLIR dimension `broadcast_dimensions[i]` of the result. the operand is permuted
# so that its dimensions follow the order of the result ones, reshaped with size 1 for the missing dimensions and
# broadcast into the result
function handle(::Val{Symbol("stablehlo.broadcast_in_dim")}, (a,), attributes, (result,))
    bdims = attributes.broadcast_dimensions
    N = length(result.size)
    M = ndims(a)

    # result Julia dimension of each operand Julia dimension
    rdims = [N - bdims[M - j + 1] for j in 1:M]
    perm = sortperm(rdims)
    shape = ntuple(Val(N)) do k
        j = findfirst(==(k), rdims)
        isnothing(j) ? 1 : size(a, j)
    end

    out = Array{result.eltype}(undef, result.size)
    out .= reshape(permutedims(a, perm), shape)
    return out
end

# handle dynamic info and call `stablehlo.broadcast_in_dim` handler
function handle(
    ::Val{Symbol("stablehlo.dynamic_broadcast_in_dim")},
    (a, output_dimensions),
    attributes,
    (result,),
)
    dims = ntuple(j -> Int(output_dimensions[end - j + 1]), Val(length(result.size)))
    static = NamedTuple{(:eltype, :size),Tuple{Type{result.eltype},typeof(dims)}}((
        result.eltype, dims
    ))
    return handle(Val(Symbol("stablehlo.broadcast_in_dim")), (a,), attributes, (static,))
end

function handle(::Val{Symbol("stablehlo.concatenate")}, operands, attributes, (result,))
    N = length(result.size)
    j = N - attributes.dimension
    out = Array{result.eltype}(undef, result.size)
    offset = 0
    for x in operands
        n = size(x, j)
        view(
            out,
            ntuple(k -> k == j ? ((offset + 1):(offset + n)) : (1:size(out, k)), Val(N))...,
        ) .= x
        offset += n
    end
    return out
end

# element `k` (1-based) of the operand along a dimension goes to `low + (k - 1) * (interior + 1) + 1` of the result.
# negative `low` or `high` crop the operand, so only the elements that land within the result are copied
function handle(::Val{Symbol("stablehlo.pad")}, (a, padding_value), attributes, (result,))
    (; edge_padding_low, interior_padding) = attributes
    N = length(result.size)
    out = fill(convert(result.eltype, padding_value[]), result.size)

    src = ntuple(Val(N)) do j
        d = N - j + 1
        lo, step = edge_padding_low[d], interior_padding[d] + 1
        kmin = max(1, cld(-lo, step) + 1)
        kmax = min(size(a, j), fld(size(out, j) - 1 - lo, step) + 1)
        kmin:kmax
    end
    dst = ntuple(Val(N)) do j
        d = N - j + 1
        lo, step = edge_padding_low[d], interior_padding[d] + 1
        (lo + (first(src[j]) - 1) * step + 1):step:(lo + (last(src[j]) - 1) * step + 1)
    end
    view(out, dst...) .= view(a, src...)
    return out
end

# handle dynamic info and call `stablehlo.pad` handler
function handle(
    ::Val{Symbol("stablehlo.dynamic_pad")},
    (a, padding_value, low, high, interior),
    _,
    (result,),
)
    lo, hi, in = collect(Int, low), collect(Int, high), collect(Int, interior)
    N = ndims(a)
    dims = ntuple(Val(N)) do j
        d = N - j + 1
        n = size(a, j)
        lo[d] + n + max(n - 1, 0) * in[d] + hi[d]
    end
    static = NamedTuple{(:eltype, :size),Tuple{Type{result.eltype},typeof(dims)}}((
        result.eltype, dims
    ))
    attributes = (; edge_padding_low=lo, edge_padding_high=hi, interior_padding=in)
    return handle(Val(Symbol("stablehlo.pad")), (a, padding_value), attributes, (static,))
end

function __gather_scatter_julia_index(idx, ::Val{N}) where {N}
    return CartesianIndex(ntuple(j -> idx[N - j + 1] + 1, Val(N)))
end

function handle(
    ::Val{Symbol("stablehlo.gather")}, (operand, start_indices), attributes, (result,)
)
    dn = attributes.dimension_numbers
    slice_sizes = attributes.slice_sizes
    ivd = dn.index_vector_dim
    out = Array{result.eltype}(undef, result.size)

    batch_dims = [d for d in 0:(ndims(out) - 1) if d ∉ dn.offset_dims]
    offset_operand_dims = [
        d for d in 0:(ndims(operand) - 1) if
        d ∉ dn.collapsed_slice_dims && d ∉ dn.operand_batching_dims
    ]

    operand_index = zeros(Int, ndims(operand))
    sindex = zeros(Int, ndims(start_indices))
    for I in CartesianIndices(out)
        # `gather` follow the StableHLO spec, which is written in 0-based MLIR indices
        # these convert them from and to the Julia indices of the reversed arrays
        ridx = reverse(Tuple(I)) .- 1
        fill!(operand_index, 0)

        # the batch index, with a gap at `index_vector_dim` for the component of the start index
        for (k, d) in enumerate(batch_dims)
            sindex[k <= ivd ? k : k + 1] = ridx[d + 1]
        end
        for (k, d_operand) in enumerate(dn.start_index_map)
            ivd < ndims(start_indices) && (sindex[ivd + 1] = k - 1)
            start = Int(
                start_indices[__gather_scatter_julia_index(
                    sindex, Val(ndims(start_indices))
                )],
            )
            mlir_size = size(operand, ndims(operand) - d_operand)
            operand_index[d_operand + 1] = clamp(
                start, 0, mlir_size - slice_sizes[d_operand + 1]
            )
        end
        for (i, d_operand) in enumerate(dn.operand_batching_dims)
            d_start = dn.start_indices_batching_dims[i]
            operand_index[d_operand + 1] += ridx[batch_dims[d_start - (d_start < ivd ? 0 : 1) + 1] + 1]
        end
        for (k, d) in enumerate(dn.offset_dims)
            operand_index[offset_operand_dims[k] + 1] += ridx[d + 1]
        end

        out[I] = operand[__gather_scatter_julia_index(operand_index, Val(ndims(operand)))]
    end
    return out
end

# handle dynamic info and call `stablehlo.gather` handler
function handle(
    ::Val{Symbol("stablehlo.dynamic_gather")},
    (operand, start_indices, slice_sizes),
    attributes,
    (result,),
)
    dn = attributes.dimension_numbers
    sizes = collect(Int, slice_sizes)
    S = ndims(start_indices)
    batch = [size(start_indices, S - d) for d in 0:(S - 1) if d != dn.index_vector_dim]
    offset = [
        sizes[d + 1] for d in 0:(ndims(operand) - 1) if
        d ∉ dn.collapsed_slice_dims && d ∉ dn.operand_batching_dims
    ]
    R = length(result.size)
    mlir_dims = ntuple(Val(R)) do d
        k = findfirst(==(d - 1), dn.offset_dims)
        isnothing(k) ? batch[d - count(<(d - 1), dn.offset_dims)] : offset[k]
    end
    dims = reverse(mlir_dims)
    static = NamedTuple{(:eltype, :size),Tuple{Type{result.eltype},typeof(dims)}}((
        result.eltype, dims
    ))
    return handle(
        Val(Symbol("stablehlo.gather")),
        (operand, start_indices),
        merge(attributes, (; slice_sizes=sizes)),
        (static,),
    )
end

# operands are the `inputs`, `scatter_indices` and `updates`, and `update_computation` combines an element of each
# input with the corresponding element of each update (all as 0-dimensional arrays). like in the spec, updates whose
# index falls outside of the inputs are dropped, and they are applied sequentially in index order
function handle(
    ::Val{Symbol("stablehlo.scatter")}, operands, attributes, results, (update_computation,)
)
    n = length(results)
    outs = ntuple(i -> copy(operands[i]), Val(n))
    scatter_indices = operands[n + 1]
    updates = ntuple(i -> operands[n + 1 + i], Val(n))

    dn = attributes.scatter_dimension_numbers
    ivd = dn.index_vector_dim
    R = ndims(first(outs))
    update_scatter_dims = [
        d for d in 0:(ndims(first(updates)) - 1) if d ∉ dn.update_window_dims
    ]
    window_input_dims = [
        d for d in 0:(R - 1) if d ∉ dn.inserted_window_dims && d ∉ dn.input_batching_dims
    ]

    result_index = zeros(Int, R)
    sindex = zeros(Int, ndims(scatter_indices))
    for I in CartesianIndices(first(updates))
        # `scatter` follow the StableHLO spec, which is written in 0-based MLIR indices
        # these convert them from and to the Julia indices of the reversed arrays
        uidx = reverse(Tuple(I)) .- 1
        fill!(result_index, 0)

        for (k, d) in enumerate(update_scatter_dims)
            sindex[k <= ivd ? k : k + 1] = uidx[d + 1]
        end
        for (k, d_input) in enumerate(dn.scatter_dims_to_operand_dims)
            ivd < ndims(scatter_indices) && (sindex[ivd + 1] = k - 1)
            result_index[d_input + 1] = Int(
                scatter_indices[__gather_scatter_julia_index(
                    sindex, Val(ndims(scatter_indices))
                )],
            )
        end
        for (i, d_input) in enumerate(dn.input_batching_dims)
            d_start = dn.scatter_indices_batching_dims[i]
            result_index[d_input + 1] += uidx[update_scatter_dims[d_start - (d_start < ivd ? 0 : 1) + 1] + 1]
        end
        for (k, d) in enumerate(dn.update_window_dims)
            result_index[window_input_dims[k] + 1] += uidx[d + 1]
        end

        mlir_size = (d) -> size(first(outs), ndims(first(outs)) - d)
        all(d -> 0 <= result_index[d + 1] < mlir_size(d), 0:(R - 1)) || continue
        J = __gather_scatter_julia_index(result_index, Val(R))
        new = update_computation(
            map(o -> fill(o[J]), outs)..., map(u -> fill(u[I]), updates)...
        )
        foreach((o, v) -> o[J] = v[], outs, new)
    end
    return n == 1 ? only(outs) : outs
end

# both branches take no arguments and return the results of the op
function handle(
    ::Val{Symbol("stablehlo.if")}, (pred,), _, results, (true_branch, false_branch)
)
    rets = pred[] ? true_branch() : false_branch()
    length(results) == 0 && return nothing
    return length(results) == 1 ? only(rets) : rets
end

# like `stablehlo.if` with 0-based `index` selecting the branch; out-of-range indices run the last branch
function handle(::Val{Symbol("stablehlo.case")}, (index,), _, results, branches)
    i = index[]
    rets = (0 <= i < length(branches) ? branches[i + 1] : last(branches))()
    length(results) == 0 && return nothing
    return length(results) == 1 ? only(rets) : rets
end

# `cond` and `body` take the loop-carried values; `cond` returns a 0-dimensional boolean and `body` the next values
function handle(::Val{Symbol("stablehlo.while")}, operands, _, results, (cond, body))
    vals = operands
    while only(cond(vals...))[]
        vals = body(vals...)
    end
    length(results) == 0 && return nothing
    return length(results) == 1 ? only(vals) : vals
end

# operands are the `inputs` and their `init_values`, and `body` combines the accumulated values with an element of each
# input (all as 0-dimensional arrays). `dimensions` are the reduced MLIR dimensions, and the result keeps the other
# dimensions in order, which on the reversed arrays are the remaining Julia dimensions in increasing order
function handle(::Val{Symbol("stablehlo.reduce")}, operands, attributes, results, (body,))
    n = length(results)
    inputs = ntuple(i -> operands[i], Val(n))
    inits = ntuple(i -> operands[n + i], Val(n))
    outs = ntuple(
        i -> fill(convert(results[i].eltype, inits[i][]), results[i].size), Val(n)
    )

    N = ndims(first(inputs))
    reduced = [N - d for d in attributes.dimensions]
    kept = [j for j in 1:N if j ∉ reduced]
    M = Val(length(first(results).size))
    for I in CartesianIndices(first(inputs))
        J = CartesianIndex(ntuple(k -> I[kept[k]], M))
        new = body(map(o -> fill(o[J]), outs)..., map(x -> fill(x[I]), inputs)...)
        foreach((o, v) -> o[J] = v[], outs, new)
    end
    return n == 1 ? only(outs) : outs
end

function handle(::Val{Symbol("stablehlo.bitcast_convert")}, (a,), _, (result,))
    return copyto!(
        Array{result.eltype}(undef, result.size), reinterpret(result.eltype, vec(a))
    )
end

function handle(::Val{Symbol("stablehlo.cholesky")}, (a,), attributes, _)
    lower = get(attributes, :lower, false)
    out = similar(a)
    for I in CartesianIndices(axes(a)[3:end])
        m = permutedims(view(a, :, :, I))
        c = cholesky(Hermitian(m, lower ? :L : :U); check=false)
        f = issuccess(c) ? Matrix(lower ? c.L : c.U) : fill(eltype(a)(NaN), size(m))
        out[:, :, I] = permutedims(f)
    end
    return out
end

# types without BLAS support (e.g. `Float16`) are solved in a promoted type
function handle(::Val{Symbol("stablehlo.triangular_solve")}, (a, b), attributes, (result,))
    T = result.eltype <: BlasFloat ? result.eltype : promote_type(result.eltype, Float32)
    sa = convert(Array{T}, a)
    sx = copyto!(similar(sa, size(b)), b)
    side = attributes.left_side ? 'R' : 'L'
    uplo = attributes.lower ? 'U' : 'L'
    trans = attributes.transpose_a
    trans = if trans === :NO_TRANSPOSE
        'N'
    elseif trans === :TRANSPOSE
        'T'
    else
        'C'
    end
    diag = attributes.unit_diagonal ? 'U' : 'N'
    for I in CartesianIndices(axes(sa)[3:end])
        BLAS.trsm!(side, uplo, trans, diag, one(T), view(sa, :, :, I), view(sx, :, :, I))
    end
    return convert(Array{result.eltype}, sx)
end

function handle(
    ::Val{Symbol("stablehlo.batch_norm_inference")},
    (x, scale, offset, mean, variance),
    attributes,
    _,
)
    N = ndims(x)
    shape = ntuple(k -> k == N - attributes.feature_index ? length(scale) : 1, Val(N))
    s, o, m, v = map(a -> reshape(a, shape), (scale, offset, mean, variance))
    eps = convert(eltype(x), attributes.epsilon)
    stddev = map(w -> w < 0 ? oftype(w, NaN) : sqrt(w), v .+ eps)
    return @. s * (x - m) / stddev + o
end

function handle(
    ::Val{Symbol("stablehlo.batch_norm_training")}, (x, scale, offset), attributes, _
)
    N = ndims(x)
    j = N - attributes.feature_index
    other = Tuple(k for k in 1:N if k != j)
    n = convert(eltype(x), prod(k -> size(x, k), other; init=1))
    shape = ntuple(k -> k == j ? length(scale) : 1, Val(N))
    s, o = reshape(scale, shape), reshape(offset, shape)
    eps = convert(eltype(x), attributes.epsilon)

    m = sum(x; dims=other) ./ n
    v = sum(abs2, x .- m; dims=other) ./ n
    out = @. s * (x - m) / sqrt(v + eps) + o
    return out, vec(m), vec(v)
end

function handle(
    ::Val{Symbol("stablehlo.batch_norm_grad")},
    (x, scale, mean, variance, dy),
    attributes,
    _,
)
    N = ndims(x)
    j = N - attributes.feature_index
    other = Tuple(k for k in 1:N if k != j)
    n = convert(eltype(x), prod(k -> size(x, k), other; init=1))
    shape = ntuple(k -> k == j ? length(scale) : 1, Val(N))
    s, m, v = map(a -> reshape(a, shape), (scale, mean, variance))
    eps = convert(eltype(x), attributes.epsilon)

    centered = x .- m
    stddev = map(w -> w < 0 ? oftype(w, NaN) : sqrt(w), v .+ eps)
    grad_offset = sum(dy; dims=other)
    grad_scale = sum(dy .* centered ./ stddev; dims=other)
    i3 = sum(dy .* centered; dims=other)
    grad_x = @. s / stddev / n * (n * dy - grad_offset - i3 * centered / (v + eps))
    return grad_x, vec(grad_scale), vec(grad_offset)
end

# `dimensions` are MLIR dimensions, which are Julia dimensions `N - d`
function handle(::Val{Symbol("stablehlo.reverse")}, (a,), attributes, _)
    return reverse(a; dims=Tuple(ndims(a) - d for d in attributes.dimensions))
end

# the size of MLIR dimension `dimension` as a 0-dimensional i32 tensor
function handle(::Val{Symbol("stablehlo.get_dimension_size")}, (a,), attributes, _)
    return fill(Int32(size(a, ndims(a) - attributes.dimension)))
end

function handle(::Val{Symbol("stablehlo.optimization_barrier")}, operands, _, results)
    length(results) == 0 && return nothing
    return length(results) == 1 ? only(operands) : operands
end

# all operands are sorted together along `dimension` (an MLIR dimension, possibly negative) by the permutation that
# sorts their 1-dimensional slices with `comparator`, which takes `(lhs_1, rhs_1, lhs_2, rhs_2, ...)` as 0-dimensional
# arrays and returns whether lhs < rhs. the sort is always stable, which is valid for `is_stable = false` too
function handle(
    ::Val{Symbol("stablehlo.sort")}, operands, attributes, results, (comparator,)
)
    n = length(results)
    outs = map(copy, operands)
    N = ndims(first(outs))
    d = attributes.dimension
    j = N - (d < 0 ? d + N : d)
    kept = Tuple(k for k in 1:N if k != j)
    for vs in zip(map(o -> eachslice(o; dims=kept), outs)...)
        lt =
            (p, q) -> only(
                comparator(
                    ntuple(k -> fill(vs[(k + 1) >> 1][isodd(k) ? p : q]), Val(2n))...
                ),
            )[]
        perm = sortperm(eachindex(first(vs)); lt)
        foreach(v -> v .= v[perm], vs)
    end
    return n == 1 ? only(outs) : outs
end

struct Token end

# joins tokens into a new one; ops already run in program order, so there is nothing to wait for
handle(::Val{Symbol("stablehlo.after_all")}, _, _, _) = Token()

# TODO this can be veeery slow. fix it by using "mixed-radix cooley-tukey" for small factors, and bluestein/rader for
# large prime factors
# TODO also, plan FFTs ahead
# 1-dimensional discrete Fourier transform of `x` written into `out`, which must not alias `x`, unnormalized and with
# the sign of the exponent given by `inverse`. even lengths transform the even and odd elements of `x` into the two
# halves of `out` and combine them in place (radix-2 Cooley-Tukey), and odd lengths are computed directly in O(n^2)
function __fft_1d!(
    out::AbstractVector{T}, x::AbstractVector, inverse::Bool
) where {T<:Complex}
    n = length(x)
    if n <= 1
        n == 1 && (out[1] = x[1])
        return out
    end
    s = inverse ? 2 : -2
    if iseven(n)
        h = n ÷ 2
        __fft_1d!(view(out, 1:h), view(x, 1:2:n), inverse)
        __fft_1d!(view(out, (h + 1):n), view(x, 2:2:n), inverse)
        for k in 1:h
            e = out[k]
            t = T(cispi(s * (k - 1) / n)) * out[k + h]
            out[k] = e + t
            out[k + h] = e - t
        end
        return out
    end
    # the exponent is reduced modulo `n` to keep the twiddle factors accurate
    for k in 1:n
        out[k] = sum(x[j] * T(cispi(s * mod((j - 1) * (k - 1), n) / n)) for j in 1:n)
    end
    return out
end

# transforms the `length(fft_length)` minor-most MLIR dimensions, i.e. the first Julia dimensions, with
# `fft_length[end]` being the size of Julia dimension 1. inverse transforms are normalized by the product of
# `fft_length`. `RFFT` keeps the `n ÷ 2 + 1` non-redundant elements of Julia dimension 1, and `IRFFT` rebuilds the
# others from the Hermitian symmetry before its last (real) transform.
# the first transform reads the operand and writes the result, and the next ones work on the result in place through
# a scratch vector, since `__fft_1d!` can't transform in place
function handle(::Val{Symbol("stablehlo.fft")}, (a,), attributes, (result,))
    fft_type = attributes.fft_type
    L = length(attributes.fft_length)
    n = attributes.fft_length[end]
    inverse = fft_type === :IFFT || fft_type === :IRFFT
    C = complex(real(result.eltype))
    N = ndims(a)
    # vectors along Julia dimension `j`
    slices(x, j) = eachslice(x; dims=ntuple(k -> k < j ? k : k + 1, Val(N - 1)))

    out = Array{result.eltype}(undef, result.size)
    tmp = Vector{C}(undef, maximum(attributes.fft_length))
    full = Vector{C}(undef, n)

    if fft_type === :IRFFT
        # the complex transforms run first, so they need a complex copy of the operand unless there are none
        x = L > 1 ? Array{C}(a) : a
    elseif fft_type === :RFFT
        h = n ÷ 2 + 1
        for (w, v) in zip(slices(out, 1), slices(a, 1))
            copyto!(w, view(__fft_1d!(full, v, false), 1:h))
        end
        x = out
    else
        for (w, v) in zip(slices(out, 1), slices(a, 1))
            __fft_1d!(w, v, inverse)
        end
        x = out
    end

    for j in 2:L
        for w in slices(x, j)
            __fft_1d!(w, copyto!(view(tmp, 1:length(w)), w), inverse)
        end
    end

    if fft_type === :IRFFT
        for (w, v) in zip(slices(out, 1), slices(x, 1))
            for k in 1:n
                tmp[k] = k <= length(v) ? v[k] : conj(v[n - k + 2])
            end
            w .= real.(__fft_1d!(full, view(tmp, 1:n), true))
        end
    end
    if inverse
        out ./= prod(attributes.fft_length)
    end
    return out
end

function handle(::Val{Symbol("stablehlo.convolution")}, (lhs, rhs), attributes, (result,))
    dn = attributes.dimension_numbers
    nspatial = length(dn.input_spatial_dimensions)
    strides = get(attributes, :window_strides, ones(Int, nspatial))
    padding = get(attributes, :padding, zeros(Int, 2, nspatial)) # (low, high) × spatial dimension
    lhs_dilation = get(attributes, :lhs_dilation, ones(Int, nspatial))
    rhs_dilation = get(attributes, :rhs_dilation, ones(Int, nspatial))
    reversal = get(attributes, :window_reversal, falses(nspatial))
    G = get(attributes, :feature_group_count, 1)
    B = get(attributes, :batch_group_count, 1)

    out = zeros(result.eltype, result.size)
    NL, NR, NO = ndims(lhs), ndims(rhs), ndims(out)
    lsp = [NL - d for d in dn.input_spatial_dimensions]
    rsp = [NR - d for d in dn.kernel_spatial_dimensions]
    osp = [NO - d for d in dn.output_spatial_dimensions]
    lb, lf = NL - dn.input_batch_dimension, NL - dn.input_feature_dimension
    ri, ro = NR - dn.kernel_input_feature_dimension, NR - dn.kernel_output_feature_dimension
    ob, of = NO - dn.output_batch_dimension, NO - dn.output_feature_dimension

    cin = size(rhs, ri)
    cout = size(out, of)
    nbatch = size(lhs, lb) ÷ B
    window = CartesianIndices(ntuple(s -> size(rhs, rsp[s]), nspatial))
    lidx = zeros(Int, NL)
    ridx = zeros(Int, NR)
    for I in CartesianIndices(out)
        o = I[of] - 1
        lidx[lb] = (o ÷ (cout ÷ B)) * nbatch + I[ob]
        ridx[ro] = I[of]
        acc = zero(result.eltype)
        for K in window
            valid = true
            for s in 1:nspatial
                k = K[s] - 1
                # position in the padded and dilated lhs, which must fall on an element of the lhs
                q = (I[osp[s]] - 1) * strides[s] + k * rhs_dilation[s] - padding[1, s]
                if q < 0 ||
                    q % lhs_dilation[s] != 0 ||
                    q ÷ lhs_dilation[s] >= size(lhs, lsp[s])
                    valid = false
                    break
                end
                lidx[lsp[s]] = q ÷ lhs_dilation[s] + 1
                ridx[rsp[s]] = reversal[s] ? size(rhs, rsp[s]) - k : k + 1
            end
            valid || continue
            for c in 1:cin
                lidx[lf] = (o ÷ (cout ÷ G)) * cin + c
                ridx[ri] = c
                acc +=
                    lhs[CartesianIndex(ntuple(k -> lidx[k], NL))] *
                    rhs[CartesianIndex(ntuple(k -> ridx[k], NR))]
            end
        end
        out[I] = acc
    end
    return out
end

function handle(
    ::Val{Symbol("stablehlo.dynamic_conv")}, (lhs, rhs, padding), attributes, (result,)
)
    dn = attributes.dimension_numbers
    nspatial = length(dn.input_spatial_dimensions)
    pad = Matrix{Int}(padding)
    strides = get(attributes, :window_strides, ones(Int, nspatial))
    lhs_dilation = get(attributes, :lhs_dilation, ones(Int, nspatial))
    rhs_dilation = get(attributes, :rhs_dilation, ones(Int, nspatial))
    B = get(attributes, :batch_group_count, 1)

    NL, NR = ndims(lhs), ndims(rhs)
    mlir_dims = zeros(Int, length(result.size))
    mlir_dims[dn.output_batch_dimension + 1] = size(lhs, NL - dn.input_batch_dimension) ÷ B
    mlir_dims[dn.output_feature_dimension + 1] = size(
        rhs, NR - dn.kernel_output_feature_dimension
    )
    for s in 1:nspatial
        n = size(lhs, NL - dn.input_spatial_dimensions[s])
        k = size(rhs, NR - dn.kernel_spatial_dimensions[s])
        dilated = n == 0 ? 0 : (n - 1) * lhs_dilation[s] + 1
        window = k == 0 ? 0 : (k - 1) * rhs_dilation[s] + 1
        padded = dilated + pad[1, s] + pad[2, s]
        mlir_dims[dn.output_spatial_dimensions[s] + 1] =
            padded < window ? 0 : (padded - window) ÷ strides[s] + 1
    end

    dims = ntuple(j -> mlir_dims[end - j + 1], Val(length(result.size)))
    static = NamedTuple{(:eltype, :size),Tuple{Type{result.eltype},typeof(dims)}}((
        result.eltype, dims
    ))
    return handle(
        Val(Symbol("stablehlo.convolution")),
        (lhs, rhs),
        merge(attributes, (; padding=pad)),
        (static,),
    )
end

# Threefry-2x32 with 20 rounds (Salmon et al., "Parallel random numbers: as easy as 1, 2, 3", SC 2011)
function __threefry2x32(k0::UInt32, k1::UInt32, c0::UInt32, c1::UInt32)
    rotations = (13, 15, 26, 6, 17, 29, 16, 24)
    ks = (k0, k1, 0x1bd11bda ⊻ k0 ⊻ k1)
    x0, x1 = c0 + k0, c1 + k1
    for r in 0:19
        x0 += x1
        x1 = bitrotate(x1, rotations[r % 8 + 1]) ⊻ x0
        if r % 4 == 3
            i = (r + 1) ÷ 4
            x0 += ks[i % 3 + 1]
            x1 += ks[(i + 1) % 3 + 1] + UInt32(i)
        end
    end
    return (x0, x1)
end

# Philox-4x32 with 10 rounds, from the same paper
function __philox4x32(k0::UInt32, k1::UInt32, c::NTuple{4,UInt32})
    for r in 1:10
        p0 = UInt64(0xd2511f53) * c[1]
        p1 = UInt64(0xcd9e8d57) * c[3]
        c = (
            (p1 >> 32) % UInt32 ⊻ c[2] ⊻ k0,
            p1 % UInt32,
            (p0 >> 32) % UInt32 ⊻ c[4] ⊻ k1,
            p0 % UInt32,
        )
        k0 += 0x9e3779b9
        k1 += 0xbb67ae85
    end
    return c
end

function handle(
    ::Val{Symbol("stablehlo.rng_bit_generator")}, (state,), attributes, (_, result)
)
    algorithm = get(attributes, :rng_algorithm, :DEFAULT)
    algorithm in (:PHILOX, :DEFAULT, :THREE_FRY) ||
        throw(invalid_argument("unknown RNG algorithm $algorithm"))
    n = length(state)
    (algorithm === :THREE_FRY ? n == 2 : n in (2, 3)) ||
        throw(invalid_argument("$algorithm RNG state must not have $n elements"))

    key = state[1]
    k0, k1 = key % UInt32, (key >> 32) % UInt32
    lo, hi = state[2], n == 3 ? state[3] : zero(UInt64)

    out = Array{result.eltype}(undef, result.size)
    nbytes = sizeof(out)
    blocksize = algorithm === :THREE_FRY ? 2 : 4
    nblocks = cld(cld(nbytes, 4), blocksize)
    words = Vector{UInt32}(undef, nblocks * blocksize)
    for b in 0:(nblocks - 1)
        # 128-bit counter `(hi, lo) + b`
        clo = lo + UInt64(b)
        chi = hi + UInt64(clo < lo)
        block = if algorithm === :THREE_FRY
            __threefry2x32(k0, k1, clo % UInt32, (clo >> 32) % UInt32)
        else
            __philox4x32(
                k0,
                k1,
                (clo % UInt32, (clo >> 32) % UInt32, chi % UInt32, (chi >> 32) % UInt32),
            )
        end
        for (i, w) in enumerate(block)
            words[b * blocksize + i] = w
        end
    end
    bytes = view(reinterpret(UInt8, words), 1:nbytes)
    copyto!(vec(out), reinterpret(result.eltype, bytes))

    new_state = copy(state)
    new_state[2] = lo + UInt64(nblocks)
    if n == 3 && !(algorithm !== :THREE_FRY)
        new_state[3] = hi + UInt64(new_state[2] < lo)
    end
    return (new_state, out)
end

# convert an MLIR attribute into a Julia value that doesn't depend on its MLIR context.
# dense elements become arrays with reversed dimensions (like operands), dense arrays keep MLIR order (e.g. dimension
# indices are 0-based MLIR dimensions), enums become a `Symbol` with their value (e.g.
# `#stablehlo<comparison_direction LT>` becomes `:LT`), and attributes of any other kind become their textual form.
function decode_attribute(attr)
    # reads a list of dimensions through the size and element getters of the C API
    getdims(getsize, getelem) = Int64[getelem(attr, i) for i in 0:(getsize(attr) - 1)]

    if IR.isdenseelements(attr)
        T = IR.julia_type(eltype(IR.type(attr)))
        dims = reverse(size(IR.type(attr)))
        if IR.issplat(attr)
            splat = IR.Attribute(API.mlirDenseElementsAttrGetSplatValue(attr))
            x = if T === Bool
                Bool(splat)
            elseif T <: Integer
                T(Int64(splat))
            else
                T(Float64(splat))
            end
            return fill(x, dims)
        end

        # dense boolean attributes are bit-packed
        T === Bool &&
            throw(unimplemented("non-splat boolean dense attributes are not supported"))
        return copy(
            unsafe_wrap(Array, Ptr{T}(API.mlirDenseElementsAttrGetRawData(attr)), dims)
        )
    elseif IR.isdensearray(attr, Int64)
        return Int64[API.mlirDenseI64ArrayGetElement(attr, i) for i in 0:(length(attr) - 1)]
    elseif API.mlirAttributeIsADenseBoolArray(attr)
        return Bool[
            API.mlirDenseBoolArrayGetElement(attr, i) for
            i in 0:(API.mlirDenseArrayGetNumElements(attr) - 1)
        ]
    elseif IR.isbool(attr)
        return Bool(attr)
    elseif IR.isinteger(attr)
        return Int64(attr)
    elseif IR.isfloat(attr)
        return Float64(attr)
    elseif IR.isstring(attr)
        return String(attr)
    elseif API.stablehloAttributeIsADotDimensionNumbers(attr)
        return (;
            lhs_batching_dimensions=getdims(
                API.stablehloDotDimensionNumbersGetLhsBatchingDimensionsSize,
                API.stablehloDotDimensionNumbersGetLhsBatchingDimensionsElem,
            ),
            rhs_batching_dimensions=getdims(
                API.stablehloDotDimensionNumbersGetRhsBatchingDimensionsSize,
                API.stablehloDotDimensionNumbersGetRhsBatchingDimensionsElem,
            ),
            lhs_contracting_dimensions=getdims(
                API.stablehloDotDimensionNumbersGetLhsContractingDimensionsSize,
                API.stablehloDotDimensionNumbersGetLhsContractingDimensionsElem,
            ),
            rhs_contracting_dimensions=getdims(
                API.stablehloDotDimensionNumbersGetRhsContractingDimensionsSize,
                API.stablehloDotDimensionNumbersGetRhsContractingDimensionsElem,
            ),
        )
    elseif API.stablehloAttributeIsAGatherDimensionNumbers(attr)
        return (;
            offset_dims=getdims(
                API.stablehloGatherDimensionNumbersGetOffsetDimsSize,
                API.stablehloGatherDimensionNumbersGetOffsetDimsElem,
            ),
            collapsed_slice_dims=getdims(
                API.stablehloGatherDimensionNumbersGetCollapsedSliceDimsSize,
                API.stablehloGatherDimensionNumbersGetCollapsedSliceDimsElem,
            ),
            operand_batching_dims=getdims(
                API.stablehloGatherDimensionNumbersGetOperandBatchingDimsSize,
                API.stablehloGatherDimensionNumbersGetOperandBatchingDimsElem,
            ),
            start_indices_batching_dims=getdims(
                API.stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsSize,
                API.stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsElem,
            ),
            start_index_map=getdims(
                API.stablehloGatherDimensionNumbersGetStartIndexMapSize,
                API.stablehloGatherDimensionNumbersGetStartIndexMapElem,
            ),
            index_vector_dim=API.stablehloGatherDimensionNumbersGetIndexVectorDim(attr),
        )
    elseif API.stablehloAttributeIsAScatterDimensionNumbers(attr)
        return (;
            update_window_dims=getdims(
                API.stablehloScatterDimensionNumbersGetUpdateWindowDimsSize,
                API.stablehloScatterDimensionNumbersGetUpdateWindowDimsElem,
            ),
            inserted_window_dims=getdims(
                API.stablehloScatterDimensionNumbersGetInsertedWindowDimsSize,
                API.stablehloScatterDimensionNumbersGetInsertedWindowDimsElem,
            ),
            input_batching_dims=getdims(
                API.stablehloScatterDimensionNumbersGetInputBatchingDimsSize,
                API.stablehloScatterDimensionNumbersGetInputBatchingDimsElem,
            ),
            scatter_indices_batching_dims=getdims(
                API.stablehloScatterDimensionNumbersGetScatterIndicesBatchingDimsSize,
                API.stablehloScatterDimensionNumbersGetScatterIndicesBatchingDimsElem,
            ),
            scatter_dims_to_operand_dims=getdims(
                API.stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsSize,
                API.stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsElem,
            ),
            # the C API exports the scatter getter of `index_vector_dim` under this name
            index_vector_dim=API.stablehloDimensionNumbersGetIndexVectorDim(attr),
        )
    elseif API.stablehloAttributeIsAConvDimensionNumbers(attr)
        return (;
            input_batch_dimension=API.stablehloConvDimensionNumbersGetInputBatchDimension(
                attr
            ),
            input_feature_dimension=API.stablehloConvDimensionNumbersGetInputFeatureDimension(
                attr
            ),
            input_spatial_dimensions=getdims(
                API.stablehloConvDimensionNumbersGetInputSpatialDimensionsSize,
                API.stablehloConvDimensionNumbersGetInputSpatialDimensionsElem,
            ),
            kernel_input_feature_dimension=API.stablehloConvDimensionNumbersGetKernelInputFeatureDimension(
                attr
            ),
            kernel_output_feature_dimension=API.stablehloConvDimensionNumbersGetKernelOutputFeatureDimension(
                attr
            ),
            kernel_spatial_dimensions=getdims(
                API.stablehloConvDimensionNumbersGetKernelSpatialDimensionsSize,
                API.stablehloConvDimensionNumbersGetKernelSpatialDimensionsElem,
            ),
            output_batch_dimension=API.stablehloConvDimensionNumbersGetOutputBatchDimension(
                attr
            ),
            output_feature_dimension=API.stablehloConvDimensionNumbersGetOutputFeatureDimension(
                attr
            ),
            output_spatial_dimensions=getdims(
                API.stablehloConvDimensionNumbersGetOutputSpatialDimensionsSize,
                API.stablehloConvDimensionNumbersGetOutputSpatialDimensionsElem,
            ),
        )
    elseif IR.isarray(attr)
        return Any[
            decode_attribute(IR.Attribute(API.mlirArrayAttrGetElement(attr, i))) for
            i in 0:(API.mlirArrayAttrGetNumElements(attr) - 1)
        ]
    end

    str = string(attr)
    m = match(r"^#\w+<\w+ (\w+)>$", str)
    return isnothing(m) ? str : Symbol(m[1])
end

## HostClient
mutable struct HostClient <: Client
    devices::Vector{Device}
    memories::Vector{Memory}
end

function HostClient()
    devs = Vector{Device}(undef, 1)
    mems = Vector{Memory}(undef, 1)

    dev = HostDevice(HostDeviceDescription(), -1, mems)
    mem = HostMemory(0, "device", devs)

    devs[1] = dev
    mems[1] = mem

    return HostClient(devs, mems)
end

platform_name(::HostClient) = "julia-cpu"
platform_version(::HostClient) = string(VERSION)
process_index(::HostClient) = 0
devices(client::HostClient) = client.devices
addressable_devices(client::HostClient) = devices(client)
addressable_memories(client::HostClient) = client.memories

function buffer_from_host_buffer(
    ::HostClient,
    dev::HostDevice,
    mem::HostMemory,
    data,
    _eltype,
    _size,
    byte_strides,
    _layout,
    _host_buffer_semantics,
)
    _elsize = pjrt_elsize(_eltype)
    _ndims = length(_size)

    size_in_bytes = prod(_size; init=_elsize)
    minor_to_major = collect((_ndims - 1):-1:0)

    # TODO support byte strides, layout and host buffer semantics

    # only dense major-to-minor host data is supported; strides of size-1 dims are irrelevant
    # num_byte_strides = Int(unsafe_load(args.num_byte_strides))
    # if num_byte_strides != 0
    #     num_byte_strides != num_dims && throw(invalid_argument())
    #     byte_strides = unsafe_load(args.byte_strides)
    #     stride = elsize
    #     for i in num_dims:-1:1
    #         dim = unsafe_load(host_dims, i)
    #         dim != 1 && unsafe_load(byte_strides, i) != stride && throw(unimplemented())
    #         stride *= dim
    #     end
    # end

    # relayout is not supported, so `device_layout` must be dense major-to-minor
    # if device_layout != C_NULL
    #     layout_type = unsafe_load(device_layout.type)
    #     layout_type != CAPI.PJRT_Buffer_MemoryLayout_Type_Tiled && throw(unimplemented())

    #     tiled = unsafe_load(device_layout.tiled)
    #     tiled.num_tiles != 0 && throw(unimplemented())
    #     tiled.minor_to_major_size != num_dims && throw(invalid_argument())
    #     for i in 1:num_dims
    #         unsafe_load(tiled.minor_to_major, i) != num_dims - i && throw(unimplemented())
    #     end
    # end

    # the copy is synchronous, so all `host_buffer_semantics` are honored
    new_data = Libc.malloc(size_in_bytes)
    new_data == C_NULL && throw(resource_exhausted())

    Libc.memcpy(new_data, data, size_in_bytes)

    return HostBuffer(new_data, size_in_bytes, _eltype, _size, minor_to_major, dev, mem)
end

# like `buffer_from_host_buffer`, buffers are dense in major-to-minor order and `layout` is not checked
function create_uninitialized_buffer(
    ::HostClient, dev::HostDevice, mem::HostMemory, _eltype, _size, _layout
)
    size_in_bytes = prod(_size; init=pjrt_elsize(_eltype))
    minor_to_major = collect((length(_size) - 1):-1:0)
    # `malloc(0)` may return `C_NULL`, which would mark the buffer as deleted
    data = Libc.malloc(max(size_in_bytes, 1))
    data == C_NULL && throw(resource_exhausted())
    return HostBuffer(data, size_in_bytes, _eltype, _size, minor_to_major, dev, mem)
end

# each op becomes a call to its handler, `%N = handle(Val(opname), operands, attributes, results)`
# like in MLIR, an op with several results binds `%N#0`, `%N#1`, ... instead.
# ops with regions also get `regions`, a tuple with an anonymous function per region, which takes the block
# arguments and returns a tuple with the operands of the region terminator.
# `codegen!` translates the ops of `block` into `exprs` and returns the operands of its terminator. `vars` maps MLIR
# values to the names of their Julia variables, and `nvals` counts the ops with results to name them
function codegen!(exprs, block, vars, nvals)
    for op in block
        opname = IR.name(op)
        operands = [vars[IR.operand(op, i).ref.ptr] for i in 1:IR.noperands(op)]
        opname in ("func.return", "stablehlo.return") && return operands
        codegen_op!(exprs, op, opname, operands, vars, nvals)
    end
    throw(invalid_argument("block has no terminator"))
end

function codegen_op!(exprs, op, opname, operands, vars, nvals)
    opval = Val(Symbol(opname))
    nregions = IR.nregions(op)
    signature = if nregions == 0
        Tuple{typeof(opval),Tuple,NamedTuple,Tuple}
    else
        Tuple{typeof(opval),Tuple,NamedTuple,Tuple,Tuple}
    end
    hasmethod(handle, signature) ||
        throw(unimplemented("`$opname` is not supported by `HostClient`"))

    # attributes are decoded here, since the MLIR context doesn't outlive `compile!`
    attributes = (;
        (
            Symbol(String(API.mlirIdentifierStr(na.ref.name))) =>
                decode_attribute(IR.Attribute(na.ref.attribute)) for
            na in (IR.getattr(op, i) for i in 1:IR.nattrs(op))
        )...
    )
    nres = IR.nresults(op)
    results = ntuple(nres) do i
        type = IR.type(IR.result(op, i))
        API.stablehloTypeIsAToken(type) && return Token
        IR.istensor(type) || throw(
            unimplemented(
                "`$opname` returns a value that is neither a tensor nor a token, which is not supported",
            ),
        )
        T = IR.julia_type(eltype(type))
        dims = reverse(size(type))

        # `eltype` is typed as `Type{T}` instead of `DataType`, so that handlers that build their result from it
        # (e.g. `Array{result.eltype}`) are type-stable
        NamedTuple{(:eltype, :size),Tuple{Type{T},typeof(dims)}}((T, dims))
    end

    call = :(handle($opval, ($(operands...),), $attributes, $results))
    if nregions > 0
        id = nvals[]
        nvals[] += 1
        regions = map(1:nregions) do r
            rblock = IR.first_block(IR.region(op, r))
            args = map(1:IR.nargs(rblock)) do i
                vars[IR.argument(rblock, i).ref.ptr] = Symbol(
                    "%", id, "^", r - 1, "#arg", i - 1
                )
            end
            rexprs = Expr(:block)
            rets = codegen!(rexprs, rblock, vars, nvals)
            Expr(
                :->,
                Expr(:tuple, args...),
                Expr(:block, rexprs.args..., :(return ($(rets...),))),
            )
        end
        push!(call.args, Expr(:tuple, regions...))
    end

    if nres == 0
        push!(exprs.args, call)
    elseif nres == 1
        var = vars[IR.result(op, 1).ref.ptr] = Symbol("%", nvals[])
        push!(exprs.args, :($var = $call))
        nvals[] += 1
    else
        resvars = map(1:nres) do i
            vars[IR.result(op, i).ref.ptr] = Symbol("%", nvals[], "#", i - 1)
        end
        push!(exprs.args, :(($(resvars...),) = $call::NTuple{$nres,Any}))
        nvals[] += 1
    end
    return nothing
end

function compile!(client::HostClient, code, options)
    # `code` lives in a context owned by the caller, so only copies of its contents are kept
    main = IR.lookup(IR.SymbolTable(code), "main")
    isnothing(main) && throw(invalid_argument("program has no `main` function"))
    ftype = IR.FunctionType(main)
    output_types = map(1:IR.nresults(ftype)) do i
        pjrt_eltype(IR.julia_type(eltype(IR.result(ftype, i))))
    end

    name_attr = IR.getattr(IR.Operation(code), "sym_name")
    _name = isnothing(name_attr) ? "main" : String(name_attr)

    exe = HostExecutable(_name, string(code), output_types, options)
    if num_replicas(exe) * num_partitions(exe) != 1
        throw(unimplemented("`HostClient` only supports single-device executables"))
    end

    # run on the device given by the device assignment, or on the first one if there is none
    build_options = options.executable_build_options
    assignment = isnothing(build_options) ? nothing : build_options.device_assignment
    dev = if isnothing(assignment)
        first(client.devices)
    else
        dev_id = only(only(assignment.computation_devices).replica_device_ids)
        idx = findfirst(d -> id(description(d)) == dev_id, client.devices)
        isnothing(idx) &&
            throw(invalid_argument("device $dev_id is not addressable by `HostClient`"))
        client.devices[idx]
    end

    # translate `@main` op by op into the body of an `execute` method specific to this executable.
    # buffers are row-major, so they are seen from Julia as column-major arrays with reversed dimensions.
    block = IR.first_block(IR.region(main, 1))
    nargs = IR.nargs(block)
    vars = Dict{Ptr{Cvoid},Symbol}()
    body = quote
        length(arguments) == 1 || throw(
            invalid_argument(
                string("expected arguments for 1 device, got ", length(arguments))
            ),
        )
        isnothing(dev) ||
            dev === exec.devices[1] ||
            throw(invalid_argument("cannot execute on another device"))
        buffers = only(arguments)
        length(buffers) == $nargs || throw(
            invalid_argument(
                string("expected ", $nargs, " arguments, got ", length(buffers))
            ),
        )
    end

    # wrap operand buffers into Array without copying
    # TODO support wrapping other MLIR types apart of `tensor`
    for i in 1:nargs
        arg = IR.argument(block, i)
        T = IR.julia_type(eltype(IR.type(arg)))
        dims = reverse(size(IR.type(arg)))
        var = vars[arg.ref.ptr] = Symbol("%arg", i - 1)
        push!(
            body.args,
            quote
                isdeleted(buffers[$i]) &&
                    throw(invalid_argument(string("argument ", $i, " is deleted")))
                $var = unsafe_wrap(Array, Ptr{$T}((buffers[$i]::HostBuffer).data), $dims)
            end,
        )
    end

    # each op becomes a call to its handler
    outputs = codegen!(body, block, vars, Ref(0))

    # outputs are copied into new buffers, since they may alias inputs or embedded constants
    # TODO support buffer donation?
    push!(
        body.args,
        quote
            out_dev = exec.devices[1]
            out_mem = default_memory(out_dev)
            results = Buffer[]
            for out in ($(outputs...),)
                nbytes = sizeof(out::Array)
                data = Libc.malloc(max(nbytes, 1))
                data == C_NULL && throw(resource_exhausted())
                GC.@preserve out unsafe_copyto!(
                    Ptr{UInt8}(data), Ptr{UInt8}(pointer(out)), nbytes
                )
                dims = collect(Int64, reverse(size(out)))
                minor_to_major = collect(Int64, (ndims(out) - 1):-1:0)
                push!(
                    results,
                    HostBuffer(
                        data,
                        nbytes,
                        pjrt_eltype(eltype(out)),
                        dims,
                        minor_to_major,
                        out_dev,
                        out_mem,
                    ),
                )
            end
            return [results]
        end,
    )

    tag = gensym(:main)
    @eval function execute(
        exec::HostLoadedExecutable{$(QuoteNode(tag))},
        arguments::Vector{Vector{Buffer}},
        dev,
    )
        return $body
    end

    # single-device executables run as replica 0 of partition 0
    logical_ids = [CAPI.PJRT_LogicalDeviceIds(0, 0)]
    return HostLoadedExecutable{tag}(exe, Device[dev], logical_ids, assignment, false)
end

const plugin_api = Ref{PluginApi}()

function make_pjrt_client(;
    node_id::Integer=0,
    num_nodes::Integer=1,
    distributed_runtime_client=nothing,
    allowed_devices::Union{Nothing,Vector{Int}}=nothing,
)
    @assert node_id == 0
    @assert num_nodes == 1 "`make_pjrt_client` does not support num_nodes > 1"
    @assert isnothing(distributed_runtime_client) "`make_pjrt_client` does not \
                                                    support distributed_runtime_client"
    @assert isnothing(allowed_devices)

    errstr = Ref{Cstring}()
    api = unsafe_convert(Ptr{CAPI.PJRT_Api}, plugin_api[])
    ptr = API.MakeClientFromApi(api, "julia", "JULIA", errstr)
    ptr == C_NULL && error(unsafe_string(errstr[]))
    return Reactant.XLA.PJRT.Client(ptr)
end

function make_ifrt_client(; kwargs...)
    pjrt_client = make_pjrt_client(; kwargs...)
    errstr = Ref{Cstring}()
    ptr = GC.@preserve pjrt_client errstr begin
        API.ifrt_pjrt_make_client_with_default_kv_store(
            pjrt_client.client, 0, 1, C_NULL, errstr, "julia"
        )
    end
    ptr == C_NULL && error(unsafe_string(errstr[]))
    return Reactant.XLA.IFRT.Client(ptr)
end

function __init__()
    plugin_api[] = make_api(HostClient)

    Reactant.Accelerators.Registration.register_backend(
        "julia";
        priority=100,
        pjrt_initialize_function=make_pjrt_client,
        ifrt_initialize_function=make_ifrt_client,
    )

    return nothing
end

end

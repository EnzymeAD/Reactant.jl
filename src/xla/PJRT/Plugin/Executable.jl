"""
    Executable

A compiled program, independent of the devices it runs on. Returned by `executable(::LoadedExecutable)`.

### Interface

- `name`
- `nreplicas` (defaults to the one in `compile_options`)
- `npartitions` (defaults to the one in `compile_options`)
- `noutputs` (defaults to the length of `output_pjrt_eltypes`)
- `codesize`
- `fingerprint`
- `cost_analysis`
- `output_eltypes`
- `output_pjrt_eltypes` (defaults to `map(pjrt_eltype, output_eltypes(exe))`)
- `output_sizes`
- `parameter_memory_kinds`
- `output_memory_kinds`
- `optimized_program`
- `serialize`
- `compile_options`
- `compiled_memory_stats`
"""
abstract type Executable end

@pjrt_catch_return function pjrt_executable_destroy(
    args::Ptr{CAPI.PJRT_Executable_Destroy_Args}
)::Ptr{CAPI.PJRT_Error}
    ptr = unsafe_load(args.executable)
    ptr != C_NULL && unroot!(ptr)
    return ok()
end

function pjrt_serialized_bytes_deleter(ptr::Ptr{Cvoid})::Cvoid
    ptr != C_NULL && unroot!(ptr)
    return nothing
end

"""
    name(exe::Executable) -> String

Return a string that identifies `exe` (e.g. the name of the compiled module).

The returned value doesn't need to be owned by `exe`, but it must be the same on every call: the C wrapper calls it at
most once per `exe` and keeps a copy for as long as `exe` is alive, since C keeps pointers to it.

!!! warning
`PjRtCApiExecutable` treats any error as fatal.

Backs `PJRT_Executable_Name`.
"""
name(::Executable)

# C borrows the string, so it is kept for as long as the executable is alive
const executable_name_cache = WeakKeyIdDict{Executable,String}()

@pjrt_catch_return function pjrt_executable_name(
    args::Ptr{CAPI.PJRT_Executable_Name_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    str = get!(executable_name_cache, exe) do
        String(name(exe))
    end
    unsafe_store!(args.executable_name, Cstring(pointer(str)))
    unsafe_store!(args.executable_name_size, sizeof(str))
    return ok()
end

"""
    nreplicas(exe::Executable) -> Integer

Return the number of replicas of `exe`, i.e. `executable_build_options.num_replicas` of the `CompileOptionsProto` it
was compiled with.

Defaults to the one in `compile_options(exe)`, or `1` if it's not set.

!!! warning
`PjRtCApiExecutable` treats any error as fatal.

Backs `PJRT_Executable_NumReplicas`.
"""
function nreplicas(exe::Executable)
    options = compile_options(exe)::Reactant.Proto.xla.CompileOptionsProto
    build_options = options.executable_build_options
    return isnothing(build_options) ? 1 : max(build_options.num_replicas, 1)
end

@pjrt_catch_return function pjrt_executable_num_replicas(
    args::Ptr{CAPI.PJRT_Executable_NumReplicas_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    unsafe_store!(args.num_replicas, nreplicas(exe))
    return ok()
end

"""
    npartitions(exe::Executable) -> Integer

Return the number of partitions of `exe`, i.e. `executable_build_options.num_partitions` of the `CompileOptionsProto`
it was compiled with.

Defaults to the one in `compile_options(exe)`, or `1` if it's not set.

!!! warning
`PjRtCApiExecutable` treats any error as fatal.

Backs `PJRT_Executable_NumPartitions`.
"""
function npartitions(exe::Executable)
    options = compile_options(exe)::Reactant.Proto.xla.CompileOptionsProto
    build_options = options.executable_build_options
    return isnothing(build_options) ? 1 : max(build_options.num_partitions, 1)
end

@pjrt_catch_return function pjrt_executable_num_partitions(
    args::Ptr{CAPI.PJRT_Executable_NumPartitions_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    unsafe_store!(args.num_partitions, npartitions(exe))
    return ok()
end

"""
    noutputs(exe::Executable) -> Integer

Return the number of outputs that `exe` produces per device.

`PjRtCApiLoadedExecutable` uses it to allocate the output lists filled by `execute`, so it must match the length of
each output vector that `execute` returns.

Defaults to `length(output_pjrt_eltypes(exe))`.

Backs `PJRT_Executable_NumOutputs`.
"""
noutputs(exe::Executable) = length(output_pjrt_eltypes(exe))

@pjrt_catch_return function pjrt_executable_num_outputs(
    args::Ptr{CAPI.PJRT_Executable_NumOutputs_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    unsafe_store!(args.num_outputs, noutputs(exe))
    return ok()
end

"""
    codesize(exe::Executable) -> Integer

Return the size in bytes of the code generated for `exe`, or `0` if unknown.

!!! warning
`PjRtCApiExecutable` treats any error as fatal.

Backs `PJRT_Executable_SizeOfGeneratedCodeInBytes`.
"""
codesize(::Executable)

@pjrt_catch_return function pjrt_executable_size_of_generated_code_in_bytes(
    args::Ptr{CAPI.PJRT_Executable_SizeOfGeneratedCodeInBytes_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    unsafe_store!(args.size_in_bytes, codesize(exe))
    return ok()
end

"""
    fingerprint(exe::Executable) -> String

Return a fingerprint of `exe`. Executables compiled from identical inputs (program, compile options, compiler
version, ...) must have the same fingerprint. Throw `unimplemented` if not supported.

The returned value doesn't need to be owned by `exe`, but it must be the same on every call: the C wrapper calls it at
most once per `exe` and keeps a copy for as long as `exe` is alive, since C keeps pointers to it.

Backs `PJRT_Executable_Fingerprint`.
"""
fingerprint(::Executable)

# C borrows the string, so it is kept for as long as the executable is alive
const executable_fingerprint_cache = WeakKeyIdDict{Executable,String}()

@pjrt_catch_return function pjrt_executable_fingerprint(
    args::Ptr{CAPI.PJRT_Executable_Fingerprint_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    str = get!(executable_fingerprint_cache, exe) do
        String(fingerprint(exe))
    end
    unsafe_store!(args.executable_fingerprint, Cstring(pointer(str)))
    unsafe_store!(args.executable_fingerprint_size, sizeof(str))
    return ok()
end

# C views of the results of `cost_analysis`, which C keeps for as long as the executable is alive
const cost_analysis_cache = WeakKeyIdDict{
    Executable,Tuple{Vector{CAPI.PJRT_NamedValue},Vector{Any}}
}()

"""
    cost_analysis(exe::Executable)

Return the cost analysis of `exe` as `name => value` pairs (e.g. `"flops" => 1.0e6`, `"bytes accessed" => 4096`).
Values must be `AbstractString`, `Bool`, `Integer`, `AbstractVector{<:Integer}` or `AbstractFloat`, which C receives as
`String`, `Bool`, `Int64`, `Vector{Int64}` and `Float32`.

The returned value doesn't need to be owned by `exe`, but it must be the same on every call: the C wrapper calls it at
most once per `exe` and keeps a copy for as long as `exe` is alive, since C keeps pointers to it.

Backs `PJRT_Executable_GetCostAnalysis`.
"""
cost_analysis(::Executable)

@pjrt_catch_return function pjrt_executable_get_cost_analysis(
    args::Ptr{CAPI.PJRT_Executable_GetCostAnalysis_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    props, _ = get!(cost_analysis_cache, exe) do
        pjrt_named_values(cost_analysis(exe))
    end
    unsafe_store!(args.num_properties, length(props))
    unsafe_store!(args.properties, isempty(props) ? C_NULL : pointer(props))
    return ok()
end

"""
    output_eltypes(exe::Executable) -> Vector{<:Type}

Return the Julia element type of each output of `exe`.

Backs `PJRT_Executable_OutputElementTypes` through the default of [`output_pjrt_eltypes`](@ref).
"""
output_eltypes(::Executable)

"""
    output_pjrt_eltypes(exe::Executable) -> Vector{CAPI.PJRT_Buffer_Type}

Return the PJRT element type of each output of `exe`.

Defaults to `map(pjrt_eltype, output_eltypes(exe))`. Override it if an element type has no Julia equivalent, or if
`exe` already stores the PJRT element types.

The returned value doesn't need to be owned by `exe`, but it must be the same on every call: the C wrapper calls it at
most once per `exe` and keeps a copy for as long as `exe` is alive, since C keeps pointers to it.

Backs `PJRT_Executable_OutputElementTypes`.
"""
output_pjrt_eltypes(exe::Executable) = map(pjrt_eltype, output_eltypes(exe))

# C borrows the element types, so they are kept for as long as the executable is alive
const output_pjrt_eltypes_cache = WeakKeyIdDict{Executable,Vector{CAPI.PJRT_Buffer_Type}}()

@pjrt_catch_return function pjrt_executable_output_element_types(
    args::Ptr{CAPI.PJRT_Executable_OutputElementTypes_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    types = get!(output_pjrt_eltypes_cache, exe) do
        collect(CAPI.PJRT_Buffer_Type, output_pjrt_eltypes(exe))
    end
    unsafe_store!(args.output_types, pointer(types))
    unsafe_store!(args.num_output_types, length(types))
    return ok()
end

"""
    output_sizes(exe::Executable) -> Vector{<:Dims}

Return the dimensions of each output of `exe`, like `Base.size`.

The returned value doesn't need to be owned by `exe`, but it must be the same on every call: the C wrapper calls it at
most once per `exe` and keeps a copy for as long as `exe` is alive, since C keeps pointers to it.

Backs `PJRT_Executable_OutputDimensions`.
"""
output_sizes(::Executable)

const output_sizes_cache = WeakKeyIdDict{Executable,Tuple{Vector{Int64},Vector{Csize_t}}}()

@pjrt_catch_return function pjrt_executable_output_dimensions(
    args::Ptr{CAPI.PJRT_Executable_OutputDimensions_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    # C expects the dimensions of all outputs back to back, plus the number of dimensions of each output
    dims, ranks = get!(output_sizes_cache, exe) do
        shapes = output_sizes(exe)::Vector{<:Dims}
        (
            Int64[d for shape in shapes for d in shape],
            Csize_t[length(shape) for shape in shapes],
        )
    end
    unsafe_store!(args.num_outputs, length(ranks))
    unsafe_store!(args.dims, pointer(dims))
    unsafe_store!(args.dim_sizes, pointer(ranks))
    return ok()
end

"""
    parameter_memory_kinds(exe::Executable) -> Vector{String}

Return the memory kind of each parameter of `exe` (e.g. `"device"`, `"pinned_host"`).

The returned value doesn't need to be owned by `exe`, but it must be the same on every call: the C wrapper calls it at
most once per `exe` and keeps a copy for as long as `exe` is alive, since C keeps pointers to it.

Backs `PJRT_Executable_ParameterMemoryKinds`.
"""
parameter_memory_kinds(::Executable)

const parameter_memory_kinds_cache = WeakKeyIdDict{
    Executable,Tuple{Vector{String},Vector{Cstring},Vector{Csize_t}}
}()

@pjrt_catch_return function pjrt_executable_parameter_memory_kinds(
    args::Ptr{CAPI.PJRT_Executable_ParameterMemoryKinds_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    _, ptrs, sizes = get!(parameter_memory_kinds_cache, exe) do
        kinds = parameter_memory_kinds(exe)::Vector{String}
        (kinds, [Cstring(pointer(s)) for s in kinds], Csize_t[sizeof(s) for s in kinds])
    end
    unsafe_store!(args.num_parameters, length(ptrs))
    unsafe_store!(args.memory_kinds, pointer(ptrs))
    unsafe_store!(args.memory_kind_sizes, pointer(sizes))
    return ok()
end

"""
    output_memory_kinds(exe::Executable) -> Vector{String}

Return the memory kind of each output of `exe` (e.g. `"device"`, `"pinned_host"`).

The returned value doesn't need to be owned by `exe`, but it must be the same on every call: the C wrapper calls it at
most once per `exe` and keeps a copy for as long as `exe` is alive, since C keeps pointers to it.

Backs `PJRT_Executable_OutputMemoryKinds`.
"""
output_memory_kinds(::Executable)

const output_memory_kinds_cache = WeakKeyIdDict{
    Executable,Tuple{Vector{String},Vector{Cstring},Vector{Csize_t}}
}()

@pjrt_catch_return function pjrt_executable_output_memory_kinds(
    args::Ptr{CAPI.PJRT_Executable_OutputMemoryKinds_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    _, ptrs, sizes = get!(output_memory_kinds_cache, exe) do
        kinds = output_memory_kinds(exe)::Vector{String}
        (kinds, [Cstring(pointer(s)) for s in kinds], Csize_t[sizeof(s) for s in kinds])
    end
    unsafe_store!(args.num_outputs, length(ptrs))
    unsafe_store!(args.memory_kinds, pointer(ptrs))
    unsafe_store!(args.memory_kind_sizes, pointer(sizes))
    return ok()
end

"""
    optimized_program(exe::Executable) -> String

Return the optimized program of `exe` as MLIR text (StableHLO dialect).

`PJRT_Executable_OptimizedProgram` is called twice per query (first for the size, then for the data), so this must
return the same string every time.

Backs `PJRT_Executable_OptimizedProgram`, which `PjRtCApiExecutable` uses for `GetHloModules` and for the default
output and parameter shardings.
"""
optimized_program(::Executable)

@pjrt_catch_return function pjrt_executable_optimized_program(
    args::Ptr{CAPI.PJRT_Executable_OptimizedProgram_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    code = optimized_program(exe)::String
    program = unsafe_load(args.program)

    # string literals are never freed, so the implementation keeps owning `format`
    format = "mlir"
    unsafe_store!(program.format, Cstring(pointer(format)))
    unsafe_store!(program.format_size, sizeof(format))

    # first call: only report the size of the buffer that the caller must allocate
    code_ptr = Ptr{UInt8}(unsafe_load(program.code))
    if code_ptr == C_NULL
        unsafe_store!(program.code_size, sizeof(code))
        return ok()
    end

    # second call: `code_size` holds the size of the caller's buffer
    capacity = unsafe_load(program.code_size)
    if capacity < sizeof(code)
        throw(
            invalid_argument(
                "program buffer of $capacity bytes is smaller than the $(sizeof(code)) bytes required",
            ),
        )
    end
    GC.@preserve code unsafe_copyto!(code_ptr, pointer(code), sizeof(code))
    unsafe_store!(program.code_size, sizeof(code))
    return ok()
end

"""
    serialize(exe::Executable) -> Vector{UInt8}

Return a platform-specific serialization of `exe`. It doesn't need to be stable across versions.

The returned vector is GC rooted until C calls its deleter, so it must be a new vector that `exe` doesn't hand out
anywhere else.

Backs `PJRT_Executable_Serialize`.
"""
serialize(::Executable)

@pjrt_catch_return function pjrt_executable_serialize(
    args::Ptr{CAPI.PJRT_Executable_Serialize_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    bytes = root!(serialize(exe)::Vector{UInt8})
    unsafe_store!(args.serialized_bytes, Cstring(pointer(bytes)))
    unsafe_store!(args.serialized_bytes_size, length(bytes))
    unsafe_store!(args.serialized_executable, pointer_from_objref(bytes))
    unsafe_store!(
        args.serialized_executable_deleter,
        @cfunction(pjrt_serialized_bytes_deleter, Cvoid, (Ptr{Cvoid},))
    )
    return ok()
end

"""
    compile_options(exe::Executable) -> Reactant.Proto.xla.CompileOptionsProto

Return the compile options that `exe` was compiled with, i.e. the ones given to `compile`.

Backs `PJRT_Executable_GetCompileOptions`.
"""
compile_options(::Executable)

@pjrt_catch_return function pjrt_executable_get_compile_options(
    args::Ptr{CAPI.PJRT_Executable_GetCompileOptions_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    proto = compile_options(exe)::Reactant.Proto.xla.CompileOptionsProto
    bytes = root!(Reactant.ProtoUtils.proto_to_bytes(proto))
    unsafe_store!(args.serialized_bytes, Cstring(pointer(bytes)))
    unsafe_store!(args.serialized_bytes_size, length(bytes))
    unsafe_store!(args.serialized_compile_options, pointer_from_objref(bytes))
    unsafe_store!(
        args.serialized_compile_options_deleter,
        @cfunction(pjrt_serialized_bytes_deleter, Cvoid, (Ptr{Cvoid},))
    )
    return ok()
end

"""
    compiled_memory_stats(exe::Executable)

Return the memory usage of `exe`, mirroring `xla::CompiledMemoryStats`, as an object (e.g. a `NamedTuple`) with the
following integer properties:

- device memory: `generated_code_size_in_bytes`, `argument_size_in_bytes`, `output_size_in_bytes`,
  `alias_size_in_bytes` (how much of the arguments is reused for the outputs), `temp_size_in_bytes`
- host memory: `host_generated_code_size_in_bytes`, `host_argument_size_in_bytes`, `host_output_size_in_bytes`,
  `host_alias_size_in_bytes`, `host_temp_size_in_bytes`
- device memory totals: `peak_memory_in_bytes`, `total_size_in_bytes`, `total_allocation_bytes`,
  `indefinite_allocations`, `peak_unpadded_heap_bytes`

Backs `PJRT_Executable_GetCompiledMemoryStats`.
"""
compiled_memory_stats(::Executable)

@pjrt_catch_return function pjrt_executable_get_compiled_memory_stats(
    args::Ptr{CAPI.PJRT_Executable_GetCompiledMemoryStats_Args}
)::Ptr{CAPI.PJRT_Error}
    exe = unsafe_pointer_to_objref(unsafe_load(args.executable))::Executable
    stats = compiled_memory_stats(exe)
    # skip `struct_size`, `extension_start` and `executable`
    for field in fieldnames(CAPI.PJRT_Executable_GetCompiledMemoryStats_Args)[4:end]
        unsafe_store!(getproperty(args, field), getproperty(stats, field))
    end
    return ok()
end

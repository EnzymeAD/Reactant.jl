# sizes are MLIR dimensions
mutable struct MetalExecutable <: Executable
    const name::String
    const program::String
    const input_types::Vector{DataType}
    const input_sizes::Vector{Vector{Int64}}
    const output_types::Vector{DataType}
    const output_sizes::Vector{Vector{Int64}}
    const options::Reactant.Proto.xla.CompileOptionsProto
end

name(exe::MetalExecutable) = exe.name
codesize(::MetalExecutable) = 0
function fingerprint(::MetalExecutable)
    throw(unimplemented("`MetalExecutable` has no fingerprint"))
end
output_eltypes(exe::MetalExecutable) = exe.output_types
output_sizes(exe::MetalExecutable) = Dims[Tuple(dims) for dims in exe.output_sizes]
optimized_program(exe::MetalExecutable) = exe.program
function serialize(::MetalExecutable)
    throw(unimplemented("serialization of `MetalExecutable` is not supported yet"))
end
compile_options(exe::MetalExecutable) = exe.options

mutable struct HostExecutable <: Executable
    const name::String
    const program::String
    const output_types::Vector{DataType}
    const options::Reactant.Proto.xla.CompileOptionsProto
end

name(exe::HostExecutable) = exe.name
codesize(::HostExecutable) = 0
fingerprint(::HostExecutable) = throw(unimplemented("`HostExecutable` has no fingerprint"))
output_eltypes(exe::HostExecutable) = exe.output_types
optimized_program(exe::HostExecutable) = exe.program
function serialize(::HostExecutable)
    throw(unimplemented("serialization of `HostExecutable` is not supported yet"))
end
compile_options(exe::HostExecutable) = exe.options

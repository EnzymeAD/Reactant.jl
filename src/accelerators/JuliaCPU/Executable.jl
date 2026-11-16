mutable struct JuliaExecutable <: Executable
    const name::String
    const program::String
    const output_types::Vector{DataType}
    const options::Reactant.Proto.xla.CompileOptionsProto
end

name(exe::JuliaExecutable) = exe.name
codesize(::JuliaExecutable) = 0
function fingerprint(::JuliaExecutable)
    throw(unimplemented("`JuliaExecutable` has no fingerprint"))
end
output_eltypes(exe::JuliaExecutable) = exe.output_types
optimized_program(exe::JuliaExecutable) = exe.program
function serialize(::JuliaExecutable)
    throw(unimplemented("serialization of `JuliaExecutable` is not supported yet"))
end
compile_options(exe::JuliaExecutable) = exe.options

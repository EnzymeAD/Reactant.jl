SetLogLevel(x) = @ccall MLIR.API.mlir_c.SetLogLevel(x::Cint)::Cvoid

struct ReactantInternalError <: Base.Exception
    msg::String
end

function Base.showerror(io::IO, ece::ReactantInternalError)
    return print(io, ece.msg, '\n')
end

function reactant_err(msg::Cstring)::Cvoid
    throw(ReactantInternalError(Base.unsafe_string(msg)))
end

# https://github.com/openxla/xla/blob/4bfb5c82a427151d6fe5acad8ebe12cee403036a/xla/xla_data.proto#L29
@inline primitive_type(::Type{Bool}) = 1

@inline primitive_type(::Type{Int8}) = 2
@inline primitive_type(::Type{UInt8}) = 6

@inline primitive_type(::Type{Int16}) = 3
@inline primitive_type(::Type{UInt16}) = 7

@inline primitive_type(::Type{Int32}) = 4
@inline primitive_type(::Type{UInt32}) = 8

@inline primitive_type(::Type{Int64}) = 5
@inline primitive_type(::Type{UInt64}) = 9

@inline primitive_type(::Type{Float16}) = 10
@inline primitive_type(::Type{Float32}) = 11

@inline primitive_type(::Type{Reactant.F8E5M2}) = 19
@inline primitive_type(::Type{Reactant.F8E4M3FN}) = 20
@inline primitive_type(::Type{Reactant.F8E4M3B11FNUZ}) = 23
@inline primitive_type(::Type{Reactant.F8E5M2FNUZ}) = 24
@inline primitive_type(::Type{Reactant.F8E4M3FNUZ}) = 25

@static if isdefined(Core, :BFloat16)
    @inline primitive_type(::Type{Core.BFloat16}) = 16
end

@inline primitive_type(::Type{Float64}) = 12

@inline primitive_type(::Type{Complex{Float32}}) = 15
@inline primitive_type(::Type{Complex{Float64}}) = 18

function unsafe_string_and_free(str::Cstring, args...)
    str_jl = unsafe_string(str, args...)
    @ccall free(str::Cstring)::Cvoid
    return str_jl
end

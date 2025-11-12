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
primitive_types_list = [
    (1, Bool),
    (2, Int8),
    (6, UInt8),
    (3, Int16),
    (7, UInt16),
    (4, Int32),
    (8, UInt32),
    (5, Int64),
    (9, UInt64),
    (10, Float16),
    (11, Float32),
    (19, Reactant.F8E5M2),
    (20, Reactant.F8E4M3FN),
    (23, Reactant.F8E4M3B11FNUZ),
    (24, Reactant.F8E5M2FNUZ),
    (25, Reactant.F8E4M3FNUZ),
    (12, Float64),
    (15, Complex{Float32}),
    (18, Complex{Float64}),
]

@static if isdefined(Core, :BFloat16)
    push!(primitive_types_list, (16, Core.BFloat16))
end

for (int_val, jl_type) in primitive_types_list
    @eval begin
        @inline primitive_type(::Type{$(jl_type)}) = $(int_val)
        @inline julia_type(::Val{$(int_val)}) = $(jl_type)
    end
end

@inline julia_type(@nospecialize(x::Integer)) = julia_type(Val(Int64(x)))

function unsafe_string_and_free(str::Cstring, args...)
    str_jl = unsafe_string(str, args...)
    @ccall free(str::Cstring)::Cvoid
    return str_jl
end

mutable struct HloComputation
    ptr::Ptr{Cvoid}

    function HloComputation(ptr::Ptr{Cvoid})
        @assert ptr != C_NULL
        return new(ptr)
    end
end

function free_hlo_computation(hlo_computation)
    @ccall MLIR.API.mlir_c.freeHloComputation(hlo_computation.ptr::Ptr{Cvoid})::Cvoid
end

function Base.getproperty(hlo_computation::HloComputation, sym::Symbol)
    if sym === :instructions
        return convert(Vector{HloInstruction}, hlo_computation)
    end
    return getfield(hlo_computation, sym)
end

function Base.show(io::IO, hlo_computation::HloComputation)
    GC.@preserve hlo_computation begin
        str = @ccall MLIR.API.mlir_c.hloComputationToString(
            hlo_computation.ptr::Ptr{Cvoid}, _iobuffer_to_hlo_print_options(io)::Int32
        )::Cstring
    end
    print(io, unsafe_string_and_free(str))
    return nothing
end

function Base.convert(::Type{Vector{HloInstruction}}, hlo_computation::HloComputation)
    num_instructions = @ccall MLIR.API.mlir_c.hloComputationInstructionCount(
        hlo_computation.ptr::Ptr{Cvoid}
    )::Int64
    hlo_instructions = Ref{NTuple{num_instructions,Ptr{Cvoid}}}()
    GC.@preserve hlo_computation hlo_instructions begin
        @ccall MLIR.API.mlir_c.hloComputationGetInstructionsPostOrder(
            hlo_computation.ptr::Ptr{Cvoid},
            num_instructions::Int64,
            hlo_instructions::Ptr{Ptr{Cvoid}},
        )::Cvoid
    end
    return [map(HloInstruction, hlo_instructions[])...]
end

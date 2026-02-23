mutable struct HloComputation
    ptr::Ptr{Cvoid}

    function HloComputation(ptr::Ptr{Cvoid})
        @assert ptr != C_NULL
        return new(ptr)
    end
end

function free_hlo_computation(hlo_computation)
    return MLIR.API.freeHloComputation(hlo_computation.ptr)
end

function Base.getproperty(hlo_computation::HloComputation, sym::Symbol)
    if sym === :instructions
        return convert(Vector{HloInstruction}, hlo_computation)
    end
    return getfield(hlo_computation, sym)
end

function Base.show(io::IO, hlo_computation::HloComputation)
    print(
        io,
        unsafe_string_and_free(
            MLIR.API.hloComputationToString(
                hlo_computation.ptr, _iobuffer_to_hlo_print_options(io)
            ),
        ),
    )
    return nothing
end

function Base.convert(::Type{Vector{HloInstruction}}, hlo_computation::HloComputation)
    num_instructions = MLIR.API.hloComputationInstructionCount(hlo_computation.ptr)
    hlo_instructions = Ref{NTuple{num_instructions,Ptr{Cvoid}}}()
    MLIR.API.hloComputationGetInstructionsPostOrder(
        hlo_computation.ptr, num_instructions, hlo_instructions
    )
    return [map(HloInstruction, hlo_instructions[])...]
end

mutable struct HloInstruction
    ptr::Ptr{Cvoid}

    function HloInstruction(ptr::Ptr{Cvoid})
        @assert ptr != C_NULL
        return new(ptr)
    end
end

function free_hlo_instruction(hlo_instruction)
    @ccall MLIR.API.mlir_c.freeHloInstruction(hlo_instruction.ptr::Ptr{Cvoid})::Cvoid
end

function Base.show(io::IO, hlo_instruction::HloInstruction)
    GC.@preserve hlo_instruction begin
        str = @ccall MLIR.API.mlir_c.hloInstructionToString(
            hlo_instruction.ptr::Ptr{Cvoid}, _iobuffer_to_hlo_print_options(io)::Int32
        )::Cstring
    end
    print(io, unsafe_string_and_free(str))
    return nothing
end

function Base.getproperty(hlo_instruction::HloInstruction, sym::Symbol)
    if sym === :opcode
        return HloOpcode(
            @ccall MLIR.API.mlir_c.hloInstructionGetOpcode(
                hlo_instruction.ptr::Ptr{Cvoid}
            )::UInt8
        )
    end
    if sym === :to_apply
        @assert has_to_apply(hlo_instruction)
        return HloComputation(
            @ccall MLIR.API.mlir_c.hloInstructionGetToApply(
                hlo_instruction.ptr::Ptr{Cvoid}
            )::Ptr{Cvoid}
        )
    end
    if sym in (:fusion_kind, :fused_instructions_computation)
        @assert is_fusion_instruction(hlo_instruction)
        if sym === :fusion_kind
            return HloFusionKind(
                @ccall MLIR.API.mlir_c.hloInstructionGetFusionKind(
                    hlo_instruction.ptr::Ptr{Cvoid}
                )::UInt8
            )
        else
            return HloComputation(
                @ccall MLIR.API.mlir_c.hloInstructionFusedInstructionsComputation(
                    hlo_instruction.ptr::Ptr{Cvoid}
                )::Ptr{Cvoid}
            )
        end
    end
    return getfield(hlo_instruction, sym)
end

function has_to_apply(hlo_instruction::HloInstruction)
    has_to_apply = @ccall MLIR.API.mlir_c.hloInstructionHasToApply(
        hlo_instruction.ptr::Ptr{Cvoid}
    )::UInt8
    return has_to_apply == 1
end

function is_fusion_instruction(hlo_instruction::HloInstruction)
    is_fusion = @ccall MLIR.API.mlir_c.hloInstructionIsFusion(
        hlo_instruction.ptr::Ptr{Cvoid}
    )::UInt8
    return is_fusion == 1
end

struct HloOpcode
    opcode::UInt8
end

function Base.show(io::IO, hlo_opcode::HloOpcode)
    print(
        io,
        unsafe_string_and_free(
            @ccall MLIR.API.mlir_c.hloOpcodeToString(hlo_opcode.opcode::UInt8)::Cstring
        ),
    )
    return nothing
end

struct HloFusionKind
    kind::UInt8
end

function Base.show(io::IO, fusion_kind::HloFusionKind)
    print(
        io,
        unsafe_string_and_free(
            @ccall MLIR.API.mlir_c.hloFusionKindToString(fusion_kind.kind::UInt8)::Cstring
        ),
    )
    return nothing
end

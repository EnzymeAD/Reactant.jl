mutable struct HloInstruction
    ptr::Ptr{Cvoid}

    function HloInstruction(ptr::Ptr{Cvoid})
        @assert ptr != C_NULL
        return new(ptr)
    end
end

function free_hlo_instruction(hlo_instruction)
    return MLIR.API.freeHloInstruction(hlo_instruction.ptr)
end

function Base.show(io::IO, hlo_instruction::HloInstruction)
    str = MLIR.API.hloInstructionToString(
        hlo_instruction.ptr, _iobuffer_to_hlo_print_options(io)
    )
    print(io, unsafe_string_and_free(str))
    return nothing
end

function Base.getproperty(hlo_instruction::HloInstruction, sym::Symbol)
    if sym === :opcode
        return HloOpcode(MLIR.API.hloInstructionGetOpcode(hlo_instruction.ptr))
    end
    if sym === :to_apply
        @assert has_to_apply(hlo_instruction)
        return HloComputation(MLIR.API.hloInstructionGetToApply(hlo_instruction.ptr))
    end
    if sym in (:fusion_kind, :fused_instructions_computation)
        @assert is_fusion_instruction(hlo_instruction)
        if sym === :fusion_kind
            return HloFusionKind(MLIR.API.hloInstructionGetFusionKind(hlo_instruction.ptr))
        else
            return HloComputation(
                MLIR.API.hloInstructionFusedInstructionsComputation(hlo_instruction.ptr)
            )
        end
    end
    return getfield(hlo_instruction, sym)
end

function has_to_apply(hlo_instruction::HloInstruction)
    has_to_apply = MLIR.API.hloInstructionHasToApply(hlo_instruction.ptr)
    return has_to_apply == 1
end

function is_fusion_instruction(hlo_instruction::HloInstruction)
    is_fusion = MLIR.API.hloInstructionIsFusion(hlo_instruction.ptr)
    return is_fusion == 1
end

struct HloOpcode
    opcode::UInt8
end

function Base.show(io::IO, hlo_opcode::HloOpcode)
    print(io, unsafe_string_and_free(MLIR.API.hloOpcodeToString(hlo_opcode.opcode)))
    return nothing
end

struct HloFusionKind
    kind::UInt8
end

function Base.show(io::IO, fusion_kind::HloFusionKind)
    print(io, unsafe_string_and_free(MLIR.API.hloFusionKindToString(fusion_kind.kind)))
    return nothing
end

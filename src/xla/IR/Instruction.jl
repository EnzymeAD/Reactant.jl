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
    GC.@preserve hlo_instruction begin
        str = MLIR.API.hloInstructionToString(
            hlo_instruction.ptr, _iobuffer_to_hlo_print_options(io)
        )
    end
    print(io, unsafe_string_and_free(str))
    return nothing
end

function Base.getproperty(hlo_instruction::HloInstruction, sym::Symbol)
    if sym === :opcode
        GC.@preserve hlo_instruction begin
            ptr = MLIR.API.hloInstructionGetOpcode(hlo_instruction.ptr)
        end
        return HloOpcode(ptr)
    end
    if sym === :to_apply
        @assert has_to_apply(hlo_instruction)
        GC.@preserve hlo_instruction begin
            ptr = MLIR.API.hloInstructionGetToApply(hlo_instruction.ptr)
        end
        return HloComputation(ptr)
    end
    if sym in (:fusion_kind, :fused_instructions_computation)
        @assert is_fusion_instruction(hlo_instruction)
        if sym === :fusion_kind
            GC.@preserve hlo_instruction begin
                ptr = MLIR.API.hloInstructionGetFusionKind(hlo_instruction.ptr)
            end
            return HloFusionKind(ptr)
        else
            GC.@preserve hlo_instruction begin
                ptr = MLIR.API.hloInstructionFusedInstructionsComputation(
                    hlo_instruction.ptr
                )
            end
            return HloComputation(ptr)
        end
    end
    return getfield(hlo_instruction, sym)
end

function has_to_apply(hlo_instruction::HloInstruction)
    GC.@preserve hlo_instruction begin
        has_to_apply = MLIR.API.hloInstructionHasToApply(hlo_instruction.ptr)
    end
    return has_to_apply == 1
end

function is_fusion_instruction(hlo_instruction::HloInstruction)
    GC.@preserve hlo_instruction begin
        is_fusion = MLIR.API.hloInstructionIsFusion(hlo_instruction.ptr)
    end
    return is_fusion == 1
end

struct HloOpcode
    opcode::UInt8
end

function Base.show(io::IO, hlo_opcode::HloOpcode)
    GC.@preserve hlo_opcode begin
        str = MLIR.API.hloOpcodeToString(hlo_opcode.opcode)
    end
    print(io, unsafe_string_and_free(str))
    return nothing
end

struct HloFusionKind
    kind::UInt8
end

function Base.show(io::IO, fusion_kind::HloFusionKind)
    GC.@preserve fusion_kind begin
        str = MLIR.API.hloFusionKindToString(fusion_kind.kind)
    end
    print(io, unsafe_string_and_free(str))
    return nothing
end

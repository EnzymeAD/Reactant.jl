program_id(axis::Int) = Ops.triton_get_program_id(axis)

get_num_programs(axis::Int) = Ops.triton_get_num_programs(axis)

# TODO: for TracedRNumber
function load(ptrs::TracedRArray{TTPtr{T},N}; mask=nothing) where {T,N}
    return TracedRArray{T,N}(
        (),
        MLIR.IR.result(
            MLIR.Dialects.tt.load(
                ptrs.mlir_data, mask === nothing ? nothing : mask.mlir_data
            ),
        ),
        size(ptrs),
    )
end

function store!(ptrs, values; mask=nothing)
    MLIR.Dialects.tt.store(
        ptrs.mlir_data, values.mlir_data, mask === nothing ? nothing : mask.mlir_data
    )
    return nothing
end

# currently only supported for CUDA and ROCM
struct GPUPerformanceModel
    ptr::Ptr{Cvoid}

    function GPUPerformanceModel(ptr::Ptr{Cvoid})
        @assert ptr != C_NULL
        return new(ptr)
    end
end

function GPUPerformanceModel(
    mlir_context::MLIR.IR.Context, device_description::StreamExecutorDeviceDescription
)
    return GPUPerformanceModel(
        @ccall MLIR.API.mlir_c.CreateGPUPerformanceModel(
            mlir_context::MLIR.API.MlirContext, device_description.ptr::Ptr{Cvoid}
        )::Ptr{Cvoid}
    )
end

# Runs the analysis on the given HLO module.
function (gpu_performance_model::GPUPerformanceModel)(hlo_module::HloModule)
    GC.@preserve hlo_module begin
        @ccall MLIR.API.mlir_c.RunAnalysisOnHloModule(
            gpu_performance_model.ptr::Ptr{Cvoid}, hlo_module.ptr::Ptr{Cvoid}
        )::Cvoid
    end
    return nothing
end

function (gpu_performance_model::GPUPerformanceModel)(hlo_instruction::HloInstruction)
    return estimate_runtime_for_instruction(gpu_performance_model, hlo_instruction)
end

## To keep in sync with JLEstimateRunTimeData in ReactantExtra/API.cpp
struct EstimateRunTimeData
    flops::Int64
    bytes_read::Int64
    bytes_written::Int64
    read_time_ns::Int64
    write_time_ns::Int64
    compute_time_ns::Int64
    execution_time_ns::Int64
end

function estimate_runtime_for_instruction(
    performance_model::GPUPerformanceModel, hlo_instruction::HloInstruction
)
    data = Ref{EstimateRunTimeData}()
    GC.@preserve performance_model hlo_instruction data begin
        @ccall MLIR.API.mlir_c.EstimateRunTimeForInstruction(
            performance_model.ptr::Ptr{Cvoid},
            hlo_instruction.ptr::Ptr{Cvoid},
            data::Ptr{EstimateRunTimeData},
        )::Cvoid
    end
    return data[]
end

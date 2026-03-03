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
    GC.@preserve device_description begin
        return GPUPerformanceModel(
            MLIR.API.CreateGPUPerformanceModel(mlir_context, device_description.ptr)
        )
    end
end

# Runs the analysis on the given HLO module.
function (gpu_performance_model::GPUPerformanceModel)(hlo_module::HloModule)
    GC.@preserve gpu_performance_model hlo_module begin
        MLIR.API.RunAnalysisOnHloModule(gpu_performance_model.ptr, hlo_module.ptr)
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
    GC.@preserve performance_model hlo_instruction begin
        MLIR.API.EstimateRunTimeForInstruction(
            performance_model.ptr, hlo_instruction.ptr, data
        )
    end
    return data[]
end

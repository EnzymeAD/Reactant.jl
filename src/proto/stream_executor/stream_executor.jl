module stream_executor

include("../xla_autotuning/xla_autotuning.jl")

include("cuda_compute_capability_pb.jl")
include("device_description_pb.jl")

end # module stream_executor

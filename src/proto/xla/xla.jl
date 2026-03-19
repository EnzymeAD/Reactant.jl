module xla

include("../stream_executor/stream_executor.jl")
include("../google/google.jl")
import .xla as var"#xla"

include("metrics_pb.jl")
include("execute_options_pb.jl")
include("xla_data_pb.jl")
include("hlo_pb.jl")
include("autotuner/autotuner.jl")
include("xla_pb.jl")
include("compile_options_pb.jl")

end # module xla

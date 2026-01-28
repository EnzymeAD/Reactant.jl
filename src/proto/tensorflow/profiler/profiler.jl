module profiler

import ..google

include("kernel_stats_pb.jl")
include("power_metrics_pb.jl")
include("diagnostics_pb.jl")
include("xplane_pb.jl")
include("topology_pb.jl")
include("source_info_pb.jl")
include("task_pb.jl")
include("hardware_types_pb.jl")
include("memory_profile_pb.jl")
include("source_stats_pb.jl")
include("tf_function_pb.jl")
include("input_pipeline_pb.jl")
include("op_metrics_pb.jl")
include("trace_events_pb.jl")
include("overview_page_pb.jl")
include("steps_db_pb.jl")
include("op_stats_pb.jl")
include("op_profile/op_profile.jl")
include("roofline_model/roofline_model.jl")
include("hlo_stats/hlo_stats.jl")

end # module profiler

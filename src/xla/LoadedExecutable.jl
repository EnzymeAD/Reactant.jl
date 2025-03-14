abstract type AbstractLoadedExecutable end

function num_replicas end
function num_partitions end
function num_devices end
function get_hlo_modules end
function get_output_shardings end
function get_parameter_shardings end

function compile end
function execute end
function execute_sharded end

function cost_analysis(exec::AbstractLoadedExecutable)
    hlo_modules = get_hlo_modules(exec)
    analysis = cost_analysis.((client(exec),), hlo_modules)
    length(analysis) == 1 && return only(analysis)
    return analysis
end

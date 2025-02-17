abstract type AbstractLoadedExecutable end

function num_replicas end
function num_partitions end
function get_hlo_modules end
function get_output_shardings end
function get_parameter_shardings end

function compile end
function execute end
function execute_sharded end

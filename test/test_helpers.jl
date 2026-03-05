using ParallelTestRunner, Reactant, Test

function run_specific_test(test_name::String, parsed_args)
    if (
        isempty(parsed_args.positionals) ||
        any(Base.Fix1(startswith, test_name), parsed_args.positionals)
    )
        test_file = joinpath(@__DIR__, test_name * ".jl")
        ParallelTestRunner.runtest(:(include($(test_file))), test_name, :())
    end
end

const test_worker_processes = Tuple{Int,Any}[]
const procs_lock = ReentrantLock()

const NTPUs = first(Reactant.Accelerators.TPU.num_available_tpu_chips_and_device_id())

const available_tpus = if haskey(ENV, "TPU_VISIBLE_DEVICES")
    parse.(Int, split(ENV["TPU_VISIBLE_DEVICES"], ","))
else
    collect(0:(NTPUs - 1))
end

for tpu in available_tpus
    push!(test_worker_processes, (tpu, nothing))
end

function tpu_custom_worker_launcher(name)
    tpu_id, idx = Base.@lock procs_lock begin
        _idx = findfirst(test_worker_processes) do (tpu, proc)
            isnothing(proc) || (proc isa Base.Process && !process_running(proc))
        end

        if _idx !== nothing
            _tpu_id = test_worker_processes[_idx][1]
            test_worker_processes[_idx] = (_tpu_id, :launching)
        else
            _tpu_id = 0
        end

        _tpu_id, _idx
    end

    worker = addworker(;
        env=[
            "TPU_CHIPS_PER_PROCESS_BOUNDS" => "1,1,1",
            "TPU_PROCESS_BOUNDS" => "1,1,1",
            "TPU_VISIBLE_DEVICES" => string(tpu_id),
        ],
        color=get(stdout, :color, false),
    )
    Base.@lock procs_lock begin
        if idx !== nothing
            test_worker_processes[idx] = (tpu_id, worker.w.proc)
        else
            push!(test_worker_processes, (tpu_id, worker.w.proc))
        end
    end
    return worker
end

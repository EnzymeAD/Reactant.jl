#=
Shared arm plumbing + IR evidence helpers for the tab:prelowering ablation.

The two-arm definition is the one frozen for the heat rows on 2026-08-21
(see heat_inverse_RESULTS_2026-08-21.md):

  arm A ("No Pre-Opt"): flat expand path (DEBUG_PROBPROG_DISABLE_OPT = true)
                        + disable_licm_optimization_passes = true
  arm B ("Pre-Opt")   : flat expand path + full standard passes

SICM is OFF in both arms (DEBUG_PROBPROG_DISABLE_SICM = true, and the flat
expand path structurally never runs the sicm pass); the per-cell syevd count in
the IR report is the verification hook.

`count_in_out_loops` / `ir_report` are copied verbatim from heat_inverse.jl so
that every row of the table is classified by identical code.
=#
using JSON3
using Dates: now

# arm -> (disable_opt [flat path], disable_sicm, disable_licm)
const PRELOWER_ARM_KNOBS = Dict(
    "A" => (true, true, true),    # No Pre-Opt: flat expand, LICM off
    "B" => (true, true, false),   # Pre-Opt: flat expand, full std passes
)

# ── IR analysis: count ops, split by whether they sit inside a stablehlo.while
# body region. MLIR pretty-prints `stablehlo.while(...) attributes {...}` then
# `cond { ... } do { ... }` on following lines, so brace-depth tracking of the
# op line alone is wrong; instead track region open/close via the `cond {` /
# `} do {` / `}` markers and tag while-body regions.
function count_in_out_loops(hlo_str::AbstractString, needle::AbstractString)
    region_is_while = Bool[]   # stack: is each open region a while cond/do body?
    pending_while = false
    n_in = 0
    n_out = 0
    for raw in split(hlo_str, '\n')
        line = strip(raw)
        # Closing a region (line begins with `}`, possibly `} do {`).
        if startswith(line, "}") && !isempty(region_is_while)
            pop!(region_is_while)
        end
        # Count the needle, classified by current region nesting.
        if occursin(needle, line)
            c = count(needle, line)
            if any(region_is_while)
                n_in += c
            else
                n_out += c
            end
        end
        occursin("stablehlo.while", line) && (pending_while = true)
        # Opening a region: line ends with `{` (cond {, } do {, func.func ... {).
        if endswith(line, "{")
            is_while_body = (occursin("do {", line) || endswith(line, "cond {")) &&
                            pending_while
            push!(region_is_while, is_while_body)
            occursin("do {", line) && (pending_while = false)
        end
    end
    return (in=n_in, out=n_out, total=n_in + n_out)
end

const PRELOWER_IR_NEEDLES = ["stablehlo.cholesky", "triangular_solve", "syevd",
                             "dot_general", "stablehlo.while", "custom_call"]

function ir_report(hlo_str::AbstractString)
    counts = Dict{String,Any}()
    println("  --- IR op counts (in-loop / out-of-loop / total) ---")
    for nd in PRELOWER_IR_NEEDLES
        r = count_in_out_loops(hlo_str, nd)
        counts[nd] = Dict("in" => r.in, "out" => r.out, "total" => r.total)
        println("    $(rpad(nd, 22)) in=$(r.in)  out=$(r.out)  total=$(r.total)")
    end
    counts["lines"] = count(==('\n'), hlo_str)
    println("    $(rpad("lines", 22)) $(counts["lines"])")
    return counts
end

"""Write one measured cell (timings + verification samples + IR counts)."""
function write_cell(path, payload)
    mkpath(dirname(path))
    open(io -> JSON3.write(io, payload), path, "w")
    println("  wrote cell json: $path")
end

"""Save the final HLO for offline shape-classified structural inspection."""
function save_hlo(path, hlo_str)
    mkpath(dirname(path))
    open(io -> print(io, hlo_str), path, "w")
    println("  saved HLO: $path")
end

# ── Frozen-adapted-state protocol (2026-08-25) ───────────────────────────────
#
# The figure timings are measured in the adapted regime rather than a naive
# fixed-step one, while keeping the arms bit-identical:
#
#   1. WARMUP (once per model, NOT timed, run in the Pre-Opt arm): adaptation
#      ON for `num_warmup` iterations. The adapted step size, adapted inverse
#      mass matrix and post-warmup position are written to a state JSON.
#   2. MEASURED (both arms): adaptation OFF, initialised from that SAME state.
#      Identical step size, identical inverse mass, identical starting position
#      and matched seeds, so the two arms walk bit-identical trajectories and
#      the identical-accepts gate is still the arms-equivalence check.
#
# No compiler change is needed: `mcmc_logpdf` already takes `initial_position`
# positionally and `step_size` / `inverse_mass_matrix` as keywords, and returns
# the post-run `MCMCState` carrying all three.

"""Convert a JSON nested-row array into a Matrix{Float64}."""
_rows_to_matrix(rows) =
    permutedims(reduce(hcat, [collect(Float64, r) for r in rows]))

"""Pull the adapted (step size, inverse mass, position) out of a returned
MCMCState as plain Julia values."""
function extract_adapted_state(st)
    step = Array(st.step_size)[]                    # 0-d array -> scalar
    imm = Array(st.inverse_mass_matrix)             # rank 1 (diag) or 2 (dense)
    pos = Array(st.position)                        # 1 x P
    return (; step, imm, pos)
end

"""Write the adapted sampler state plus its provenance."""
function save_adapted_state(path, model::AbstractString, adapted; num_warmup::Int,
                            produced_by_arm::AbstractString, warmup_seed,
                            initial_step_size, extra=Dict{String,Any}())
    imm = adapted.imm
    obj = Dict{String,Any}(
        "model" => model,
        "protocol" => "frozen-adapted-state: warmup with adaptation ON in the " *
                      "Pre-Opt arm, then BOTH arms measured with adaptation OFF " *
                      "from this state",
        "produced_by_arm" => produced_by_arm,
        "num_warmup" => num_warmup,
        "warmup_seed" => collect(UInt64, warmup_seed),
        "initial_step_size" => initial_step_size,
        "adapted_step_size" => adapted.step,
        "inverse_mass_rank" => ndims(imm),
        "adapted_inverse_mass" => ndims(imm) == 1 ? collect(Float64, imm) :
                                  [collect(Float64, imm[i, :]) for i in axes(imm, 1)],
        "post_warmup_position" => collect(Float64, vec(adapted.pos)),
        "created" => string(now()),
    )
    merge!(obj, extra)
    mkpath(dirname(path))
    open(io -> JSON3.write(io, obj), path, "w")
    println("  wrote adapted state: $path")
    println("    adapted step size = $(adapted.step)")
    println("    adapted inverse mass (rank $(ndims(imm))) = $(imm)")
    println("    post-warmup position = $(vec(adapted.pos))")
    return path
end

"""Read an adapted sampler state written by `save_adapted_state`."""
function load_adapted_state(path)
    isfile(path) || error("adapted state not found: $path " *
                          "(run the harness in mode=warmup first)")
    s = JSON3.read(read(path, String))
    imm = Int(s["inverse_mass_rank"]) == 1 ?
          collect(Float64, s["adapted_inverse_mass"]) :
          _rows_to_matrix(s["adapted_inverse_mass"])
    pos = reshape(collect(Float64, s["post_warmup_position"]), 1, :)
    return (; step=Float64(s["adapted_step_size"]), imm, pos, meta=s)
end

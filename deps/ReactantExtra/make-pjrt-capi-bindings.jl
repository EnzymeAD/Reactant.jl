using Clang.Generators
using Clang: CLLinkageSpec, children

const REACTANT_ROOT = dirname(dirname(@__DIR__))
const DEFAULT_OUTPUT_DIR = joinpath(REACTANT_ROOT, "src", "xla", "CAPI")

const bazel_cmd = if !isnothing(Sys.which("bazelisk"))
    "bazelisk"
elseif !isnothing(Sys.which("bazel"))
    "bazel"
else
    error("Could not find `bazel` or `bazelisk` in PATH!")
end

# get header files
bazel_targets = join(
    [
        "@xla//xla/pjrt/c:pjrt_c_api.h",
        "@xla//xla/pjrt/c:pjrt_c_api_callback_extension.h",
        "@xla//xla/pjrt/c:pjrt_c_api_collectives_extension.h",
        "@xla//xla/pjrt/c:pjrt_c_api_custom_partitioner_extension.h",
        "@xla//xla/pjrt/c:pjrt_c_api_abi_version_extension.h",
        "@xla//xla/pjrt/c:pjrt_c_api_device_event.h",
        "@xla//xla/pjrt/c:pjrt_c_api_ffi_extension.h",
        "@xla//xla/pjrt/c:pjrt_c_api_gpu_extension.h",
        "@xla//xla/pjrt/c:pjrt_c_api_layouts_extension.h",
        "@xla//xla/pjrt/c:pjrt_c_api_megascale_extension.h",
        "@xla//xla/pjrt/c:pjrt_c_api_memory_descriptions_extension.h",
        "@xla//xla/pjrt/c:pjrt_c_api_multi_slice_extension.h",
        "@xla//xla/pjrt/c:pjrt_c_api_phase_compile_extension.h",
        "@xla//xla/pjrt/c:pjrt_c_api_profiler_extension.h",
        "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_extension.h",
        "@xla//xla/pjrt/c:pjrt_c_api_shardings_extension.h",
        "@xla//xla/pjrt/c:pjrt_c_api_stream_extension.h",
        "@xla//xla/pjrt/c:pjrt_c_api_tpu_executable_extension.h",
        "@xla//xla/pjrt/c:pjrt_c_api_tpu_topology_extension.h",
        "@xla//xla/pjrt/c:pjrt_c_api_xla_transform_extension.h",
    ],
    ' ',
)

headers =
    let cmd = Cmd(
            `$(bazel_cmd) query --output=location --noshow_progress --ui_event_filters=-info "set($(bazel_targets))"`;
            dir=@__DIR__
        )
        map(eachline(IOBuffer(String(read(cmd))))) do line
            String(split(line, ':'; limit=2)[1])
        end
    end

# anonymous `enum { X_STRUCT_SIZE = ... }` from `PJRT_DEFINE_STRUCT_TRAITS(X,_)`
# are emitted as `@cenum __JL_Ctag_N::UInt32 begin X_STRUCT_SIZE = v end`
# rewrite them into `csizeof(::Type{T}) = Csize_t(v)`
struct PjrtStructTraitsExprNodeType <: Generators.AbstractExprNodeType end

function Generators.pretty_print(
    io, node::ExprNode{PjrtStructTraitsExprNodeType}, options::Dict
)
    foreach(ex -> println(io, ex), node.exprs)
    println(io)
    return nothing
end

function rewrite_pjrt_define_struct_traits(node)
    node.type isa Generators.EnumAnonymous || return nothing
    length(node.exprs) == 2 || return nothing

    member = node.exprs[2]
    Meta.isexpr(member, :(=)) || return nothing

    name, value = member.args
    typname = Symbol(chopsuffix(string(name), "_STRUCT_SIZE"))
    typname != name || return nothing

    ex = Expr(:(=), :(csizeof(::Type{$typname})), :(Csize_t($value)))
    return ExprNode(
        node.id,
        PjrtStructTraitsExprNodeType(),
        node.cursor,
        Expr[ex],
        node.premature_exprs,
        node.adj,
    )
end

function rewrite!(dag::ExprDAG)
    for (i, node) in enumerate(dag.nodes)
        new_node = rewrite_pjrt_define_struct_traits(node)
        isnothing(new_node) && continue
        dag.nodes[i] = new_node
    end
    return dag
end

function (@main)(argv)
    output = joinpath(DEFAULT_OUTPUT_DIR, "PJRT.jl")

    bad = filter(h -> !endswith(h, ".h"), headers)
    isempty(bad) || error("expected only header files as inputs, got: $(join(bad, ", "))")

    options = load_options(joinpath(@__DIR__, "wrap-pjrt-capi.toml"))
    options["general"]["output_file_path"] = output

    args = Generators.get_default_args()
    CORE_HEADER = joinpath("xla", "pjrt", "c", "pjrt_c_api.h")
    include_path = let idx = findfirst(h -> endswith(h, CORE_HEADER), headers)
        isnothing(idx) && error("`$(CORE_HEADER)` must be one of the input headers")
        chopsuffix(headers[idx], CORE_HEADER)
    end
    append!(args, ["-I", include_path, "-x", "c"])

    ctx = create_context(headers, args, options)

    # build without printing so we can do custom rewriting
    build!(ctx, BUILDSTAGE_NO_PRINTING)
    rewrite!(ctx.dag)
    build!(ctx, BUILDSTAGE_PRINTING_ONLY)
    return nothing
end

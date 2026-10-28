using Clang.Generators
using Clang: CLLinkageSpec, children

const OUTPUT_DIR = joinpath(dirname(dirname(@__DIR__)), "src", "xla", "CAPI")

const bazel_cmd = if !isnothing(Sys.which("bazelisk"))
    "bazelisk"
elseif !isnothing(Sys.which("bazel"))
    "bazel"
else
    error("Could not find `bazel` or `bazelisk` in PATH!")
end

targets = [
    (
        "PJRT.jl",
        "xla/pjrt/c/pjrt_c_api.h",
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
    ),
    (
        "FFI.jl",
        "xla/ffi/api/c_api.h",
        [
            "@xla//xla/ffi/api:c_api.h",
            "@xla//xla/ffi/api:collectives_c_api.h",
            "@xla//xla/ffi/api:record_c_api.h",
        ],
    ),
    (
        "TPU.jl",
        "xla/tpu/c_api_decl.h",
        [
            "@xla//xla/tpu:c_api_decl.h",
            "@xla//xla/tpu:libtftpu.h",
            "@xla//xla/tpu:tpu_executor_c_api.h",
            "@xla//xla/tpu:tpu_ops_c_api.h",
            "@xla//xla/tpu:tpu_profiler_c_api.h",
        ],
    ),
]

function bazel_query_header_paths(targets)
    targets = join(targets, ' ')
    cmd = Cmd(
        `$(bazel_cmd) query --output=location --noshow_progress --ui_event_filters=-info "set($(targets))"`;
        dir=@__DIR__
    )
    map(eachline(IOBuffer(String(read(cmd))))) do line
        String(split(line, ':'; limit=2)[1])
    end
end

# anonymous `enum { X_STRUCT_SIZE = ... }` from `PJRT/XLA_FFI_DEFINE_STRUCT_TRAITS(X,_)`
# are emitted as `@cenum __JL_Ctag_N::UInt32 begin X_STRUCT_SIZE = v end`
# rewrite them into `csizeof(::Type{T}) = Csize_t(v)`
struct StructTraitsExprNodeType <: Generators.AbstractExprNodeType end

function Generators.pretty_print(
    io, node::ExprNode{StructTraitsExprNodeType}, options::Dict
)
    foreach(ex -> println(io, ex), node.exprs)
    println(io)
    return nothing
end

# fix `@cenum` disaster of `PJRT_DEFINE_STRUCT_TRAITS` macro
function rewrite_define_struct_traits(node)
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
        StructTraitsExprNodeType(),
        node.cursor,
        Expr[ex],
        node.premature_exprs,
        node.adj,
    )
end

# automatically set `struct_size` field (first) of `*_Args` structs
function rewrite_args_struct_constructor(node)
    node.type isa Generators.StructDefinition || return nothing
    ex = first(node.exprs)
    @assert Meta.isexpr(ex, :struct)
    Base.remove_linenums!(ex)

    ismut, name, body = ex.args
    endswith(string(name), "_Args") || return nothing

    fieldnames = map(body.args) do arg
        if Meta.isexpr(arg, :(::))
            arg.args[1]::Symbol
        elseif arg isa Symbol
            arg
        else
            error("Unknown field")
        end
    end
    fieldnames[1] == :struct_size || return nothing
    deleteat!(fieldnames, 1)

    @assert Meta.isexpr(body, :block)
    push!(
        body.args,
        Expr(
            :function,
            Expr(:call, name, fieldnames...),
            Expr(
                :block,
                Expr(
                    :return, Expr(:call, :new, Expr(:call, :csizeof, name), fieldnames...)
                ),
            ),
        ),
    )
    return node
end

function rewrite!(dag::ExprDAG)
    for (i, node) in enumerate(dag.nodes)
        # fix `@cenum` disaster of `PJRT_DEFINE_STRUCT_TRAITS` macro
        new_node = rewrite_define_struct_traits(node)
        if !isnothing(new_node)
            dag.nodes[i] = new_node
        end

        # automatically set `struct_size` field (first) of `*_Args` structs
        new_node = rewrite_args_struct_constructor(node)
        if !isnothing(new_node)
            dag.nodes[i] = new_node
        end
    end
    return dag
end

# matches any struct name ending in `suffix` for `field_access_method_list`
struct SuffixMatch
    suffix::String
end
Base.in(name::AbstractString, m::SuffixMatch) = endswith(name, m.suffix)

function (@main)(argv)
    for (output, core_header, target_set) in targets
        output = joinpath(OUTPUT_DIR, output)
        headers = bazel_query_header_paths(target_set)

        bad = filter(h -> !endswith(h, ".h"), headers)
        isempty(bad) ||
            error("expected only header files as inputs, got: $(join(bad, ", "))")

        options = load_options(joinpath(@__DIR__, "wrap-xla.toml"))
        options["general"]["output_file_path"] = output
        options["codegen"]["field_access_method_list"] = SuffixMatch("_Args")

        args = Generators.get_default_args()
        include_path = let idx = findfirst(h -> endswith(h, core_header), headers)
            isnothing(idx) && error("`$(core_header)` must be one of the input headers")
            chopsuffix(headers[idx], core_header)
        end
        append!(args, ["-I", include_path, "-x", "c"])

        ctx = create_context(headers, args, options)

        # build without printing so we can do custom rewriting
        build!(ctx, BUILDSTAGE_NO_PRINTING)
        rewrite!(ctx.dag)
        build!(ctx, BUILDSTAGE_PRINTING_ONLY)
    end
    return nothing
end

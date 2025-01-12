import BinaryBuilderBase:
    PkgSpec, Prefix, temp_prefix, setup_dependencies, cleanup_dependencies, destdir
using Clang.Generators

options = load_options(joinpath(@__DIR__, "wrap.toml"))

function rewrite!(dag::ExprDAG) end

@add_def off_t
@add_def MlirTypesCallback

let options = deepcopy(options)
    options["general"]["output_file_path"] = ARGS[end]

    include_dir = joinpath(splitpath(ARGS[1])[1:(end - 4)]...)
    args = Generators.get_default_args()
    ll_include_dir = joinpath(splitpath(ARGS[2])[1:(end - 2)]...)

    genarg = first(eachsplit(ARGS[3], " "))

    gen_include_dir = joinpath(splitpath(genarg)[1:(end - 3)]...)

    hlo_include_dir = joinpath(splitpath(ARGS[end - 1])[1:(end - 1)]...)

    append!(
        args,
        [
            "-I",
            include_dir,
            "-I",
            ll_include_dir,
            "-I",
            gen_include_dir,
            "-I",
            hlo_include_dir,
            "-x",
            "c++",
        ],
    )

    headers = [
        detect_headers(include_dir, args, Dict(), endswith("Python/Interop.h"))...,
        detect_headers(hlo_include_dir, args, Dict())...,
    ]

    ctx = create_context(headers, args, options)

    # build without printing so we can do custom rewriting
    build!(ctx, BUILDSTAGE_NO_PRINTING)

    rewrite!(ctx.dag)

    # print
    build!(ctx, BUILDSTAGE_PRINTING_ONLY)
end

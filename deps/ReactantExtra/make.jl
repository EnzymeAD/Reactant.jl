using Pkg: Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

import BinaryBuilderBase:
    PkgSpec, Prefix, temp_prefix, setup_dependencies, cleanup_dependencies, destdir
using Clang.Generators
using Clang: CLLinkageSpec, children

# Add support for extern "C" blocks - Clang.jl doesn't handle these by default
# We need to recurse into the children of the linkage spec to process the declarations inside
function Generators.collect_top_level_nodes!(
    nodes::Vector{ExprNode}, cursor::CLLinkageSpec, options
)
    for child in children(cursor)
        Generators.collect_top_level_nodes!(nodes, child, options)
    end
    return nodes
end

options = load_options(joinpath(@__DIR__, "wrap.toml"))

function rewrite!(::ExprDAG) end

@add_def off_t
@add_def MlirTypesCallback

let options = deepcopy(options)
    options["general"]["output_file_path"] = ARGS[end]

    args = Generators.get_default_args()

    # Robustly find include directories from ARGS
    include_dirs = Set{String}()
    push!(include_dirs, @__DIR__)
    push!(include_dirs, ".")

    for arg in ARGS
        # Handle multiple files in one arg (from locations)
        for path in split(arg, " ")
            isempty(path) && continue
            isfile(path) || continue

            # Add the directory of the file
            d = dirname(path)
            push!(include_dirs, d)

            # If it's in an 'include' directory, add the 'include' directory itself
            # This helps find things like 'mlir/...' when we have '.../include/mlir/...'
            parts = splitpath(path)
            idx = findlast(x -> x == "include", parts)
            if idx !== nothing
                push!(include_dirs, joinpath(parts[1:idx]...))
            end

            # Handle external repositories in Bazel (external/REPO/...)
            # This helps find things like 'xla/...' when we have 'external/xla/xla/...'
            if length(parts) >= 2 && parts[1] == "external"
                push!(include_dirs, joinpath(parts[1:2]...))
            end
        end
    end

    for d in include_dirs
        append!(args, ["-I", d])
    end

    append!(
        args,
        [
            "-DREACTANT_BINDINGS_GENERATION=1",
            "-x",
            "c++",
            "-DLLVM_ATTRIBUTE_C_DEPRECATED(decl, msg)=decl",
        ],
    )

    extract_api =
        get(options["general"], "extract_api", false) ||
        any(x -> x == "--extract-api", ARGS)

    extracted_api_path = joinpath(pwd(), "API_extracted_inner.h")
    if extract_api
        open(extracted_api_path, "w") do io
            println(
                io,
                """
    #include <stddef.h>
    #include <stdint.h>
    #ifndef __cplusplus
    #include <stdbool.h>
    #endif
    """,
            )

            # Extract structs and functions from API.cpp and xla_ffi.cpp
            structs = Dict{String,String}() # name -> full definition
            ptrs = Set{String}()
            signatures = String[]

            for name in ["API.cpp", "xla_ffi.cpp"]
                file = joinpath(@__DIR__, name)
                if !isfile(file)
                    continue
                end
                content = read(file, String)

                # Extract structs
                lines = split(content, "\n")
                for (i, line) in enumerate(lines)
                    m = match(r"struct\s+([A-Za-z0-9_]+)\s*\{", line)
                    if m !== nothing
                        struct_name = m.captures[1]
                        # Check if previous line contains 'template'
                        if i > 1 && contains(lines[i - 1], "template")
                            continue
                        end

                        # Find completion of struct
                        s = line
                        j = i + 1
                        open_braces =
                            count(x -> x == '{', line) - count(x -> x == '}', line)
                        while open_braces > 0 && j <= length(lines)
                            s *= "\n" * lines[j]
                            open_braces +=
                                count(x -> x == '{', lines[j]) -
                                count(x -> x == '}', lines[j])
                            j += 1
                        end
                        if open_braces == 0
                            # Final check for template inside
                            contains(s, "template") && continue
                            structs[struct_name] = s
                        end
                    end
                end

                # Extract functions
                for m in
                    eachmatch(r"REACTANT_ABI\s+(.*?)(?:\{|=(\s*nullptr\s*)?;|;)"s, content)
                    sig = m.match
                    sig = replace(sig, r"\s*\{\s*$" => ";")
                    sig = replace(sig, r"\s*=\s*nullptr\s*;\s*$" => ";")
                    sig = replace(sig, "REACTANT_ABI" => "MLIR_CAPI_EXPORTED")
                    # Remove namespaces for C mode compatibility
                    # E.g. tsl::ProfilerSession * -> ProfilerSession *
                    sig = replace(sig, r"\b(?:\w+::)+" => "")
                    # error on encountering C++ specific things that don't map nicely to C
                    if (
                        contains(sig, "std::string") ||
                        contains(sig, "std::string_view") ||
                        contains(sig, "std::vector<int64_t>")
                    )
                        error("Found C++ specific thing in signature: $sig")
                    end
                    push!(signatures, sig)

                    for word in eachmatch(r"\b([A-Z]\w+)\s*\*", sig) # Match words starting with Capital letter
                        push!(ptrs, word.captures[1])
                    end
                    for word in eachmatch(r"\b\w+Ptr\b", sig)
                        push!(ptrs, word.match)
                    end
                end
            end

            for (name, s) in structs
                println(io, s)
                println(io, "")
            end

            for ptr in ptrs
                if !haskey(structs, ptr) &&
                    !in(
                    ptr,
                    [
                        "MlirContext",
                        "MlirBlock",
                        "MlirOperation",
                        "MlirType",
                        "MlirAttribute",
                        "MlirStringRef",
                        "MlirLocation",
                        "MlirModule",
                        "MlirDialectRegistry",
                        "MlirValue",
                    ],
                )
                    println(io, "typedef void *", ptr, ";")
                end
            end
            println(io, "")
            for sig in signatures
                println(io, sig)
            end
        end
    end

    headers = String[]
    for arg in ARGS
        # Handle multiple files in one arg
        for path in split(arg, " ")
            isempty(path) && continue
            isfile(path) || continue
            # Only wrap actual header files passed as entry points
            if endswith(path, ".h")
                # Avoid wrapping core LLVM or internal config headers that break Clang.jl
                if contains(path, "llvm-c/") || contains(path, "Config/")
                    continue
                end
                push!(headers, path)
            end
        end
    end
    if extract_api
        push!(headers, extracted_api_path)
    end

    ctx = create_context(headers, args, options)

    # build without printing so we can do custom rewriting
    build!(ctx, BUILDSTAGE_NO_PRINTING)

    rewrite!(ctx.dag)

    # print
    build!(ctx, BUILDSTAGE_PRINTING_ONLY)
end

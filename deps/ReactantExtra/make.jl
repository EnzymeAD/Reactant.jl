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

function rewrite!(dag::ExprDAG) end

@add_def off_t
@add_def MlirTypesCallback

let options = deepcopy(options)
    options["general"]["output_file_path"] = ARGS[end]

    extract_api =
        get(options["general"], "extract_api", false) ||
        any(x -> x == "--extract-api", ARGS)

    first_arg_is_extract_api = first(ARGS) == "--extract-api"

    start_idx = first_arg_is_extract_api ? 2 : 1
    include_dir = joinpath(splitpath(ARGS[start_idx])[1:(end - 4)]...)
    ll_include_dir = joinpath(splitpath(ARGS[start_idx + 1])[1:(end - 2)]...)
    genarg = first(eachsplit(ARGS[start_idx + 2], " "))
    gen_include_dir = joinpath(splitpath(genarg)[1:(end - 4)]...)
    gen_llvm_include_dir = joinpath(splitpath(ARGS[start_idx + 1])[1:(end - 2)]...)

    hlo_include_dir = joinpath(splitpath(ARGS[end - 7])[1:(end - 1)]...)
    sdy_include_dir = joinpath(splitpath(ARGS[end - 6])[1:(end - 1)]...)
    triton_include_dir = joinpath(splitpath(ARGS[end - 5])[1:(end - 1)]...)
    mosaic_tpu_include_dir = joinpath(splitpath(ARGS[end - 4])[1:(end - 1)]...)
    mosaic_gpu_include_dir = joinpath(splitpath(ARGS[end - 3])[1:(end - 1)]...)
    enzymexla_include_dir = joinpath(splitpath(ARGS[end - 2])[1:(end - 1)]...)
    enzymemlir_include_dir = joinpath(splitpath(ARGS[end - 1])[1:(end - 1)]...)

    args = Generators.get_default_args()

    # Robustly find include directories from ARGS
    include_dirs = Set{String}()
    for dir in [
        include_dir,
        ll_include_dir,
        gen_include_dir,
        gen_llvm_include_dir,
        hlo_include_dir,
        sdy_include_dir,
        triton_include_dir,
        mosaic_tpu_include_dir,
        mosaic_gpu_include_dir,
        enzymexla_include_dir,
        enzymemlir_include_dir,
    ]
        push!(include_dirs, dir)
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

    extracted_api_path = joinpath(pwd(), "API_extracted_inner.h")
    if extract_api
        open(extracted_api_path, "w") do io
            println(
                io,
                """
    #ifndef REACTANT_EXTRACTED_API_H
    #define REACTANT_EXTRACTED_API_H

    #include <stddef.h>
    #include <stdint.h>
    #ifndef __cplusplus
    #include <stdbool.h>
    #endif

    #include "mlir-c/IR.h"
    #include "mlir-c/Support.h"

    #ifndef MLIR_CAPI_EXPORTED
    #ifdef _WIN32
    #ifdef MLIR_CAPI_BUILDING_LIBRARY
    #define MLIR_CAPI_EXPORTED __declspec(dllexport)
    #else
    #define MLIR_CAPI_EXPORTED __declspec(dllimport)
    #endif
    #else
    #define MLIR_CAPI_EXPORTED __attribute__((visibility("default")))
    #endif
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

                    # Skip dump functions
                    # Match function name: optional return type, space, then name
                    m_name = match(r"REACTANT_ABI\s+(?:.*?\s+)?(\w+)\s*\(", sig)
                    if m_name !== nothing && startswith(m_name.captures[1], "dump_")
                        continue
                    end

                    sig = replace(sig, r"\s*\{\s*$" => ";")
                    sig = replace(sig, r"\s*=\s*nullptr\s*;\s*$" => ";")

                    # error on encountering C++ specific things that don't map nicely to C
                    if (
                        contains(sig, "std::string") ||
                        contains(sig, "std::string_view") ||
                        contains(sig, "std::vector") ||
                        contains(sig, "std::shared_ptr") ||
                        contains(sig, "std::optional") ||
                        contains(sig, "std::unique_ptr")
                    )
                        # error("Found C++ specific thing in signature: $sig")
                        @error("Found C++ specific thing in signature: $sig")
                        continue
                    end

                    sig = replace(sig, "REACTANT_ABI" => "MLIR_CAPI_EXPORTED")
                    # Remove namespaces for C mode compatibility
                    # E.g. tsl::ProfilerSession * -> ProfilerSession *
                    sig = replace(sig, r"\b(?:\w+::)+" => "")
                    # Fix double MLIR_CAPI_EXPORTED
                    sig = replace(
                        sig, "MLIR_CAPI_EXPORTED MLIR_CAPI_EXPORTED" => "MLIR_CAPI_EXPORTED"
                    )

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
                        "MlirTypeID",
                    ],
                )
                    println(io, "typedef void *", ptr, ";")
                end
            end
            println(io, "")
            for sig in signatures
                println(io, sig)
            end

            println(io, "#endif")
        end
    end

    headers = [
        detect_headers(include_dir, args, Dict(), endswith("Python/Interop.h"))...,
        detect_headers(hlo_include_dir, args, Dict())...,
        detect_headers(sdy_include_dir, args, Dict())...,
        detect_headers(triton_include_dir, args, Dict())...,
        detect_headers(mosaic_tpu_include_dir, args, Dict())...,
        detect_headers(mosaic_gpu_include_dir, args, Dict())...,
        detect_headers(enzymexla_include_dir, args, Dict())...,
        detect_headers(enzymemlir_include_dir, args, Dict())...,
    ]
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

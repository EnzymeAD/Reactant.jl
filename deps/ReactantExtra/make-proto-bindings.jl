# Script to generate Julia bindings from proto files using ProtoBuf.jl
#
# This script is self-contained and will:
# 1. Run bazel to fetch and stage proto files from XLA/TSL/xprof
# 2. Extract the proto files to a staging directory
# 3. Generate Julia bindings using ProtoBuf.jl
#
# Usage:
#   julia --project=. make-proto-bindings.jl [--force] [output_dir]
#
# Options:
#   --force    - Force rebuild even if proto files already exist
#
# Arguments:
#   output_dir - Directory for generated Julia files (default: ../../src/proto)

using Pkg: Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using ProtoBuf

# Constants
const REACTANT_ROOT = dirname(dirname(@__DIR__))
const DEFAULT_OUTPUT_DIR = joinpath(REACTANT_ROOT, "src", "proto")

# Bazel command detection
const BAZEL_CMD = if !isnothing(Sys.which("bazelisk"))
    "bazelisk"
elseif !isnothing(Sys.which("bazel"))
    "bazel"
else
    nothing
end

"""
    ensure_proto_files_staged(; force::Bool=false) -> String

Build and stage proto files using bazel.
Returns the path to the staged proto files directory.

If `force=true`, always rebuilds even if files already exist.
"""
function ensure_proto_files_staged(; force::Bool=false)
    tar_file = joinpath(@__DIR__, "bazel-bin", "proto_files.tar")
    staging_dir = joinpath(@__DIR__, "proto_staging")

    # Check if we need to rebuild
    needs_build = force || !isfile(tar_file)
    needs_extract = force || !isdir(staging_dir) || isempty(readdir(staging_dir))

    if !needs_build && !needs_extract
        println("Using existing staged proto files at: $staging_dir")
        return staging_dir
    end

    # Build with bazel if needed
    if needs_build
        if isnothing(BAZEL_CMD)
            error("""
            bazel/bazelisk not found in PATH.

            Please install bazel or bazelisk to fetch proto files from XLA/TSL.
            """)
        end

        println("Building proto files with bazel...")
        run(
            Cmd(
                `$(BAZEL_CMD) build --repo_env HERMETIC_PYTHON_VERSION="3.10" --check_visibility=false --verbose_failures //:proto_files`;
                dir=@__DIR__,
            ),
        )

        if !isfile(tar_file)
            error("Bazel build completed but tar file not found at: $tar_file")
        end
    end

    # Extract the tar file
    if needs_extract || needs_build
        println("Extracting proto files to: $staging_dir")
        rm(staging_dir; force=true, recursive=true)
        mkpath(staging_dir)
        run(`tar -xf $tar_file -C $staging_dir`)
    end

    return staging_dir
end

"""
    find_proto_files(staging_dir::String) -> Vector{Tuple{String, String}}

Find all proto files in the staging directory.
Returns a list of (relative_path, absolute_path) tuples.
"""
function find_proto_files(staging_dir::String)
    proto_files = Tuple{String,String}[]

    for (root, dirs, files) in walkdir(staging_dir)
        for file in files
            if endswith(file, ".proto")
                abs_path = joinpath(root, file)
                rel_path = relpath(abs_path, staging_dir)
                push!(proto_files, (rel_path, abs_path))
            end
        end
    end

    return proto_files
end

"""
    generate_bindings(staging_dir::String, output_dir::String)

Generate Julia bindings from proto files using ProtoBuf.jl.
"""
function apply_parser_fixes!(staging_dir::String)
    # Define files that are known to cause parser errors
    # These often involve complex cross-references or Proto3 'optional'
    problematic = [
        "xla/xla.proto",
        "xla/pjrt/proto/compile_options.proto",
        "xla/backends/autotuner/backends.proto",
        "xla/stream_executor/sycl/oneapi_compute_capability.proto",
    ]

    # Types that frequently cause resolution issues when used unqualified in the same package
    types_to_qualify = [
        "HloModuleProto",
        "HloSnapshot",
        "ShapeProto",
        "LayoutProto",
        "ExecutionOptions",
        "DebugOptions",
        "PrecisionConfig",
        "CompilationEnvironmentsProto",
        "ProgramShapeProto",
        "DeviceAssignmentProto",
        "HloInstructionProto",
        "HloComputationProto",
        "OpSharding",
        "LiteralProto",
    ]

    println("\n  Applying parser fixes to problematic proto files...")

    for rel in problematic
        abs_path = joinpath(staging_dir, rel)
        if isfile(abs_path)
            content = read(abs_path, String)
            original_content = content

            # 1. Remove Proto3 'optional' keyword. 
            # ProtoBuf.jl 1.x's handwritten parser sometimes chokes on this in complex files.
            content = replace(content, r"(^|\n) *optional " => s"\1 ")

            # 2. Qualify common types in the xla package to help the parser.
            for t in types_to_qualify
                # Qualify if preceded by space/newline and NOT already qualified
                # and followed by space, semicolon or newline.
                # Example: " ShapeProto " -> " xla.ShapeProto "
                content = replace(
                    content,
                    Regex("(?<=[ \\n])(?!" * "xla\\.)" * t * "(?=[ ;\\n])") => "xla." * t,
                )
            end

            if content != original_content
                write(abs_path, content)
                println("    ✓ Patched $rel")
            end
        end
    end
end

function generate_bindings(staging_dir::String, output_dir::String)
    mkpath(output_dir)

    # Find all proto files
    all_proto_files = find_proto_files(staging_dir)

    if isempty(all_proto_files)
        @warn "No proto files found in $staging_dir"
        println("\nDirectory contents:")
        for f in readdir(staging_dir)
            println("  $f")
        end
        return nothing
    end

    println("\nFound $(length(all_proto_files)) proto files:")
    for (rel, _) in all_proto_files
        println("  - $rel")
    end

    println("\nGenerating Julia bindings...")

    # Apply fixes to problematic files before parsing
    apply_parser_fixes!(staging_dir)

    # The include paths for proto resolution
    include_paths = [staging_dir]

    # Collect all proto file paths (relative to staging_dir)
    # CRITICAL: Exclude google/protobuf files from the list of files to generate
    # bindings for. ProtoBuf.jl has issues generating bindings for descriptor.proto,
    # and well-known types should be treated as imports only.
    proto_rel_paths = String[]
    for (rel, _) in all_proto_files
        if !startswith(rel, "google/")
            push!(proto_rel_paths, rel)
        end
    end

    # DEBUG: Try to identify the problematic file by processing them individually first
    # into a dummy directory. This helps isolate which file triggers the ReferencedType MethodError.
    println("\n" * "="^40)
    println("DEBUG: Identifying problematic files")
    println("="^40)
    problematic_files = String[]
    mktempdir() do tmp_out
        for (i, rel) in enumerate(proto_rel_paths)
            print("[$i/$(length(proto_rel_paths))] Testing $rel ... ")
            try
                ProtoBuf.protojl(
                    [rel],
                    include_paths,
                    tmp_out;
                    always_use_modules=true,
                    parametrize_oneofs=false,
                    add_kwarg_constructors=false,
                )
                println("✓")
            catch e
                println("✖")
                if e isa MethodError && contains(string(e), "ReferencedType")
                    println("      --> TRIGGERED THE ReferencedType MethodError")
                else
                    println("      --> Failed with: $(typeof(e))")
                end
                push!(problematic_files, rel)
            end
        end
    end
    println("="^40 * "\n")

    if !isempty(problematic_files)
        println(
            "Excluding $(length(problematic_files)) problematic files: $(join(problematic_files, ", "))",
        )
        proto_rel_paths = filter(p -> p ∉ problematic_files, proto_rel_paths)
    end

    if isempty(proto_rel_paths)
        @warn "No valid proto files left to process!"
        return nothing
    end

    # Generate bindings for the remaining files in a single call
    println("Processing $(length(proto_rel_paths)) remaining proto files together...")
    try
        ProtoBuf.protojl(
            proto_rel_paths,
            include_paths,
            output_dir;
            always_use_modules=true,
            parametrize_oneofs=false,
            add_kwarg_constructors=false,
        )
        println("    ✓ Generated bindings successfully")
    catch e
        @warn "Failed to generate bindings in batch even after exclusions" exception = (
            e, catch_backtrace()
        )
        return nothing
    end

    # Remove headers from generated files to minimize diffs
    remove_proto_headers(output_dir)

    # Create a main module that includes all generated files
    write_main_module(output_dir, proto_rel_paths)

    return proto_rel_paths
end

"""
    remove_proto_headers(output_dir::String)

Remove the first 3 lines of generated Julia files if they are comments or empty.
This helps minimize diffs by removing the autogeneration timestamp.
"""
function remove_proto_headers(output_dir::String)
    println("\n  Removing headers from generated files...")
    for (root, dirs, files) in walkdir(output_dir)
        for file in files
            endswith(file, ".jl") || continue
            file == "Proto.jl" && continue

            path = joinpath(root, file)
            lines = readlines(path)

            if (
                length(lines) >= 3 &&
                startswith(lines[1], "# Autogenerated") &&
                startswith(lines[2], "# original file") &&
                (isempty(strip(lines[3])) || startswith(lines[3], "#"))
            )
                open(path, "w") do io
                    for i in 4:length(lines)
                        println(io, lines[i])
                    end
                end
            end
        end
    end
end

"""
    write_main_module(output_dir::String, generated::Vector{String})

Create a main Proto module file that exports all generated types.
"""
function write_main_module(output_dir::String, generated::Vector{String})
    # Find all generated Julia files
    jl_files = [
        file for file in readdir(output_dir) if endswith(file, ".jl") && file != "Proto.jl"
    ]

    # Also check for generated directories (ProtoBuf.jl sometimes creates module dirs)
    for dir in readdir(output_dir)
        dir_path = joinpath(output_dir, dir)
        if isdir(dir_path) && isfile(joinpath(dir_path, "$dir.jl"))
            push!(jl_files, joinpath(dir, "$dir.jl"))
        end
    end

    if isempty(jl_files)
        println("\n⚠ No Julia files generated")
        return nothing
    end

    # Generate the main module file
    main_file = joinpath(output_dir, "Proto.jl")

    content = """
    # Auto-generated file - DO NOT EDIT
    # Generated by make-proto-bindings.jl from bazel-staged proto files
    #
    # This module provides Julia bindings for XLA/TSL proto messages.
    # Regenerate with:
    #   cd deps/ReactantExtra
    #   julia --project=. make-proto-bindings.jl --force

    module Proto

    using ProtoBuf

    # Generated proto bindings
    """

    for file in jl_files
        content *= "include(\"$file\")\n"
    end

    content *= """

    end # module Proto
    """

    write(main_file, content)
    println("\n✓ Created main module: $main_file")
    return println("  Includes $(length(jl_files)) generated files")
end

"""
    main(; output_dir::String=DEFAULT_OUTPUT_DIR, force::Bool=false)

Main entry point for proto binding generation.
"""
function main(; output_dir::String=DEFAULT_OUTPUT_DIR, force::Bool=false)
    println("="^60)
    println("Proto Bindings Generator for Reactant.jl")
    println("="^60)

    # Stage proto files via bazel
    println("\n[1/3] Staging proto files via bazel...")
    staging_dir = ensure_proto_files_staged(; force=force)

    # Generate Julia bindings
    println("\n[2/3] Generating Julia bindings...")
    generated = generate_bindings(staging_dir, output_dir)

    # Summary
    println("\n[3/3] Summary")
    println("="^60)
    if !isnothing(generated) && !isempty(generated)
        println("✓ Generated $(length(generated)) proto bindings")
        println("  Output directory: $output_dir")
    else
        println("⚠ No bindings generated")
    end
    return println("="^60)
end

# Main entry point
if (abspath(PROGRAM_FILE) == @__FILE__) || isinteractive()
    # Parse command line arguments
    force = "--force" in ARGS || "-f" in ARGS

    # Filter out flags to get positional args
    positional_args = filter(a -> !startswith(a, "-"), ARGS)
    output_dir = isempty(positional_args) ? DEFAULT_OUTPUT_DIR : positional_args[1]

    main(; output_dir=output_dir, force=force)
end

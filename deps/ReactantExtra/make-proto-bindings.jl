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

    # The include paths for proto resolution
    include_paths = [staging_dir, joinpath(staging_dir, "google", "protobuf")]

    # Collect all proto file paths (relative to staging_dir)
    proto_rel_paths = [rel for (rel, _) in all_proto_files]

    # Generate bindings for ALL proto files in a single call
    # This is critical - calling protojl separately for each file will overwrite
    # the module files (like xla.jl) since each call generates its own module structure
    println("\n  Processing all $(length(proto_rel_paths)) proto files together...")
    try
        ProtoBuf.protojl(
            proto_rel_paths,  # Pass all files as a vector
            include_paths,
            output_dir;
            always_use_modules=true,
            parametrize_oneofs=false,
            add_kwarg_constructors=false,
        )
        println("    ✓ Generated bindings for $(length(proto_rel_paths)) proto files")
    catch e
        @warn "Failed to generate bindings" exception = (e, catch_backtrace())
        return nothing
    end

    # Remove headers from generated files to minimize diffs
    remove_proto_headers(output_dir)

    # Convert all structs to mutable structs
    make_structs_mutable(output_dir)

    # Convert large structs to use dict-based storage
    convert_large_structs_to_dict(output_dir)

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
    make_structs_mutable(output_dir::String)

Convert all `struct` declarations to `mutable struct` in generated Julia files.
This allows proto message types to be modified after construction.
"""
function make_structs_mutable(output_dir::String)
    println("\n  Converting structs to mutable structs...")
    for (root, dirs, files) in walkdir(output_dir)
        for file in files
            endswith(file, ".jl") || continue
            file == "Proto.jl" && continue

            path = joinpath(root, file)
            content = read(path, String)

            # Replace "struct " with "mutable struct " but avoid double-replacement
            # Use word boundary to avoid replacing inside other words
            new_content = replace(content, r"\bstruct\s+" => "mutable struct ")

            if content != new_content
                write(path, new_content)
            end
        end
    end
end

"""
    convert_large_structs_to_dict(output_dir::String; min_fields::Int=8)

Convert structs with more than `min_fields` to use a single Dict{Symbol,Any} storage.
This reduces compile time and memory usage for large proto structs.

Also comments out `PB.default_values` and `PB.field_numbers` for these structs,
and updates the constructor to use kwargs that populate the dict.
"""
function convert_large_structs_to_dict(output_dir::String; min_fields::Int=8)
    println("\n  Converting large structs to dict-based storage...")

    for (root, dirs, files) in walkdir(output_dir)
        for file in files
            endswith(file, ".jl") || continue
            file == "Proto.jl" && continue

            path = joinpath(root, file)
            content = read(path, String)
            original_content = content

            # Track which structs we convert
            converted_structs = String[]

            # Find all struct definitions with their fields
            # Pattern: mutable struct Name\n    field1::Type1\n    field2::Type2\n...\nend
            struct_pattern = r"(mutable struct\s+)(var\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_]*)\s*\n((?:\s+[a-z_][a-z0-9_]*::[^\n]+\n)+)end"

            for m in eachmatch(struct_pattern, content)
                struct_prefix = m.captures[1]
                struct_name = m.captures[2]
                fields_block = m.captures[3]

                # Parse field names and types
                field_matches = collect(
                    eachmatch(r"\s+([a-z_][a-z0-9_]*)::([^\n]+)", fields_block)
                )
                num_fields = length(field_matches)

                if num_fields > min_fields
                    push!(converted_structs, struct_name)
                    field_names = [fm.captures[1] for fm in field_matches]
                    field_types = [strip(fm.captures[2]) for fm in field_matches]

                    # Build default values dict for getproperty
                    defaults_entries = ["    :$(fn) => $(get_default_for_type(ft))" for (fn, ft) in zip(field_names, field_types)]
                    # Sanitize struct name for use as variable name
                    # Handle var"Foo.Bar" style names by removing var, quotes, and replacing dots
                    safe_name = struct_name
                    safe_name = replace(safe_name, "var\"" => "")
                    safe_name = replace(safe_name, "\"" => "")
                    safe_name = replace(safe_name, "." => "_")

                    # Build new struct with dict storage
                    new_struct = """$(struct_prefix)$(struct_name)
    __data::Dict{Symbol,Any}
end

# Default values for $(struct_name) fields
const _$(safe_name)_defaults = Dict{Symbol,Any}(
$(join(defaults_entries, ",\n"))
)

# Keyword constructor for $(struct_name)
function $(struct_name)(; kwargs...)
    __data = Dict{Symbol,Any}(kwargs)
    return $(struct_name)(__data)
end

# Field accessors for $(struct_name)
function Base.getproperty(x::$(struct_name), s::Symbol)
    s === :__data && return getfield(x, :__data)
    d = getfield(x, :__data)
    return get(d, s, get(_$(safe_name)_defaults, s, nothing))
end
function Base.setproperty!(x::$(struct_name), s::Symbol, v)
    getfield(x, :__data)[s] = v
end
Base.propertynames(::$(struct_name)) = ($(join([":$(fn)" for fn in field_names], ", ")),)"""

                    # Replace the struct definition
                    content = replace(content, m.match => new_struct)

                    # Also need to replace positional constructor calls in PB.decode
                    # Pattern: return StructName(arg1, arg2, ..., argN)
                    # Find the return statement for this struct type
                    # The args may contain [], so we need to be careful
                    constructor_pattern = Regex(
                        "return $(replace(struct_name, "\"" => "\\\""))\\(([^)]+)\\)"
                    )
                    for cm in eachmatch(constructor_pattern, content)
                        args_str = cm.captures[1]
                        # Split by comma, but be careful with nested brackets
                        args = split_args(args_str)
                        if length(args) == num_fields
                            # Build keyword constructor call
                            kwargs = join(["$(fn)=$(strip(arg))" for (fn, arg) in zip(field_names, args)], ", ")
                            new_call = "return $(struct_name)(; $(kwargs))"
                            content = replace(content, cm.match => new_call)
                        end
                    end
                end
            end

            # Comment out PB.default_values and PB.field_numbers only for converted structs
            for struct_name in converted_structs
                # Escape special regex characters in struct name (for var"..." names)
                escaped_name = replace(struct_name, "\"" => "\\\"")
                escaped_name = replace(escaped_name, "." => "\\.")

                # Comment out PB.default_values for this struct
                dv_pattern = Regex(
                    "^(PB\\.default_values\\(::Type\\{$(escaped_name)\\}\\).*?)\$", "m"
                )
                content = replace(content, dv_pattern => s"# \1")

                # Comment out PB.field_numbers for this struct
                fn_pattern = Regex(
                    "^(PB\\.field_numbers\\(::Type\\{$(escaped_name)\\}\\).*?)\$", "m"
                )
                content = replace(content, fn_pattern => s"# \1")

                # Comment out PB.reserved_fields for this struct
                rf_pattern = Regex(
                    "^(PB\\.reserved_fields\\(::Type\\{$(escaped_name)\\}\\).*?)\$", "m"
                )
                content = replace(content, rf_pattern => s"# \1")
            end

            if content != original_content
                write(path, content)
            end
        end
    end
end

# Helper function to split arguments by comma, respecting nested brackets
function split_args(s::AbstractString)
    args = String[]
    current = IOBuffer()
    depth = 0
    for c in s
        if c in ('(', '[', '{')
            depth += 1
            write(current, c)
        elseif c in (')', ']', '}')
            depth -= 1
            write(current, c)
        elseif c == ',' && depth == 0
            push!(args, String(take!(current)))
        else
            write(current, c)
        end
    end
    remaining = String(take!(current))
    if !isempty(strip(remaining))
        push!(args, remaining)
    end
    return args
end

# Helper function to get a default value expression for a type
function get_default_for_type(type_str::AbstractString)
    type_str = strip(type_str)
    if startswith(type_str, "Vector{")
        return "$(type_str)()"
    elseif startswith(type_str, "Dict{")
        return "$(type_str)()"
    elseif startswith(type_str, "Union{Nothing,")
        return "nothing"
    elseif type_str == "String"
        return "\"\""
    elseif type_str == "Bool"
        return "false"
    elseif type_str in ("Int32", "Int64", "UInt32", "UInt64", "Float32", "Float64")
        return "zero($(type_str))"
    elseif endswith(type_str, ".T")  # Enum types
        return "nothing"  # Can't easily determine default enum value
    else
        return "nothing"
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
if abspath(PROGRAM_FILE) == @__FILE__
    # Parse command line arguments
    force = "--force" in ARGS || "-f" in ARGS

    # Filter out flags to get positional args
    positional_args = filter(a -> !startswith(a, "-"), ARGS)
    output_dir = isempty(positional_args) ? DEFAULT_OUTPUT_DIR : positional_args[1]

    main(; output_dir=output_dir, force=force)
end

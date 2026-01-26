#!/usr/bin/env julia
"""
Script to update TODO/FIXME/XXX comments to reference the meta tracking issue #2224.

This is a pragmatic first step. More specific issues can be created later and
references can be updated accordingly.
"""

function update_todo_in_file(filepath::String, line_num::Int, old_line::String, issue_number::Int)
    lines = readlines(filepath)
    
    if line_num > length(lines)
        @warn "Line number $line_num out of range in $filepath"
        return false, lines
    end
    
    # Check if the line still matches (approximately)
    current_line = lines[line_num]
    
    # Check if issue reference already exists
    if occursin(r"#\d+", current_line) || occursin(r"github\.com/.*/issues/\d+", current_line)
        return false, lines
    end
    
    # Check if this is actually a TODO/FIXME/XXX line
    if !occursin(r"\b(TODO|FIXME|XXX|HACK)\b", current_line)
        return false, lines
    end
    
    # Add issue reference at the end of the line (before newline)
    new_line = rstrip(current_line) * " (see #$issue_number)\n"
    lines[line_num] = new_line
    
    return true, lines
end

function process_file(filepath::String, line_numbers::Vector{Int}, issue_number::Int)
    # Read all lines
    lines = readlines(filepath; keep=false)
    modified = false
    
    # Process each line
    for line_num in sort(line_numbers; rev=true)  # Process from bottom to top
        if line_num > length(lines)
            continue
        end
        
        current_line = lines[line_num]
        
        # Check if issue reference already exists
        if occursin(r"#\d+", current_line) || occursin(r"github\.com/.*/issues/\d+", current_line)
            continue
        end
        
        # Check if this is actually a TODO/FIXME/XXX line
        if !occursin(r"\b(TODO|FIXME|XXX|HACK)\b", current_line)
            continue
        end
        
        # Add issue reference at the end of the line
        lines[line_num] = rstrip(current_line) * " (see #$issue_number)"
        modified = true
    end
    
    if modified
        # Write back the file
        open(filepath, "w") do io
            for line in lines
                println(io, line)
            end
        end
        return true
    end
    
    return false
end

function main()
    root_dir = dirname(dirname(@__FILE__))
    issue_number = 2224  # The meta-issue for TODO tracking
    
    # Load the todos CSV
    todos_file = joinpath(root_dir, "scripts", "todos.csv")
    if !isfile(todos_file)
        error("TODOs file not found. Run collect_todos.jl first.")
    end
    
    # Group TODOs by file
    file_todos = Dict{String, Vector{Int}}()
    
    todos_lines = readlines(todos_file)
    for line in todos_lines[2:end]  # Skip header
        if isempty(strip(line))
            continue
        end
        
        # Parse CSV line
        parts = split(line, ','; limit=6)
        if length(parts) < 3
            continue
        end
        
        file = strip(parts[2], '"')
        line_num = parse(Int, strip(parts[3]))
        
        if !haskey(file_todos, file)
            file_todos[file] = Int[]
        end
        push!(file_todos[file], line_num)
    end
    
    println("Processing $(length(file_todos)) files...")
    
    updated_count = 0
    for (file, line_nums) in file_todos
        filepath = joinpath(root_dir, file)
        
        if !isfile(filepath)
            @warn "File not found: $filepath"
            continue
        end
        
        if process_file(filepath, line_nums, issue_number)
            updated_count += 1
            println("✓ Updated $file ($(length(line_nums)) TODOs)")
        end
    end
    
    println("\nSummary:")
    println("  Files updated: $updated_count")
    println("  All TODOs now reference issue #$issue_number")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
